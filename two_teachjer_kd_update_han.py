# -*- coding: utf-8 -*-
"""Dual-teacher distillation (RSAGE + meta-path teacher) for graph-free MLP student.

This script trains the relation-aware RSAGE teacher, a meta-path aware HAN-style teacher,
then distils their knowledge (relation geometry + meta-path semantics + logits) into a
simple MLP student that does not need the graph at inference.

The implementation integrates:
- relation-level adapters/distillation from `relation_distill_only.py`
- meta-path aware semantics inspired by `han_only.py` / `han_test.py`
- optional MetaPath2Vec positional encodings for the student (unchanged interface)

Datasets: TMDB, ArXiv, DBLP, IMDB (see `dataloader.py`).
"""

import os
import math
import json
import time
import copy
import random
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, HANConv
import torch_geometric.transforms as T
from torch_geometric.nn.models import MetaPath2Vec
from torch_sparse import SparseTensor
import torch_sparse

from dataloader import load_data

# =========================
# Benchmark utilities
# =========================

def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_forward(fn, warmup: int, runs: int, device: torch.device) -> Tuple[float, float]:
    _sync_if_cuda(device)
    for _ in range(max(0, warmup)):
        fn()
        _sync_if_cuda(device)

    timings = []
    for _ in range(max(0, runs)):
        start = time.perf_counter()
        fn()
        _sync_if_cuda(device)
        timings.append(time.perf_counter() - start)

    if not timings:
        return 0.0, 0.0

    t = np.array(timings, dtype=np.float64)
    return float(t.mean()), float(t.std())

# =========================
# Utilities & Reliability
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed}")


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    if idx.numel() == 0:
        return 0.0
    pred = logits[idx].argmax(dim=-1)
    return (pred == y[idx]).float().mean().item()


def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 1.0,
          reduce: bool = True) -> torch.Tensor:
    ps = F.log_softmax(student_logits / T, dim=-1)
    pt = F.log_softmax(teacher_logits / T, dim=-1)
    kl = torch.sum(torch.exp(pt) * (pt - ps), dim=-1) * (T * T)
    return kl.mean() if reduce else kl


def js_divergence(prob_a: torch.Tensor, prob_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = 0.5 * (prob_a + prob_b)
    log_a = torch.log(prob_a.clamp_min(eps))
    log_b = torch.log(prob_b.clamp_min(eps))
    log_m = torch.log(m.clamp_min(eps))
    kl_am = torch.sum(prob_a * (log_a - log_m), dim=-1)
    kl_bm = torch.sum(prob_b * (log_b - log_m), dim=-1)
    return 0.5 * (kl_am + kl_bm)


def _call_teacher_forward(teacher: nn.Module, hetero: HeteroData,
                          feats_override: Optional[torch.Tensor] = None):
    if feats_override is not None:
        try:
            return teacher(hetero, feats_override)
        except TypeError:
            return teacher(hetero)
    else:
        try:
            return teacher(hetero)
        except TypeError:
            return teacher(hetero, None)


@torch.no_grad()
def compute_logits_with_reliability(teacher: nn.Module,
                                    hetero: HeteroData,
                                    category: str,
                                    device: torch.device,
                                    feats_override: Optional[torch.Tensor] = None,
                                    noise_std: float = 0.05,
                                    alpha: float = 0.5):
    teacher.eval()

    hetero_clean = hetero.clone().to(device)
    logits = _call_teacher_forward(teacher, hetero_clean, feats_override)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)

    if feats_override is not None:
        noisy_feats = feats_override + noise_std * torch.randn_like(feats_override)
        logits_noise = _call_teacher_forward(teacher, hetero_clean, noisy_feats)
    else:
        hetero_noise = hetero.clone().to(device)
        hetero_noise[category].x = hetero_noise[category].x + noise_std * torch.randn_like(hetero_noise[category].x)
        logits_noise = _call_teacher_forward(teacher, hetero_noise, None)

    if isinstance(logits_noise, (tuple, list)):
        logits_noise = logits_noise[0]
    probs_noise = torch.softmax(logits_noise, dim=-1)

    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    ent = (ent / math.log(probs.size(1))).clamp(0, 1)
    conf = 1.0 - ent

    stab = torch.exp(- (probs - probs_noise).abs().sum(dim=-1) / 0.5)
    reliability = (alpha * conf + (1 - alpha) * stab).clamp(0, 1)

    return logits.detach(), probs.detach(), reliability.detach()
# =========================
# Meta-path utilities
# =========================

def metapath_to_str(mp: List[Tuple[str, str, str]]) -> str:
    return "__".join([f"{s}-{r}-{d}" for (s, r, d) in mp])


def parse_metapath_name_sequences(hetero: HeteroData,
                                  category: str,
                                  rel_name_seqs: List[List[str]]) -> List[List[Tuple[str, str, str]]]:
    typed = []
    etypes = list(hetero.edge_types)
    for rel_names in rel_name_seqs:
        cur_src = category
        mp_typed: List[Tuple[str, str, str]] = []
        ok = True
        for rel in rel_names:
            match = None
            for (s, r, d) in etypes:
                if s == cur_src and r == rel:
                    match = (s, r, d)
                    break
            if match is None:
                ok = False
                break
            mp_typed.append(match)
            cur_src = match[2]
        if ok and mp_typed and mp_typed[-1][2] == category:
            typed.append(mp_typed)
    return typed


def ensure_metapaths_to_category(metapaths: List[List[Tuple[str, str, str]]], category: str) -> List[List[Tuple[str, str, str]]]:
    filtered = []
    for mp in metapaths:
        if mp and mp[-1][2] == category:
            filtered.append(mp)
    return filtered


def build_sparse_relation(hetero: HeteroData,
                          et: Tuple[str, str, str],
                          device: torch.device,
                          normalize: bool = True) -> Tuple[SparseTensor, SparseTensor]:
    src, _, dst = et
    edge_index = hetero[et].edge_index.to(device)
    num_src = hetero[src].num_nodes
    num_dst = hetero[dst].num_nodes
    value = torch.ones(edge_index.size(1), device=device, dtype=torch.float32)
    A = SparseTensor(row=edge_index[1], col=edge_index[0], value=value,
                     sparse_sizes=(num_dst, num_src)).coalesce()
    if not normalize:
        return A, A
    deg = A.sum(dim=1).to_dense().clamp_min(1.0)
    row, col, val = A.coo()
    norm_val = val / deg[row]
    A_norm = SparseTensor(row=row, col=col, value=norm_val,
                          sparse_sizes=A.sparse_sizes()).coalesce()
    return A_norm, A


def build_metapath_operators(hetero: HeteroData,
                             metapaths: List[List[Tuple[str, str, str]]],
                             device: torch.device) -> Dict[str, Dict[str, List[SparseTensor]]]:
    ops: Dict[str, Dict[str, List[SparseTensor]]] = {}
    for mp in metapaths:
        key = metapath_to_str(mp)
        norm_ops: List[SparseTensor] = []
        raw_ops: List[SparseTensor] = []
        for et in mp:
            A_norm, A_raw = build_sparse_relation(hetero, et, device)
            norm_ops.append(A_norm)
            raw_ops.append(A_raw)
        chain = norm_ops[0]
        for nxt in norm_ops[1:]:
            chain = torch_sparse.matmul(nxt, chain)
        ops[key] = {
            'typed': mp,
            'norm_ops': norm_ops,
            'raw_ops': raw_ops,
            'chain': chain,
        }
    return ops
# =========================
# Student components (MLP + adapters)
# =========================

class MLPClassifier(nn.Module):
    def __init__(self, d_in: int, n_classes: int,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        dims = [d_in] + [hidden] * max(0, num_layers - 1) + [n_classes]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelKDAdapter(nn.Module):
    """Train-time relation adapter that mirrors the RSAGE taps."""

    def __init__(self, node_dims: Dict[str, int], d_rel: int):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(node_dims[nt], d_rel, bias=False)
                                   for nt in node_dims})

    @torch.no_grad()
    def _build_adj(self, hetero: HeteroData, et: Tuple[str, str, str], device: torch.device):
        s, _, d = et
        edge_index = hetero[et].edge_index.to(device)
        num_s = hetero[s].num_nodes
        num_d = hetero[d].num_nodes
        value = torch.ones(edge_index.size(1), device=device)
        A = SparseTensor(row=edge_index[1], col=edge_index[0], value=value,
                         sparse_sizes=(num_d, num_s)).coalesce()
        deg = A.sum(dim=1).to_dense().clamp_min(1.0)
        row, col, val = A.coo()
        norm_val = val / deg[row]
        A_norm = SparseTensor(row=row, col=col, value=norm_val,
                              sparse_sizes=(num_d, num_s)).coalesce()
        return A_norm, deg

    def forward(self, hetero: HeteroData, etypes: List[Tuple[str, str, str]],
                node_overrides: Optional[Dict[str, torch.Tensor]] = None,
                device: Optional[torch.device] = None):
        device = device or hetero[etypes[0][0]].x.device
        x_proj = {}
        for nt in hetero.node_types:
            feats = node_overrides[nt] if node_overrides and nt in node_overrides else hetero[nt].x
            x_proj[nt] = self.proj[nt](feats.to(device))

        rel_embs: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        for et in etypes:
            s, _, d = et
            A_norm, deg = self._build_adj(hetero, et, device)
            dst = A_norm.matmul(x_proj[s])
            rel_embs[et] = {
                'dst': dst,
                'src_in': x_proj[s],
                'deg_dst': deg
            }
        return rel_embs


class MetaPathAdapter(nn.Module):
    def __init__(self, node_dims: Dict[str, int],
                 metapaths: List[List[Tuple[str, str, str]]],
                 d_mp: int,
                 ops_template: Dict[str, Dict[str, List[SparseTensor]]]):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(node_dims[nt], d_mp, bias=False)
                                   for nt in node_dims})
        self.metapath_keys = [metapath_to_str(mp) for mp in metapaths]
        self.metapaths = {metapath_to_str(mp): mp for mp in metapaths}
        self.ops_template = ops_template
        self.d_mp = d_mp

    def forward(self, hetero: HeteroData,
                node_overrides: Optional[Dict[str, torch.Tensor]] = None,
                device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        if len(self.metapath_keys) == 0:
            return {}
        device = device or next(iter(self.proj.values())).weight.device
        x_proj = {}
        for nt in hetero.node_types:
            feats = node_overrides[nt] if node_overrides and nt in node_overrides else hetero[nt].x
            x_proj[nt] = self.proj[nt](feats.to(device))

        mp_embs: Dict[str, torch.Tensor] = {}
        for key in self.metapath_keys:
            mp = self.metapaths[key]
            ops_info = self.ops_template[key]
            z = x_proj[mp[0][0]]
            for op in ops_info['norm_ops']:
                op_dev = op.to(device)
                z = op_dev.matmul(z)
            mp_embs[key] = z
        return mp_embs


class SemanticHead(nn.Module):
    """Lightweight semantic attention head to predict meta-path importance."""

    def __init__(self, d_mp: int, hidden: int = 128):
        super().__init__()
        self.lin = nn.Linear(d_mp, hidden)
        self.attn = nn.Linear(hidden, 1)

    def forward(self, mp_embs: List[torch.Tensor]) -> torch.Tensor:
        if len(mp_embs) == 0:
            raise ValueError("SemanticHead requires at least one meta-path embedding")
        stacked = torch.stack(mp_embs, dim=1)  # [N, M, d]
        scores = torch.tanh(self.lin(stacked))
        scores = self.attn(scores).squeeze(-1)
        beta = torch.softmax(scores, dim=1)
        return beta
# =========================
# Teacher-R: RSAGE (unchanged)
# =========================

class RSAGE_Hetero(nn.Module):
    def __init__(self,
                 etypes: List[Tuple[str, str, str]],
                 in_dim: int,
                 hid_dim: int,
                 num_classes: int,
                 category: str,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 norm_type: str = "none",
                 node_type_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.category = category
        self.etypes = list(etypes)
        self.num_layers = int(max(1, num_layers))
        self.hid_dim = int(hid_dim)
        self.num_classes = int(num_classes)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.node_type_dims = node_type_dims or {}
        self.projectors = nn.ModuleDict()
        if self.node_type_dims:
            for nt, dim in self.node_type_dims.items():
                if dim != in_dim:
                    self.projectors[nt] = nn.Linear(dim, in_dim, bias=False)
                else:
                    self.projectors[nt] = nn.Identity()

        def conv_dict(in_channels, out_channels):
            return {et: SAGEConv(in_channels, out_channels, aggr='mean') for et in self.etypes}

        if self.num_layers == 1:
            self.layers.append(HeteroConv(conv_dict(in_dim, num_classes), aggr='mean'))
        else:
            self.layers.append(HeteroConv(conv_dict(in_dim, hid_dim), aggr='mean'))
            for _ in range(self.num_layers - 2):
                self.layers.append(HeteroConv(conv_dict(hid_dim, hid_dim), aggr='mean'))
            self.layers.append(HeteroConv(conv_dict(hid_dim, num_classes), aggr='mean'))

        self.use_norm = norm_type in {"batch", "layer"}
        if self.use_norm and self.num_layers > 1:
            norm_cls = nn.BatchNorm1d if norm_type == "batch" else nn.LayerNorm
            for _ in range(self.num_layers - 1):
                self.norms.append(norm_cls(hid_dim))

    def _apply_each(self, xdict: Dict[str, torch.Tensor], fn):
        return {k: fn(v) for k, v in xdict.items()}

    def forward(self, data: HeteroData, feats_override: Optional[torch.Tensor] = None):
        x_dict = {nt: data[nt].x for nt in data.node_types}
        if feats_override is not None:
            x_dict[self.category] = feats_override
        if self.projectors:
            x_dict = {nt: self.projectors[nt](feat) if nt in self.projectors else feat
                      for nt, feat in x_dict.items()}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        h = x_dict
        for l, layer in enumerate(self.layers):
            h = layer(h, edge_index_dict)
            h = {nt: val.view(val.shape[0], -1) for nt, val in h.items()}
            if l != len(self.layers) - 1:
                if self.use_norm and l < len(self.norms):
                    h = {nt: self.norms[l](val) for nt, val in h.items()}
                h = self._apply_each(h, self.activation)
                h = self._apply_each(h, self.dropout)
        return h[self.category]

    @torch.no_grad()
    def forward_with_relation_taps(self, data: HeteroData,
                                   feats_override: Optional[torch.Tensor] = None):
        x_dict = {nt: data[nt].x for nt in data.node_types}
        if feats_override is not None:
            x_dict[self.category] = feats_override
        if self.projectors:
            x_dict = {nt: self.projectors[nt](feat) if nt in self.projectors else feat
                      for nt, feat in x_dict.items()}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        h = x_dict
        for l, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index_dict)
            h = {nt: val.view(val.shape[0], -1) for nt, val in h.items()}
            if self.use_norm and l < len(self.norms):
                h = {nt: self.norms[l](val) for nt, val in h.items()}
            h = self._apply_each(h, self.activation)
            h = self._apply_each(h, self.dropout)
        h_in = {nt: val.view(val.shape[0], -1) for nt, val in h.items()}

        taps: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        last_layer: HeteroConv = self.layers[-1]
        for et, conv in last_layer.convs.items():
            src, _, dst = et
            ei = edge_index_dict[et]
            emb_dst = conv((h_in[src], h_in[dst]), ei)
            taps[et] = {
                'dst': emb_dst.view(emb_dst.shape[0], -1),
                'src_in': h_in[src]
            }
        h_out = last_layer(h_in, edge_index_dict)
        h_out = {nt: val.view(val.shape[0], -1) for nt, val in h_out.items()}
        return h_out[self.category], taps
# =========================
# Teacher-M: Meta-path semantic teacher (HAN-inspired)
# =========================

class MetaPathTeacher(nn.Module):
    """HAN-based meta-path teacher with cached AddMetaPaths graph.

    This module mirrors the standalone HAN examples (`han_dblp.py`, `han_imdb.py`).
    We materialize the meta-path-induced graph once on the CPU, instantiate a
    HANConv stack on top of it, and simply refresh node features from the input
    `HeteroData` on every forward. This avoids repeatedly running the transform
    at train time (which was both slow and numerically brittle) while keeping
    the KD-facing API identical to the legacy teacher.
    """

    def __init__(
        self,
        node_dims: Dict[str, int],
        category: str,
        metapaths: List[List[Tuple[str, str, str]]],
        ops_template: Dict[str, Dict[str, List[SparseTensor]]],
        d_hid: int,
        num_classes: int,
        semantic_hidden: int = 128,  # kept for interface compatibility (unused)
        dropout: float = 0.6,
        heads: int = 8,
        drop_orig_edge_types: bool = True,
        drop_unconnected_node_types: bool = True,
    ) -> None:
        super().__init__()
        self.category = category
        self.node_dims = dict(node_dims)
        self.metapath_keys = [metapath_to_str(mp) for mp in metapaths]
        self.metapaths = {metapath_to_str(mp): mp for mp in metapaths}
        self.ops_template = ops_template

        self.d_hid = int(d_hid)
        self.num_classes = int(num_classes)
        self.dropout_p = float(dropout)
        self.heads = int(heads)

        self.tail_align = nn.Linear(self.node_dims[category], self.d_hid, bias=False)
        self.delta_projector = nn.Linear(self.d_hid, self.d_hid, bias=False)
        self.delta_projector_student = nn.Linear(self.d_hid, self.d_hid, bias=False)

        self.lin = nn.Linear(self.d_hid, self.num_classes)
        self.act = nn.ELU()
        self.dropout = nn.Dropout(self.dropout_p)

        self.node_proj = nn.ModuleDict(
            {nt: nn.Linear(self.node_dims[nt], self.d_hid, bias=False) for nt in self.node_dims}
        )

        self._mp_pairs: List[List[Tuple[str, str]]] = [
            [(s, d) for (s, _r, d) in mp] for mp in metapaths
        ]
        # 将长度为1的元路径与长度>=2的元路径分开处理
        self._mp_pairs_long: List[List[Tuple[str, str]]] = [p for p in self._mp_pairs if len(p) >= 2]
        self._mp_pairs_single: List[List[Tuple[str, str]]] = [p for p in self._mp_pairs if len(p) == 1]
        # 记录单跳元路径对应的typed三元组（用于后续手动添加）
        self._single_typed: List[List[Tuple[str, str, str]]] = [
            self.metapaths[key] for key in self.metapath_keys if len(self.metapaths[key]) == 1
        ]

        self._drop_orig_edge_types = bool(drop_orig_edge_types)
        self._drop_unconnected_node_types = bool(drop_unconnected_node_types)

        self._meta_transform = None
        if len(self._mp_pairs_long) > 0:
            self._meta_transform = T.AddMetaPaths(
                metapaths=self._mp_pairs_long,
                drop_orig_edge_types=self._drop_orig_edge_types,
                drop_unconnected_node_types=self._drop_unconnected_node_types,
            )

        self.han_conv1: Optional[HANConv] = None
        self.han_conv2: Optional[HANConv] = None

        self._meta_graph_cpu: Optional[HeteroData] = None
        self._meta_metadata: Optional[Tuple[List[str], List[Tuple[str, str, str]]]] = None

    # ------------------------------------------------------------------
    # Meta-graph construction helpers
    # ------------------------------------------------------------------
    def _ensure_required_pairs(self, data: HeteroData) -> None:
        required: Set[Tuple[str, str]] = set()
        for path in self._mp_pairs:
            required.update(path)
        for src, dst in required:
            if any(s == src and d == dst for (s, _, d) in data.edge_types):
                continue
            reverse = None
            for (s, rel, d) in data.edge_types:
                if s == dst and d == src:
                    reverse = (s, rel, d)
                    break
            if reverse is None:
                raise RuntimeError(
                    f"Meta-path requires relation {src}->{dst} but neither direct nor reverse edge exists."
                )
            edge_index = data[reverse].edge_index.flip(0).contiguous()
            rel_name = f"{reverse[1]}_rev"
            suffix = 1
            while (src, rel_name, dst) in data.edge_types:
                suffix += 1
                rel_name = f"{reverse[1]}_rev{suffix}"
            data[(src, rel_name, dst)].edge_index = edge_index

    def _materialize_meta_graph(self, hetero: HeteroData) -> None:
        if self._meta_graph_cpu is not None:
            return
        base = hetero.cpu().clone()
        self._ensure_required_pairs(base)

        # 保留原始图以便从中复制单跳关系
        pre_base = base.clone()
        if self._meta_transform is not None:
            base = self._meta_transform(base)

        # 手动将单跳元路径添加为 metapath_* 边类型
        def _count_existing_meta_edges(g: HeteroData) -> int:
            return sum(1 for (_s, r, _d) in g.edge_types if isinstance(r, str) and r.startswith('metapath_'))

        meta_counter = _count_existing_meta_edges(base)
        for typed in self._single_typed:
            if not typed:
                continue
            s, r, d = typed[0]
            if (s, r, d) not in pre_base.edge_types:
                continue
            ei = pre_base[(s, r, d)].edge_index
            new_rel = f"metapath_{meta_counter}"
            base[(s, new_rel, d)].edge_index = ei.contiguous()
            meta_counter += 1

        # 如未运行AddMetaPaths且要求丢弃原始边，则仅保留metapath_*边
        if self._meta_transform is None and self._drop_orig_edge_types:
            keep = [et for et in base.edge_types if isinstance(et[1], str) and et[1].startswith('metapath_')]
            pruned = HeteroData()
            for nt in base.node_types:
                pruned[nt].num_nodes = base[nt].num_nodes
                if 'x' in base[nt]:
                    pruned[nt].x = base[nt].x
            for et in keep:
                pruned[et].edge_index = base[et].edge_index
            base = pruned

        self._meta_graph_cpu = base
        self._meta_metadata = base.metadata()

        if self.han_conv1 is None:
            param_device = self.tail_align.weight.device
            self.han_conv1 = HANConv(-1, self.d_hid, metadata=self._meta_metadata, heads=self.heads, dropout=self.dropout_p).to(param_device)
            self.han_conv2 = HANConv(self.d_hid, self.d_hid, metadata=self._meta_metadata, heads=self.heads, dropout=self.dropout_p).to(param_device)

    def prepare_meta_graph(self, hetero: HeteroData) -> None:
        """Explicitly build (or rebuild) the cached meta-path graph on CPU."""
        self._meta_graph_cpu = None
        self._meta_metadata = None
        self.han_conv1 = None
        self.han_conv2 = None
        self._materialize_meta_graph(hetero)
    def _clone_meta_graph(
        self,
        hetero: HeteroData,
        device: torch.device,
        feats_override: Optional[torch.Tensor],
    ) -> HeteroData:
        self._materialize_meta_graph(hetero)
        assert self._meta_graph_cpu is not None
        meta_graph = self._meta_graph_cpu.clone()
        for nt in meta_graph.node_types:
            if 'x' not in meta_graph[nt] or 'x' not in hetero[nt]:
                continue
            if nt == self.category and feats_override is not None:
                feat = feats_override.detach()
            else:
                feat = hetero[nt].x.detach()
            meta_graph[nt].x = feat.cpu()
        return meta_graph.to(device)

    # ------------------------------------------------------------------
    # KD feature helpers
    # ------------------------------------------------------------------
    def _compute_mp_outputs_fixed(
        self,
        hetero: HeteroData,
        device: torch.device,
        feats_override: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x_dict = {}
        for nt in hetero.node_types:
            if nt == self.category and feats_override is not None:
                x_dict[nt] = feats_override.to(device)
            else:
                x_dict[nt] = hetero[nt].x.to(device)
        mp_outputs: Dict[str, torch.Tensor] = {}
        for key in self.metapath_keys:
            mp = self.metapaths[key]
            ops_info = self.ops_template[key]
            z = self.node_proj[mp[0][0]](x_dict[mp[0][0]])
            for op in ops_info['norm_ops']:
                z = op.to(device).matmul(z)
            mp_outputs[key] = z
        return mp_outputs

    def _extract_beta(
        self,
        sem_attn: Optional[Dict[str, torch.Tensor]],
        meta_graph: HeteroData,
        device: torch.device,
    ) -> torch.Tensor:
        num_nodes = meta_graph[self.category].num_nodes
        num_paths = max(1, len(self.metapath_keys))
        if not sem_attn or self.category not in sem_attn:
            return torch.full((num_nodes, num_paths), 1.0 / num_paths, device=device)
        beta = sem_attn[self.category].to(device)
        if beta.dim() == 3:
            beta = beta.mean(dim=1)
        if beta.dim() == 1:
            beta = beta.unsqueeze(0).expand(num_nodes, -1)
        elif beta.size(0) != num_nodes:
            beta = beta.reshape(num_nodes, -1)
        return beta.contiguous()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, hetero: HeteroData, feats_override: Optional[torch.Tensor] = None):
        logits, *_ = self.forward_with_details(hetero, feats_override=feats_override)
        return logits

    def forward_with_details(
        self,
        hetero: HeteroData,
        device: Optional[torch.device] = None,
        feats_override: Optional[torch.Tensor] = None,
    ):
        device = device or next(self.parameters()).device
        meta_graph = self._clone_meta_graph(hetero, device, feats_override)

        tail_baseline = self.tail_align(meta_graph[self.category].x)

        x_dict = {nt: meta_graph[nt].x for nt in meta_graph.node_types}
        edge_index_dict = {et: meta_graph[et].edge_index for et in meta_graph.edge_types}

        h_dict = self.han_conv1(x_dict, edge_index_dict)
        h_dict = {nt: self.dropout(self.act(h)) for nt, h in h_dict.items()}

        out_dict, sem_attn = self.han_conv2(
            h_dict,
            edge_index_dict,
            return_semantic_attention_weights=True,
        )
        logits = self.lin(out_dict[self.category])

        beta = self._extract_beta(sem_attn, meta_graph, device)
        mp_outputs = self._compute_mp_outputs_fixed(hetero, device, feats_override)
        alpha_maps: Dict[str, SparseTensor] = {}
        return logits, mp_outputs, beta, alpha_maps, tail_baseline

    def project_teacher_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return self.delta_projector(delta)

    def project_student_delta(self, delta: torch.Tensor) -> torch.Tensor:
        return self.delta_projector_student(delta)# =========================
# Loss components
# =========================

def _edge_relpos(emb_dst: torch.Tensor, emb_src_in: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    src, dst = ei[0].long(), ei[1].long()
    return emb_dst[dst] - emb_src_in[src]


def relation_relative_pos_l2(taps_teacher: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
                             rel_student: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
                             hetero: HeteroData,
                             category: str,
                             reliability: Optional[torch.Tensor] = None,
                             projector_t: Optional[nn.Module] = None,
                             projector_s: Optional[nn.Module] = None) -> torch.Tensor:
    losses = []
    device = reliability.device if reliability is not None else hetero[category].x.device
    for et in hetero.edge_types:
        if et not in taps_teacher or et not in rel_student:
            continue
        src, _, dst = et
        if dst != category:
            continue
        ei = hetero[et].edge_index.to(device)
        emb_t_dst = taps_teacher[et]['dst'].to(device)
        emb_t_src = taps_teacher[et]['src_in'].to(device)
        emb_s_dst = rel_student[et]['dst']
        emb_s_src = rel_student[et]['src_in']
        if projector_t is not None:
            emb_t_dst = projector_t(emb_t_dst)
            emb_t_src = projector_t(emb_t_src)
        if projector_s is not None:
            emb_s_dst = projector_s(emb_s_dst)
            emb_s_src = projector_s(emb_s_src)
        rel_t = _edge_relpos(emb_t_dst, emb_t_src, ei)
        rel_s = _edge_relpos(emb_s_dst, emb_s_src, ei)
        l2 = (rel_t - rel_s).pow(2).sum(dim=-1) / rel_t.size(1)
        if reliability is not None:
            l2 = l2 * reliability[ei[1].long()]
        losses.append(l2.mean())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def meta_path_feature_loss(mp_teacher: Dict[str, torch.Tensor],
                           mp_student: Dict[str, torch.Tensor],
                           beta: torch.Tensor,
                           reliability: torch.Tensor,
                           metapath_keys: List[str]) -> torch.Tensor:
    terms = []
    for idx, key in enumerate(metapath_keys):
        if key not in mp_teacher or key not in mp_student:
            continue
        h_t = F.normalize(mp_teacher[key], dim=-1)
        h_s = F.normalize(mp_student[key], dim=-1)
        weight = reliability * beta[:, idx]
        diff = ((h_t - h_s).pow(2).sum(dim=-1)) * weight
        terms.append(diff.mean())
    if not terms:
        return torch.tensor(0.0, device=reliability.device)
    return torch.stack(terms).mean()


def meta_path_relpos_loss(mp_teacher: Dict[str, torch.Tensor],
                          mp_student: Dict[str, torch.Tensor],
                          tail_teacher: torch.Tensor,
                          tail_student: torch.Tensor,
                          teacher_proj: nn.Module,
                          student_proj: nn.Module,
                          reliability: torch.Tensor,
                          metapath_keys: List[str]) -> torch.Tensor:
    losses = []
    for key in metapath_keys:
        if key not in mp_teacher or key not in mp_student:
            continue
        delta_t = teacher_proj(mp_teacher[key] - tail_teacher)
        delta_s = student_proj(mp_student[key] - tail_student)
        loss = (delta_t - delta_s).pow(2).sum(dim=-1) / delta_t.size(1)
        losses.append((loss * reliability).mean())
    if not losses:
        return torch.tensor(0.0, device=reliability.device)
    return torch.stack(losses).mean()


def meta_path_beta_loss(beta_teacher: torch.Tensor,
                        beta_student: torch.Tensor,
                        reliability: torch.Tensor) -> torch.Tensor:
    log_hat = beta_student.clamp_min(1e-8).log()
    loss = F.kl_div(log_hat, beta_teacher, reduction='none').sum(dim=-1)
    return (loss * reliability).mean()
# =========================
# Optional MetaPath2Vec positional encodings
# =========================

def metapath2vec_category_embeddings(hetero: HeteroData,
                                     metapaths: List[List[Tuple[str, str, str]]],
                                     category: str,
                                     emb_dim: int = 128,
                                     walk_length: int = 40,
                                     context_size: int = 5,
                                     walks_per_node: int = 10,
                                     epochs: int = 30,
                                     device: torch.device = torch.device('cpu'),
                                     cache_dir: Optional[str] = None,
                                     seed: Optional[int] = None) -> torch.Tensor:
    if not metapaths:
        return torch.zeros((hetero[category].num_nodes, emb_dim), dtype=torch.float32, device=device)

    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = {
            'dataset': getattr(hetero, 'dataset_name', 'unknown'),
            'metapaths': metapaths,
            'emb_dim': emb_dim,
            'walk_length': walk_length,
            'context': context_size,
            'walks': walks_per_node,
            'epochs': epochs,
        }
        hash_key = hashlib.md5(json.dumps(cache_key, sort_keys=True).encode('utf-8')).hexdigest()
        cache_path = Path(cache_dir) / f"mp2v_{hash_key}.pt"
        if cache_path.exists():
            print(f"[MP2V] Loading cached MetaPath2Vec embeddings from {cache_path}")
            cached = torch.load(cache_path, map_location=device)
            print(f"[MP2V] Cached embeddings shape: {cached.shape}")
            return cached
    # 采用与 relation_distill_only.py 一致的老版逻辑：逐条元路径训练，使用 metapath= 参数
    edge_index_dict = {et: hetero[et].edge_index for et in hetero.edge_types}
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    per_path_dim = max(8, emb_dim // max(1, len(metapaths)))
    all_embs: List[torch.Tensor] = []
    print(f"[MP2V] Start training MetaPath2Vec for {len(metapaths)} metapaths | emb_dim={emb_dim}, per_path_dim={per_path_dim}")
    for i, mp in enumerate(metapaths, start=1):
        print(f"[MP2V] Metapath {i}/{len(metapaths)}: {mp}")
        mp2v = MetaPath2Vec(
            edge_index_dict=edge_index_dict,
            embedding_dim=per_path_dim,
            metapath=mp,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=5,
            sparse=True,
        ).to(device)
        optimizer = torch.optim.SparseAdam(list(mp2v.parameters()), lr=0.01)
        mp2v.train()
        for ep in range(epochs):
            total_loss = 0.0
            for pos_rw, neg_rw in mp2v.loader(batch_size=128, shuffle=True):
                optimizer.zero_grad()
                loss = mp2v.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += float(loss)
            if (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
                print(f"[MP2V]   ep {ep+1:03d}/{epochs:03d} | loss {total_loss:.4f}")
        with torch.no_grad():
            out = mp2v(category)
            W = out if isinstance(out, torch.Tensor) else out.weight
            W = F.normalize(W, p=2, dim=1)
            mu, std = W.mean(0, keepdim=True), W.std(0, keepdim=True).clamp_min(1e-6)
            W = (W - mu) / std
            all_embs.append(W.detach())
    print("[MP2V] Finished training all metapaths.")

    if len(all_embs) == 0:
        z = torch.zeros((hetero[category].num_nodes, emb_dim), dtype=torch.float32, device=device)
    else:
        z = torch.cat(all_embs, dim=1)
        if z.size(1) < emb_dim:
            pad = torch.zeros(z.size(0), emb_dim - z.size(1), dtype=z.dtype, device=z.device)
            z = torch.cat([z, pad], dim=1)
        elif z.size(1) > emb_dim:
            z = z[:, :emb_dim]
    
    print(f"[MP2V] Final embeddings shape: {z.shape}")
    if cache_path is not None:
        torch.save(z, cache_path)
        print(f"[MP2V] Saved embeddings to {cache_path}")
    return z
class RelationStructuralHead(nn.Module):
    def __init__(self,
                 relations: List[Tuple[str, str, str]],
                 category: str,
                 rel_dim: int,
                 num_classes: int):
        super().__init__()
        self.category = category
        self.rel_list = [et for et in relations if et[2] == category]
        self.rel_heads = nn.ModuleDict({self._key(et): nn.Linear(rel_dim, num_classes, bias=False)
                                        for et in self.rel_list})

    @staticmethod
    def _key(et: Tuple[str, str, str]) -> str:
        return f"{et[0]}::{et[1]}::{et[2]}"

    def forward(self, rel_embs: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        logits = None
        for et in self.rel_list:
            if et not in rel_embs:
                continue
            contrib = self.rel_heads[self._key(et)](rel_embs[et]['dst'])
            logits = contrib if logits is None else logits + contrib
        return logits
# =========================
# Training helpers
# =========================

def train_rsage_teacher(args, hetero: HeteroData, category: str,
                        y: torch.Tensor, idx_train: torch.Tensor,
                        idx_val: torch.Tensor, idx_test: torch.Tensor,
                        device: torch.device) -> RSAGE_Hetero:
    node_dims = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
    model = RSAGE_Hetero(
        etypes=list(hetero.edge_types),
        in_dim=node_dims[category],
        hid_dim=args.teacher_hidden,
        num_classes=int(y.max().item()) + 1,
        category=category,
        num_layers=args.teacher_layers,
        dropout=args.teacher_dropout,
        norm_type='none',  # 与 relation_distill_only.py 保持一致，避免 BatchNorm 不稳定
        node_type_dims=node_dims,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.teacher_lr, weight_decay=args.teacher_wd)
    best_state, best_val, best_te, patience = None, -1.0, 0.0, 0
    hetero_device = hetero.to(device)

    for epoch in range(1, args.teacher_epochs + 1):
        model.train()
        logits = model(hetero_device)
        loss = F.cross_entropy(logits[idx_train], y[idx_train])
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(hetero_device)
            tr = accuracy(logits, y, idx_train)
            va = accuracy(logits, y, idx_val)
            te = accuracy(logits, y, idx_test)
        print(f"[RSAGE] ep {epoch:03d} | loss {loss.item():.4f} | tr {tr:.4f} va {va:.4f} te {te:.4f}")
        if va >= best_val:
            best_val, best_state, best_te, patience = va, copy.deepcopy(model.state_dict()), te, 0
        else:
            patience += 1
            if patience >= args.teacher_patience:
                print("[RSAGE] early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(hetero_device)
        tr = accuracy(logits, y, idx_train)
        va = accuracy(logits, y, idx_val)
        te = accuracy(logits, y, idx_test)
    print(f"[RSAGE] Final(best) | tr {tr:.4f} va {va:.4f} te {te:.4f}")
    return model


def train_metapath_teacher(args, hetero: HeteroData, category: str,
                           y: torch.Tensor, idx_train: torch.Tensor,
                           idx_val: torch.Tensor, idx_test: torch.Tensor,
                           metapaths: List[List[Tuple[str, str, str]]],
                           ops_template: Dict[str, Dict[str, List[SparseTensor]]],
                           device: torch.device) -> MetaPathTeacher:
    node_dims = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
    model = MetaPathTeacher(
        node_dims=node_dims,
        category=category,
        metapaths=metapaths,
        ops_template=ops_template,
        d_hid=args.han_hidden,
        num_classes=int(y.max().item()) + 1,
        semantic_hidden=args.han_semantic_hidden,
        dropout=args.han_dropout,
        heads=args.han_heads,
    ).to(device)
    model.prepare_meta_graph(hetero)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.han_lr, weight_decay=args.han_wd)
    best_state, best_val, best_te, patience = None, -1.0, 0.0, 0
    hetero_device = hetero.to(device)

    for epoch in range(1, args.han_epochs + 1):
        model.train()
        logits = model(hetero_device)
        loss = F.cross_entropy(logits[idx_train], y[idx_train])
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(hetero_device)
            tr = accuracy(logits, y, idx_train)
            va = accuracy(logits, y, idx_val)
            te = accuracy(logits, y, idx_test)
        print(f"[HAN] ep {epoch:03d} | loss {loss.item():.4f} | tr {tr:.4f} va {va:.4f} te {te:.4f}")
        if va >= best_val:
            best_val, best_state, best_te, patience = va, copy.deepcopy(model.state_dict()), te, 0
        else:
            patience += 1
            if patience >= args.han_patience:
                print("[HAN] early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(hetero_device)
        tr = accuracy(logits, y, idx_train)
        va = accuracy(logits, y, idx_val)
        te = accuracy(logits, y, idx_test)
    print(f"[HAN] Final(best) | tr {tr:.4f} va {va:.4f} te {te:.4f}")
    return model
    
def train_student_dual_kd(args,
                          hetero: HeteroData,
                          category: str,
                          y: torch.Tensor,
                          idx_train: torch.Tensor,
                          idx_val: torch.Tensor,
                          idx_test: torch.Tensor,
                          device: torch.device,
                          rsage_teacher: RSAGE_Hetero,
                          han_teacher: MetaPathTeacher,
                          metapaths: List[List[Tuple[str, str, str]]],
                          ops_template: Dict[str, Dict[str, List[SparseTensor]]]) -> Tuple[nn.Module, torch.Tensor]:
    num_classes = int(y.max().item()) + 1
    node_dims = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}

    base_feats = hetero[category].x.to(device)
    student_feats = base_feats
    mp2v_feats = None
    if args.use_positional_encoding:
        mp2v_feats = metapath2vec_category_embeddings(
            hetero=hetero,
            metapaths=metapaths,
            category=category,
            emb_dim=args.mp_pe_dim,
            walk_length=args.mp_walk_length,
            context_size=args.mp_context_size,
            walks_per_node=args.mp_walks_per_node,
            epochs=args.mp_epochs,
            device=device,
            cache_dir=args.mp_cache_dir,
            seed=args.seed,
        )
        student_feats = torch.cat([student_feats, mp2v_feats], dim=-1)

    student = MLPClassifier(d_in=student_feats.size(1),
                            n_classes=num_classes,
                            hidden=args.student_hidden,
                            num_layers=args.student_layers,
                            dropout=args.student_dropout).to(device)
    # 修正：开启位置编码后，category 节点维度已改变，需要更新 node_dims
    node_dims[category] = student_feats.size(1)

    rel_adapter = RelKDAdapter(node_dims, d_rel=args.rel_dim).to(device)
    mp_adapter = MetaPathAdapter(node_dims, metapaths, d_mp=args.mp_dim, ops_template=ops_template).to(device)
    semantic_head = SemanticHead(d_mp=args.mp_dim, hidden=args.mp_beta_hidden).to(device)
    struct_head = RelationStructuralHead(list(hetero.edge_types), category,
                                         rel_dim=args.rel_dim, num_classes=num_classes).to(device)

    tail_student_align = nn.Linear(student_feats.size(1), args.mp_dim, bias=False).to(device)
    teacher_delta_proj = nn.Linear(args.han_hidden, args.delta_dim, bias=False).to(device)
    student_delta_proj = nn.Linear(args.mp_dim, args.delta_dim, bias=False).to(device)

    # 优化器将在教师 taps 与投影器构建完成后创建

    hetero_device = hetero.to(device)

    rsage_teacher.eval(); han_teacher.eval()
    with torch.no_grad():
        logits_r, probs_r, rho_r = compute_logits_with_reliability(
            rsage_teacher, hetero, category, device,
            feats_override=None,
            noise_std=args.noise_std,
            alpha=args.reliability_alpha,
        )
        logits_h, probs_h, rho_h = compute_logits_with_reliability(
            han_teacher, hetero, category, device,
            feats_override=None,
            noise_std=args.noise_std,
            alpha=args.reliability_alpha,
        )
        logits_r = logits_r.to(device); probs_r = probs_r.to(device); rho_r = rho_r.to(device)
        logits_h = logits_h.to(device); probs_h = probs_h.to(device); rho_h = rho_h.to(device)
        _, taps = rsage_teacher.forward_with_relation_taps(hetero_device)
        logits_h_full, mp_teacher_embs, beta_teacher, alpha_maps, tail_teacher = han_teacher.forward_with_details(hetero_device)
        tail_teacher = tail_teacher.detach()
        mp_teacher_embs = {k: v.detach() for k, v in mp_teacher_embs.items()}
        beta_teacher = beta_teacher.detach()
        alpha_maps = {k: v.coalesce() for k, v in alpha_maps.items()}

    # 在拿到 taps 后，再构建教师侧投影与优化器，避免未绑定变量
    rel_t_dst_proj: Optional[nn.Module] = None
    rel_t_src_proj: Optional[nn.Module] = None
    if taps and len(taps) > 0:
        any_et = next(iter(taps))
        t_dst_dim = taps[any_et]['dst'].size(1)
        t_src_dim = taps[any_et]['src_in'].size(1)
        rel_t_dst_proj = (nn.Linear(t_dst_dim, args.rel_dim, bias=False).to(device)
                          if t_dst_dim != args.rel_dim else nn.Identity().to(device))
        rel_t_src_proj = (nn.Linear(t_src_dim, args.rel_dim, bias=False).to(device)
                          if t_src_dim != args.rel_dim else nn.Identity().to(device))
    
    params = list(student.parameters()) + list(rel_adapter.parameters()) + list(mp_adapter.parameters()) + \
        list(semantic_head.parameters()) + list(struct_head.parameters()) + \
        list(tail_student_align.parameters()) + list(teacher_delta_proj.parameters()) + list(student_delta_proj.parameters())
    if rel_t_dst_proj is not None and not isinstance(rel_t_dst_proj, nn.Identity):
        params += list(rel_t_dst_proj.parameters())
    if rel_t_src_proj is not None and not isinstance(rel_t_src_proj, nn.Identity):
        params += list(rel_t_src_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=args.student_lr, weight_decay=args.student_wd)

    js = js_divergence(probs_r, probs_h)
    gamma = torch.sigmoid(args.gamma_a * (1 - js) + args.gamma_b * (rho_r - rho_h))

    best_state, best_val, patience = None, -1.0, 0
    for epoch in range(1, args.student_epochs + 1):
        student.train()
        logits_s = student(student_feats)
        ce_loss = F.cross_entropy(logits_s[idx_train], y[idx_train]) * args.ce_coeff

        kd_rel = kd_kl(logits_s, logits_r, T=args.kd_T, reduce=False)
        kd_h = kd_kl(logits_s, logits_h, T=args.kd_T, reduce=False)
        kd_loss = ((gamma * rho_r * kd_rel) + ((1 - gamma) * rho_h * kd_h)).mean() * args.kd_coeff

        overrides = {category: student_feats}
        rel_student = rel_adapter(hetero, list(hetero.edge_types), node_overrides=overrides, device=device)
        # 对教师 taps 做按类型的线性投影以匹配 student 的 rel_dim
        taps_projected = {}
        for et, mm in taps.items():
            t_dst = mm['dst']
            t_src = mm['src_in']
            if rel_t_dst_proj is not None:
                t_dst = rel_t_dst_proj(t_dst)
            if rel_t_src_proj is not None:
                t_src = rel_t_src_proj(t_src)
            taps_projected[et] = {'dst': t_dst, 'src_in': t_src}

        rel_loss = relation_relative_pos_l2(taps_teacher=taps_projected,
                                            rel_student=rel_student,
                                            hetero=hetero,
                                            category=category,
                                            reliability=rho_r,
                                            projector_t=None,
                                            projector_s=None) * args.lambda_rel_pos

        struct_logits = struct_head(rel_student)
        struct_loss = torch.tensor(0.0, device=device)
        if struct_logits is not None and args.lambda_rel_struct > 0:
            struct_loss = F.cross_entropy(struct_logits[idx_train], y[idx_train]) * args.lambda_rel_struct

        mp_student_embs = mp_adapter(hetero, node_overrides=overrides, device=device)
        beta_student = semantic_head([mp_student_embs[k] for k in mp_adapter.metapath_keys])

        tail_student = tail_student_align(student_feats)
        mp_feat_loss = meta_path_feature_loss(mp_teacher_embs,
                                              mp_student_embs,
                                              beta_teacher,
                                              rho_h,
                                              mp_adapter.metapath_keys) * args.lambda_mp_feat

        mp_relpos = meta_path_relpos_loss(mp_teacher_embs,
                                          mp_student_embs,
                                          tail_teacher,
                                          tail_student,
                                          teacher_delta_proj,
                                          student_delta_proj,
                                          rho_h,
                                          mp_adapter.metapath_keys) * args.lambda_mp_relpos

        mp_beta = meta_path_beta_loss(beta_teacher, beta_student, rho_h) * args.lambda_mp_beta

        loss = ce_loss + kd_loss + rel_loss + struct_loss + mp_feat_loss + mp_relpos + mp_beta
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        student.eval()
        with torch.no_grad():
            logits_eval = student(student_feats)
            val_acc = accuracy(logits_eval, y, idx_val)
            train_acc = accuracy(logits_eval, y, idx_train)
            test_acc = accuracy(logits_eval, y, idx_test)
        print(f"[Student] ep {epoch:03d} | CE {ce_loss.item():.4f} KD {kd_loss.item():.4f} "
              f"REL {rel_loss.item():.4f} STRUCT {struct_loss.item():.4f} "
              f"MP_F {mp_feat_loss.item():.4f} MP_RP {mp_relpos.item():.4f} MP_B {mp_beta.item():.4f} | "
              f"tr {train_acc:.4f} va {val_acc:.4f} te {test_acc:.4f}")
        if val_acc >= best_val:
            best_val, best_state, patience = val_acc, copy.deepcopy(student.state_dict()), 0
        else:
            patience += 1
            if patience >= args.student_patience:
                print("[Student] early stop")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    logits_final = student(student_feats)
    test_acc = accuracy(logits_final, y, idx_test)
    print(f"[Student] Final(best) | val {best_val:.4f} | test {test_acc:.4f}")
    return student, student_feats
# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Dual-teacher (RSAGE + meta-path) distillation into graph-free MLP")
    parser.add_argument('-d', '--dataset', type=str, default='TMDB', choices=['TMDB', 'ArXiv', 'DBLP', 'IMDB', 'AMINER'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    # teacher RSAGE
    parser.add_argument('--teacher_hidden', type=int, default=128)
    parser.add_argument('--teacher_layers', type=int, default=2)
    parser.add_argument('--teacher_dropout', type=float, default=0.2)
    parser.add_argument('--teacher_norm', type=str, default='batch', choices=['none', 'batch', 'layer'])
    parser.add_argument('--teacher_lr', type=float, default=0.01)
    parser.add_argument('--teacher_wd', type=float, default=0.0)
    parser.add_argument('--teacher_epochs', type=int, default=100)
    parser.add_argument('--teacher_patience', type=int, default=30)

    # teacher HAN-like
    parser.add_argument('--han_hidden', type=int, default=128)
    parser.add_argument('--han_semantic_hidden', type=int, default=128)
    parser.add_argument('--han_dropout', type=float, default=0.5)
    parser.add_argument('--han_heads', type=int, default=8)
    parser.add_argument('--han_lr', type=float, default=0.005)
    parser.add_argument('--han_wd', type=float, default=0.001)
    parser.add_argument('--han_epochs', type=int, default=300)
    parser.add_argument('--han_patience', type=int, default=40)

    # meta-path options
    parser.add_argument('--positional_relations', type=str, nargs='*', default=[],
                        help='meta-paths expressed as comma-separated relation names, e.g. directed_by,directs performed_by,performs')

    # student
    parser.add_argument('--student_hidden', type=int, default=128)
    parser.add_argument('--student_layers', type=int, default=2)
    parser.add_argument('--student_dropout', type=float, default=0.5)
    parser.add_argument('--student_lr', type=float, default=0.002)
    parser.add_argument('--student_wd', type=float, default=0.0005)
    parser.add_argument('--student_epochs', type=int, default=1000)
    parser.add_argument('--student_patience', type=int, default=60)

    # KD weights
    parser.add_argument('--kd_T', type=float, default=1.0)
    parser.add_argument('--ce_coeff', type=float, default=1.0)
    parser.add_argument('--kd_coeff', type=float, default=1.0)
    parser.add_argument('--lambda_rel_pos', type=float, default=1.0)
    parser.add_argument('--lambda_rel_struct', type=float, default=0.5)
    parser.add_argument('--lambda_mp_feat', type=float, default=1.0)
    parser.add_argument('--lambda_mp_relpos', type=float, default=1.0)
    parser.add_argument('--lambda_mp_beta', type=float, default=0.5)
    parser.add_argument('--rel_dim', type=int, default=128)
    parser.add_argument('--mp_dim', type=int, default=128)
    parser.add_argument('--mp_beta_hidden', type=int, default=128)
    parser.add_argument('--delta_dim', type=int, default=128)

    parser.add_argument('--noise_std', type=float, default=0.05)
    parser.add_argument('--reliability_alpha', type=float, default=0.5)
    parser.add_argument('--gamma_a', type=float, default=4.0)
    parser.add_argument('--gamma_b', type=float, default=2.0)

    # positional encoding / mp2v
    parser.add_argument('--use_positional_encoding', action='store_true')
    parser.add_argument('--mp_pe_dim', type=int, default=128)
    parser.add_argument('--mp_walk_length', type=int, default=40)
    parser.add_argument('--mp_context_size', type=int, default=5)
    parser.add_argument('--mp_walks_per_node', type=int, default=10)
    parser.add_argument('--mp_epochs', type=int, default=30)
    parser.add_argument('--mp_cache_dir', type=str, default='./mp2v_cache')


    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')

    hetero, splits, gen_node_feats, metapaths_rel = load_data(dataset=args.dataset, return_mp=True)
    hetero = gen_node_feats(hetero)
    hetero.dataset_name = args.dataset
    category = hetero.category

    idx_train, idx_val, idx_test = [idx.to(device) for idx in splits]
    y = hetero[category].y.long().to(device)

    rel_sequences = []
    if args.positional_relations:
        for item in args.positional_relations:
            seq = [tok.strip() for tok in item.split(',') if tok.strip()]
            if seq:
                rel_sequences.append(seq)
    elif metapaths_rel:
        rel_sequences = metapaths_rel

    # 支持两种格式：
    # 1) 关系名序列（['directed_by','directs']）→ 需要解析
    # 2) 类型化三元组序列（[('author','to','paper'), ...]）→ 直接使用
    if rel_sequences and isinstance(rel_sequences[0], (list, tuple)) and len(rel_sequences[0]) > 0 \
       and isinstance(rel_sequences[0][0], (list, tuple)) and len(rel_sequences[0][0]) == 3:
        typed_metapaths = rel_sequences  # already typed
    else:
        typed_metapaths = parse_metapath_name_sequences(hetero, category, rel_sequences)
        typed_metapaths = ensure_metapaths_to_category(typed_metapaths, category)
    if not typed_metapaths:
        raise ValueError("No valid meta-paths ending at the category node were found.")

    ops_template = build_metapath_operators(hetero, typed_metapaths, device=torch.device('cpu'))

    rsage_teacher = train_rsage_teacher(args, hetero, category, y, idx_train, idx_val, idx_test, device)
    han_teacher = train_metapath_teacher(args, hetero, category, y, idx_train, idx_val, idx_test,
                                         typed_metapaths, ops_template, device)

    student, student_feats = train_student_dual_kd(args, hetero, category, y,
                                                   idx_train, idx_val, idx_test,
                                                   device, rsage_teacher, han_teacher,
                                                   typed_metapaths, ops_template)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    rsage_teacher.eval(); han_teacher.eval(); student.eval()
    hetero_device = hetero.to(device)
    with torch.no_grad():
        tr_r = accuracy(rsage_teacher(hetero_device), y, idx_train)
        va_r = accuracy(rsage_teacher(hetero_device), y, idx_val)
        te_r = accuracy(rsage_teacher(hetero_device), y, idx_test)
        print(f"RSAGE teacher  | train {tr_r:.4f} | val {va_r:.4f} | test {te_r:.4f}")
        logits_h = han_teacher(hetero_device)
        tr_h = accuracy(logits_h, y, idx_train)
        va_h = accuracy(logits_h, y, idx_val)
        te_h = accuracy(logits_h, y, idx_test)
        print(f"Meta teacher   | train {tr_h:.4f} | val {va_h:.4f} | test {te_h:.4f}")
        logits_s = student(student_feats)
        tr_s = accuracy(logits_s, y, idx_train)
        va_s = accuracy(logits_s, y, idx_val)
        te_s = accuracy(logits_s, y, idx_test)
        print(f"Student (MLP)  | train {tr_s:.4f} | val {va_s:.4f} | test {te_s:.4f}")

    # Inference benchmark
    print("\n" + "="*60)
    print("INFERENCE TIME COMPARISON")
    print("="*60)
    warmup = 3
    runs = 10
    
    def rsage_forward():
        return rsage_teacher(hetero_device)
    
    def han_forward():
        return han_teacher(hetero_device)
    
    def student_forward():
        return student(student_feats)
    
    mean_r, std_r = _benchmark_forward(rsage_forward, warmup, runs, device)
    mean_h, std_h = _benchmark_forward(han_forward, warmup, runs, device)
    mean_s, std_s = _benchmark_forward(student_forward, warmup, runs, device)
    
    print(f"RSAGE Teacher:  {mean_r * 1000:.3f} ± {std_r * 1000:.3f} ms")
    print(f"HAN Teacher:    {mean_h * 1000:.3f} ± {std_h * 1000:.3f} ms")
    print(f"Student (MLP):  {mean_s * 1000:.3f} ± {std_s * 1000:.3f} ms")
    if mean_s > 0:
        print(f"Speedup vs RSAGE: {mean_r / mean_s:.2f}x")
        print(f"Speedup vs HAN:   {mean_h / mean_s:.2f}x")
    print("="*60)


if __name__ == '__main__':
    main()





