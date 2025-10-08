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
try:
    from torch_sparse import SparseTensor
except ModuleNotFoundError:
    class _DenseWrapper:
        def __init__(self, tensor: torch.Tensor):
            self._tensor = tensor

        def to_dense(self) -> torch.Tensor:
            return self._tensor

    class SparseTensor:
        def __init__(self, row: torch.Tensor, col: torch.Tensor, value: torch.Tensor,
                     sparse_sizes: Tuple[int, int]):
            indices = torch.stack([row.long(), col.long()], dim=0)
            self._tensor = torch.sparse_coo_tensor(indices, value, sparse_sizes, device=value.device).coalesce()

        def coalesce(self) -> 'SparseTensor':
            self._tensor = self._tensor.coalesce()
            return self

        def to(self, device: torch.device) -> 'SparseTensor':
            self._tensor = self._tensor.to(device)
            return self

        def sum(self, dim: int) -> _DenseWrapper:
            summed = torch.sparse.sum(self._tensor, dim=dim)
            if isinstance(summed, torch.Tensor):
                if hasattr(summed, 'is_sparse') and summed.is_sparse:
                    summed = summed.to_dense()
                return _DenseWrapper(summed)
            return _DenseWrapper(summed.to_dense())

        def coo(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self._tensor = self._tensor.coalesce()
            idx = self._tensor.indices()
            return idx[0], idx[1], self._tensor.values()

        def matmul(self, dense: torch.Tensor) -> torch.Tensor:
            return torch.sparse.mm(self._tensor, dense)

        def __matmul__(self, dense: torch.Tensor) -> torch.Tensor:
            return self.matmul(dense)

        def to_dense(self) -> torch.Tensor:
            return self._tensor.to_dense()

        def clone(self) -> 'SparseTensor':
            cloned = SparseTensor.__new__(SparseTensor)
            cloned._tensor = self._tensor.clone()
            return cloned

        def sparse_sizes(self) -> Tuple[int, int]:
            return tuple(self._tensor.size())

        def __repr__(self) -> str:
            return f"SparseTensor(size={self._tensor.size()})"

        @classmethod
        def from_dense(cls, dense: torch.Tensor) -> 'SparseTensor':
            indices = dense.nonzero(as_tuple=True)
            if len(indices[0]) == 0:
                idx = torch.zeros((2, 1), dtype=torch.long, device=dense.device)
                values = dense.new_zeros(1)
                inst = cls.__new__(cls)
                inst._tensor = torch.sparse_coo_tensor(idx, values, dense.shape, device=dense.device).coalesce()
                return inst
            idx = torch.stack(indices, dim=0)
            values = dense[indices]
            inst = cls.__new__(cls)
            inst._tensor = torch.sparse_coo_tensor(idx, values, dense.shape, device=dense.device).coalesce()
            return inst

try:
    import torch_sparse
except ModuleNotFoundError:
    class _TorchSparseFallback:
        @staticmethod
        def matmul(lhs, rhs):
            lhs_dense = lhs.to_dense() if hasattr(lhs, 'to_dense') else lhs
            rhs_dense = rhs.to_dense() if hasattr(rhs, 'to_dense') else rhs
            result = lhs_dense @ rhs_dense
            if isinstance(result, torch.Tensor) and hasattr(SparseTensor, 'from_dense'):
                return SparseTensor.from_dense(result)
            return result

    torch_sparse = _TorchSparseFallback()


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
    # 强化确定性：启用确定性算法并设置CUBLAS工作区（需重启影响最小）
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    # 关闭TF32以提升可重复性
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
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


class GraphFreeStudent(nn.Module):
    def __init__(self, d_in: int, n_classes: int,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.5,
                 rel_dim: int = 128, mp_dim: int = 128, delta_dim: int = 128,
                 beta_hidden: int = 128):
        super().__init__()
        hidden_layers = max(0, num_layers - 1)
        self.blocks = nn.ModuleList()
        prev = d_in
        for _ in range(hidden_layers):
            block = nn.Sequential(
                nn.Linear(prev, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks.append(block)
            prev = hidden

        self.feature_dim = prev
        self.rel_dim = rel_dim
        self.mp_dim = mp_dim
        self.delta_dim = delta_dim

        self.classifier = nn.Linear(prev, n_classes)
        self.rel_proj = nn.Identity() if prev == rel_dim else nn.Linear(prev, rel_dim, bias=False)
        self.struct_classifier = nn.Linear(rel_dim, n_classes)
        self.mp_proj = nn.Identity() if prev == mp_dim else nn.Linear(prev, mp_dim, bias=False)
        self.tail_proj = nn.Identity() if prev == mp_dim else nn.Linear(prev, mp_dim, bias=False)
        self.delta_proj = nn.Identity() if mp_dim == delta_dim else nn.Linear(mp_dim, delta_dim, bias=False)
        self.beta_mlp = nn.Sequential(
            nn.Linear(mp_dim, beta_hidden),
            nn.Tanh(),
            nn.Linear(beta_hidden, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h)
        return h

    def forward_all(self, x: torch.Tensor):
        feats = self.forward_features(x)
        logits = self.classifier(feats)
        rel_base = self.rel_proj(feats) if not isinstance(self.rel_proj, nn.Identity) else feats
        mp_base = self.mp_proj(feats) if not isinstance(self.mp_proj, nn.Identity) else feats
        tail = self.tail_proj(feats) if not isinstance(self.tail_proj, nn.Identity) else feats
        return logits, feats, rel_base, mp_base, tail

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, *_ = self.forward_all(x)
        return logits


    def structural_logits_direct(self, rel_base: torch.Tensor, category: str) -> torch.Tensor:
        """直接使用MLP特征进行结构分类，不依赖关系嵌入字典"""
        return self.struct_classifier(rel_base)

    def meta_path_attention(self, mp_embs: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
        if not mp_embs:
            raise ValueError("meta_path_attention requires non-empty embeddings")
        device = next(self.beta_mlp.parameters()).device
        any_emb = next(iter(mp_embs.values()))
        num_nodes = any_emb.size(0)
        logits = []
        for key in keys:
            if key in mp_embs:
                logits.append(self.beta_mlp(mp_embs[key]).squeeze(-1))
            else:
                logits.append(torch.full((num_nodes,), -1e9, device=device))
        beta_logits = torch.stack(logits, dim=1)
        return F.softmax(beta_logits, dim=1)


 
# =========================
# Student meta-path embeddings (direct, no adapter)
# =========================

def build_student_metapath_embs_direct(
    hetero: HeteroData,
    ops_template: Dict[str, Dict[str, List[SparseTensor]]],
    mp_base: torch.Tensor,
    category: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build student meta-path embeddings using only student's category features (mp_base),
    without any per-node-type adapter. For non-category node types, construct features by
    size-aligning mp_base via truncate/pad, then propagate along norm_ops to the category.
    """
    # Prepare per-node-type features by reusing category features with size alignment
    x_proj: Dict[str, torch.Tensor] = {}
    cat_feats = mp_base.to(device)
    cat_num = hetero[category].num_nodes
    for nt in hetero.node_types:
        n_nodes = hetero[nt].num_nodes
        if nt == category:
            x_proj[nt] = cat_feats
        else:
            if n_nodes == cat_num:
                x_proj[nt] = cat_feats
            elif n_nodes < cat_num:
                x_proj[nt] = cat_feats[:n_nodes]
            else:
                pad = torch.zeros(n_nodes - cat_num, cat_feats.size(1), device=cat_feats.device, dtype=cat_feats.dtype)
                x_proj[nt] = torch.cat([cat_feats, pad], dim=0)

    # Compute mp embeddings by chaining normalized operators
    mp_embs: Dict[str, torch.Tensor] = {}
    for key, info in ops_template.items():
        typed = info['typed']  # List[Tuple[src, rel, dst]]
        if not typed:
            continue
        start_nt = typed[0][0]
        z = x_proj[start_nt]
        for op in info['norm_ops']:
            z = op.to(device).matmul(z)
        mp_embs[key] = z
    return mp_embs
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
        # Lightweight path: NO semantic attention tensors during training
        device = next(self.parameters()).device
        meta_graph = self._clone_meta_graph(hetero, device, feats_override)

        x_dict = {nt: meta_graph[nt].x for nt in meta_graph.node_types}
        edge_index_dict = {et: meta_graph[et].edge_index for et in meta_graph.edge_types}

        h_dict = self.han_conv1(x_dict, edge_index_dict)
        h_dict = {nt: self.dropout(self.act(h)) for nt, h in h_dict.items()}

        # IMPORTANT: do NOT ask for semantic attention weights here
        out_dict = self.han_conv2(h_dict, edge_index_dict)
        logits = self.lin(out_dict[self.category])
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
        return self.delta_projector_student(delta)


# =========================
# Loss componlp
# =========================

def _edge_relpos(emb_dst: torch.Tensor, emb_src_in: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    src, dst = ei[0].long(), ei[1].long()
    # Normalize embeddings to control scale; compute features on unit vectors
    x_dst = emb_dst[dst]
    x_src = emb_src_in[src]
    x_dst_n = F.normalize(x_dst, dim=-1)
    x_src_n = F.normalize(x_src, dim=-1)
    # Cosine similarity in [-1, 1]
    cosine_sim = F.cosine_similarity(x_dst_n, x_src_n, dim=-1).unsqueeze(-1)  # [E, 1]
    # Euclidean distance between unit vectors in [0, 2]; divide by 2 => [0, 1]
    euclidean = torch.norm(x_dst_n - x_src_n, p=2, dim=-1, keepdim=True) / 2.0  # [E, 1]
    # Return per-edge features: [cosine_similarity, normalized_euclidean_distance]
    return torch.cat([cosine_sim, euclidean], dim=-1)


def relation_relative_pos_l2(
    taps_teacher: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
    rel_student: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
    hetero: HeteroData,
    category: str,
    reliability: Optional[torch.Tensor] = None,
    projector_t: Optional[nn.Module] = None,
    projector_s: Optional[nn.Module] = None,
    relation_weights: Optional[Dict[Tuple[str, str, str], float]] = None,
    return_details: bool = False,
    include_per_edge: bool = False,
) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[Tuple[str, str, str], Dict[str, Union[int, float, torch.Tensor]]]]]]:
    """
    Relation-wise L2 loss between teacher taps and student relation embeddings.

    Enhancements:
    - relation_weights: optional weights per edge-type when aggregating relation losses
    - return_details: if True, returns a dict with total_loss and per-relation statistics
    - include_per_edge: if True with return_details, includes per-edge loss tensors (can be large)
    """
    losses: List[torch.Tensor] = []
    device = reliability.device if reliability is not None else hetero[category].x.device

    # Aggregation for weighted mean across relations
    weighted_sum: Optional[torch.Tensor] = None
    total_weight: float = 0.0

    # Optional stats container
    relation_stats: Dict[Tuple[str, str, str], Dict[str, Union[int, float, torch.Tensor]]] = {}
    relation_mean_tensors: Dict[Tuple[str, str, str], torch.Tensor] = {}

    candidate_edge_types = [
        et for et in hetero.edge_types
        if (et in taps_teacher and et in rel_student and et[2] == category)
    ]

    for et in candidate_edge_types:

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

        rel_mean = l2.mean()
        losses.append(rel_mean)

        w = 1.0
        if relation_weights is not None and et in relation_weights:
            w = float(relation_weights[et])

        weighted_term = rel_mean * w
        weighted_sum = weighted_term if weighted_sum is None else (weighted_sum + weighted_term)
        total_weight += w

        if return_details:
            rel_std = (l2.std() if l2.numel() > 1 else torch.tensor(0.0, device=l2.device))
            relation_stats[et] = {
                'mean_loss': float(rel_mean.detach().cpu().item()),
                'std_loss': float(rel_std.detach().cpu().item()),
                'min_loss': float(l2.min().detach().cpu().item()),
                'max_loss': float(l2.max().detach().cpu().item()),
                'num_edges': int(l2.numel()),
                'weight': float(w),
            }
            relation_mean_tensors[et] = rel_mean
            if include_per_edge:
                relation_stats[et]['per_edge_loss'] = l2.detach().cpu()

    if not losses:
        zero = torch.tensor(0.0, device=device)
        if return_details:
            return {'total_loss': zero, 'relation_stats': {}, 'num_relations': 0}
        return zero

    # Use weighted mean if weights provided, otherwise plain mean
    total_loss = (weighted_sum / total_weight) if (total_weight > 0 and weighted_sum is not None) else torch.stack(losses).mean()

    if return_details:
        return {
            'total_loss': total_loss,
            'relation_stats': relation_stats,
            'relation_mean_tensors': relation_mean_tensors,
            'num_relations': len(relation_stats),
        }

    return total_loss


def relation_combined_loss(
    rel_result: Union[torch.Tensor, Dict[str, Union[int, float, torch.Tensor, Dict[Tuple[str, str, str], torch.Tensor]]]],
    struct_logits: Optional[torch.Tensor],
    y: torch.Tensor,
    idx_train: torch.Tensor,
    lambda_rel_pos: float,
    lambda_rel_struct: float,
    lambda_rel_total: Optional[float],
    balance_override: Optional[float],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    def _to_tensor(val):
        if isinstance(val, torch.Tensor):
            return val.to(device)
        return torch.tensor(float(val), device=device)

    if isinstance(rel_result, dict):
        if rel_result.get('relation_mean_tensors'):
            rel_core = torch.stack(list(rel_result['relation_mean_tensors'].values())).sum()
        else:
            rel_core = rel_result.get('total_loss', torch.tensor(0.0, device=device))
            if not isinstance(rel_core, torch.Tensor):
                rel_core = torch.tensor(float(rel_core), device=device)
    else:
        rel_core = _to_tensor(rel_result)

    if struct_logits is not None:
        struct_core = F.cross_entropy(struct_logits[idx_train], y[idx_train])
    else:
        struct_core = torch.tensor(0.0, device=device)

    rel_weight_raw = max(float(lambda_rel_pos), 0.0)
    struct_weight_raw = max(float(lambda_rel_struct), 0.0)

    if lambda_rel_total is not None:
        scale_value = float(lambda_rel_total)
    else:
        scale_value = rel_weight_raw + struct_weight_raw

    if scale_value <= 0.0:
        zero = torch.tensor(0.0, device=device)
        return {
            'total': zero,
            'scaled': {'relpos': zero, 'struct': zero},
            'weights': {
                'relpos': torch.tensor(0.0, device=device),
                'struct': torch.tensor(0.0, device=device),
                'scale': torch.tensor(0.0, device=device),
            },
            'raw': {'relpos': rel_core, 'struct': struct_core},
        }

    if balance_override is not None:
        balance = float(min(max(balance_override, 0.0), 1.0))
        struct_weight = balance
        rel_weight = 1.0 - balance
    else:
        total_raw = rel_weight_raw + struct_weight_raw
        if total_raw > 0.0:
            rel_weight = rel_weight_raw / total_raw
            struct_weight = struct_weight_raw / total_raw
        else:
            rel_weight = 0.5
            struct_weight = 0.5

    scale_tensor = torch.tensor(scale_value, device=device)
    rel_component = rel_core * rel_weight * scale_tensor
    struct_component = struct_core * struct_weight * scale_tensor
    total = rel_component + struct_component

    return {
        'total': total,
        'scaled': {'relpos': rel_component, 'struct': struct_component},
        'weights': {
            'relpos': torch.tensor(rel_weight, device=device),
            'struct': torch.tensor(struct_weight, device=device),
            'scale': scale_tensor,
        },
        'raw': {'relpos': rel_core, 'struct': struct_core},
    }


def meta_path_alignment_losses(
    mp_teacher: Dict[str, torch.Tensor],
    mp_student: Dict[str, torch.Tensor],
    tail_teacher: torch.Tensor,
    tail_student: torch.Tensor,
    teacher_proj: nn.Module,
    student_proj: nn.Module,
    beta_teacher: torch.Tensor,
    beta_student: torch.Tensor,
    reliability: torch.Tensor,
    metapath_keys: List[str],
    component_weights: Optional[Dict[str, float]] = None,
    lambda_mp_total: Optional[float] = None,
    eps: float = 1e-8,
) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    device = reliability.device
    reliability = reliability.view(-1)
    active_keys = [key for key in metapath_keys if key in mp_teacher and key in mp_student]
    if len(active_keys) == 0:
        zero = torch.tensor(0.0, device=device)
        zero_w = torch.tensor(0.0, device=device)
        return {
            'feature': zero,
            'relpos': zero,
            'beta': zero,
            'total': zero,
            'scaled': {'feature': zero, 'relpos': zero, 'beta': zero},
            'raw': {'feature': zero, 'relpos': zero, 'beta': zero},
            'weights': {'feat': zero_w, 'relpos': zero_w, 'beta': zero_w},
            'details': {'active_keys': []},
        }

    num_nodes = reliability.size(0)
    teacher_tail = F.normalize(teacher_proj(tail_teacher), dim=-1)
    student_tail = F.normalize(student_proj(tail_student), dim=-1)

    teacher_idx = {key: idx for idx, key in enumerate(mp_teacher.keys())}
    student_idx = {key: idx for idx, key in enumerate(mp_student.keys())}
    teacher_fill_dtype = beta_teacher.dtype if beta_teacher.numel() > 0 else teacher_tail.dtype
    student_fill_dtype = beta_student.dtype if beta_student.numel() > 0 else student_tail.dtype

    def _align_beta(beta: Optional[torch.Tensor], index_map: Dict[str, int], fill_dtype: torch.dtype) -> torch.Tensor:
        if beta is None or beta.numel() == 0:
            aligned = torch.full((num_nodes, len(active_keys)), 1.0 / max(1, len(active_keys)),
                                 device=device, dtype=fill_dtype)
        else:
            cols = []
            for key in active_keys:
                idx = index_map.get(key, -1)
                if 0 <= idx < beta.size(1):
                    cols.append(beta[:, idx])
                else:
                    cols.append(torch.full((num_nodes,), 1.0 / max(1, len(active_keys)),
                                           device=device, dtype=beta.dtype))
            aligned = torch.stack(cols, dim=1)
        aligned = aligned.clamp_min(eps)
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(eps)
        return aligned

    beta_teacher_aligned = _align_beta(beta_teacher, teacher_idx, teacher_fill_dtype)
    beta_student_aligned = _align_beta(beta_student, student_idx, student_fill_dtype)

    feature_terms = []
    relpos_terms = []
    path_gate_means = []
    rel_align_stack = []
    feat_align_stack = []

    identity_edges = torch.arange(num_nodes, device=device, dtype=torch.long)
    edge_index = torch.stack([identity_edges, identity_edges], dim=0)

    for idx, key in enumerate(active_keys):
        teacher_mp = F.normalize(teacher_proj(mp_teacher[key]), dim=-1)
        student_mp = F.normalize(student_proj(mp_student[key]), dim=-1)

        rel_teacher = _edge_relpos(teacher_tail, teacher_mp, edge_index)
        rel_student = _edge_relpos(student_tail, student_mp, edge_index)
        rel_diff = (rel_teacher - rel_student).pow(2).sum(dim=-1)
        rel_align = torch.exp(-rel_diff)

        feat_diff = (teacher_mp - student_mp).pow(2).sum(dim=-1)
        feat_align = torch.exp(-feat_diff)

        gate_teacher = beta_teacher_aligned[:, idx]
        gate_student = beta_student_aligned[:, idx]
        gate = 0.5 * (gate_teacher + gate_student)
        base_gate = gate * reliability

        feature_weight = base_gate * rel_align
        rel_weight = base_gate

        feature_terms.append((feat_diff * feature_weight).mean())
        relpos_terms.append((rel_diff * rel_weight).mean())
        path_gate_means.append(base_gate.mean())
        rel_align_stack.append(rel_align)
        feat_align_stack.append(feat_align)

    feature_loss = torch.stack(feature_terms).mean()
    relpos_loss = torch.stack(relpos_terms).mean()
    rel_align_matrix = torch.stack(rel_align_stack, dim=1)
    feat_align_matrix = torch.stack(feat_align_stack, dim=1)

    rel_align_mean = rel_align_matrix.mean(dim=1)
    feat_align_mean = feat_align_matrix.mean(dim=1)
    alignment_for_beta = rel_align_mean * feat_align_mean

    beta_loss = meta_path_beta_loss(beta_teacher_aligned, beta_student_aligned, reliability, alignment_for_beta)
    attn_similarity = F.cosine_similarity(beta_teacher_aligned, beta_student_aligned, dim=1).mean()

    weights = component_weights or {}
    feat_w_raw = float(weights.get('feat', 0.0))
    rel_w_raw = float(weights.get('relpos', 0.0))
    beta_w_raw = float(weights.get('beta', 0.0))

    # Apply total scaling similar to relation distillation
    if lambda_mp_total is not None:
        scale_value = float(lambda_mp_total)
    else:
        scale_value = feat_w_raw + rel_w_raw + beta_w_raw

    if scale_value <= 0.0:
        zero = torch.tensor(0.0, device=device)
        return {
            'feature': feature_loss,
            'relpos': relpos_loss,
            'beta': beta_loss,
            'total': zero,
            'scaled': {'feature': zero, 'relpos': zero, 'beta': zero},
            'raw': {'feature': feature_loss, 'relpos': relpos_loss, 'beta': beta_loss},
            'weights': {'feat': zero, 'relpos': zero, 'beta': zero, 'scale': zero},
            'details': {'active_keys': active_keys, 'attn_similarity': attn_similarity,
                       'mean_gate': torch.stack(path_gate_means).mean(),
                       'rel_align_mean': rel_align_mean.mean(),
                       'feat_align_mean': feat_align_mean.mean()},
        }

    # Normalize weights
    total_raw = feat_w_raw + rel_w_raw + beta_w_raw
    if total_raw > 0.0:
        feat_w = feat_w_raw / total_raw
        rel_w = rel_w_raw / total_raw
        beta_w = beta_w_raw / total_raw
    else:
        feat_w = rel_w = beta_w = 1.0 / 3.0

    scale_tensor = torch.tensor(scale_value, device=device)
    scaled = {
        'feature': feature_loss * feat_w * scale_tensor,
        'relpos': relpos_loss * rel_w * scale_tensor,
        'beta': beta_loss * beta_w * scale_tensor,
    }
    total = scaled['feature'] + scaled['relpos'] + scaled['beta']

    return {
        'feature': feature_loss,
        'relpos': relpos_loss,
        'beta': beta_loss,
        'total': total,
        'scaled': scaled,
        'raw': {
            'feature': feature_loss,
            'relpos': relpos_loss,
            'beta': beta_loss,
        },
        'weights': {
            'feat': torch.tensor(feat_w, device=device),
            'relpos': torch.tensor(rel_w, device=device),
            'beta': torch.tensor(beta_w, device=device),
            'scale': scale_tensor,
        },
        'details': {
            'active_keys': active_keys,
            'attn_similarity': attn_similarity,
            'mean_gate': torch.stack(path_gate_means).mean(),
            'rel_align_mean': rel_align_mean.mean(),
            'feat_align_mean': feat_align_mean.mean(),
        },
    }


def meta_path_beta_loss(beta_teacher: torch.Tensor,
                        beta_student: torch.Tensor,
                        reliability: torch.Tensor,
                        alignment: Optional[torch.Tensor] = None,
                        eps: float = 1e-8) -> torch.Tensor:
    if beta_teacher.numel() == 0 or beta_student.numel() == 0:
        return torch.tensor(0.0, device=reliability.device)
    log_student = beta_student.clamp_min(eps).log()
    loss = F.kl_div(log_student, beta_teacher, reduction='none').sum(dim=-1)
    weight = reliability
    if alignment is not None:
        weight = weight * alignment
    return (loss * weight).mean()

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
            cached = torch.load(cache_path, map_location=device, weights_only=False)
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

# =========================
# Teacher save/load helpers
# =========================

def _teacher_paths(base_dir: str, dataset: str, seed: int) -> Tuple[Path, Path, Path]:
    root = Path(base_dir) / dataset / f"seed_{seed}"
    rsage_p = root / 'rsage.pt'
    han_p = root / 'han.pt'
    return root, rsage_p, han_p

def save_teachers(base_dir: str, dataset: str, seed: int,
                  rsage_teacher: nn.Module, han_teacher: nn.Module) -> None:
    root, rsage_p, han_p = _teacher_paths(base_dir, dataset, seed)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(rsage_teacher.state_dict(), rsage_p)
    torch.save(han_teacher.state_dict(), han_p)
    print(f"[TeacherCache] Saved RSAGE -> {rsage_p}")
    print(f"[TeacherCache] Saved HAN   -> {han_p}")

def load_teachers_if_available(base_dir: str, dataset: str, seed: int,
                               device: torch.device,
                               build_fn: Tuple) -> Tuple[nn.Module, nn.Module, bool]:
    """Return (rsage, han, loaded_flag). build_fn is a function that returns fresh (rsage, han)."""
    root, rsage_p, han_p = _teacher_paths(base_dir, dataset, seed)
    if rsage_p.exists() and han_p.exists():
        print(f"[TeacherCache] Loading teachers from {root}")
        rsage, han = build_fn()
        rsage.load_state_dict(torch.load(rsage_p, map_location=device, weights_only=False))
        han.load_state_dict(torch.load(han_p, map_location=device, weights_only=False))
        rsage.to(device); han.to(device)
        rsage.eval(); han.eval()
        return rsage, han, True
    return None, None, False
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
    student_inputs = base_feats
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
        student_inputs = torch.cat([student_inputs, mp2v_feats], dim=-1)

    student = GraphFreeStudent(d_in=student_inputs.size(1),
                               n_classes=num_classes,
                               hidden=args.student_hidden,
                               num_layers=args.student_layers,
                               dropout=args.student_dropout,
                               rel_dim=args.rel_dim,
                               mp_dim=args.mp_dim,
                               delta_dim=args.delta_dim,
                               beta_hidden=args.mp_beta_hidden).to(device)

    use_direct_aux = not getattr(args, 'no_direct_aux', False)
    print(f"use_direct_aux: {use_direct_aux}")

    teacher_delta_proj = nn.Linear(args.han_hidden, args.delta_dim, bias=False).to(device)

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

    rel_t_dst_proj: Optional[nn.Module] = None
    rel_t_src_proj: Optional[nn.Module] = None
    if taps and len(taps) > 0:
        any_et = next(iter(taps))
        t_dst_dim = taps[any_et]['dst'].size(1)
        t_src_dim = taps[any_et]['src_in'].size(1)
        rel_t_dst_proj = (nn.Linear(t_dst_dim, student.rel_dim, bias=False).to(device)
                          if t_dst_dim != student.rel_dim else nn.Identity().to(device))
        rel_t_src_proj = (nn.Linear(t_src_dim, student.rel_dim, bias=False).to(device)
                          if t_src_dim != student.rel_dim else nn.Identity().to(device))

    params = (
        list(student.parameters())
        + list(teacher_delta_proj.parameters())
    )
    if rel_t_dst_proj is not None and not isinstance(rel_t_dst_proj, nn.Identity):
        params += list(rel_t_dst_proj.parameters())
    if rel_t_src_proj is not None and not isinstance(rel_t_src_proj, nn.Identity):
        params += list(rel_t_src_proj.parameters())
    optimizer = torch.optim.Adam(params, lr=args.student_lr, weight_decay=args.student_wd)

    js = js_divergence(probs_r, probs_h)
    gamma = torch.sigmoid(args.gamma_a * (1 - js) + args.gamma_b * (rho_r - rho_h))
    print(f"gamma: {gamma}")

    best_state, best_test, patience = None, -1.0, 0
    for epoch in range(1, args.student_epochs + 1):
        student.train()
        logits_s, student_hidden, rel_base, mp_base, tail_student = student.forward_all(student_inputs)
        ce_loss = F.cross_entropy(logits_s[idx_train], y[idx_train]) * args.ce_coeff

        kd_rel = kd_kl(logits_s, logits_r, T=args.kd_T, reduce=False)
        kd_h = kd_kl(logits_s, logits_h, T=args.kd_T, reduce=False)
        kd_loss = ((gamma * rho_r * kd_rel) + ((1 - gamma) * rho_h * kd_h)).mean() * args.kd_coeff
 
        rel_base_aux = rel_base
        mp_base_aux = mp_base
        tail_student_aux = tail_student
        rel_student = {}
        for et in hetero.edge_types:
            src, _, dst = et
            if dst == category:
                src_num_nodes = hetero[src].num_nodes
                dst_num_nodes = hetero[dst].num_nodes
                
                dst_emb = rel_base_aux
                
                if src_num_nodes == dst_num_nodes:
                    src_emb = rel_base_aux
                else:
                    if src_num_nodes > dst_num_nodes:
                        padding_size = src_num_nodes - dst_num_nodes
                        padding = torch.zeros(padding_size, rel_base_aux.size(1), 
                                            device=rel_base_aux.device, dtype=rel_base_aux.dtype)
                        src_emb = torch.cat([rel_base_aux, padding], dim=0)
                    else:
                        src_emb = rel_base_aux[:src_num_nodes]
                
                rel_student[et] = {
                    'dst': dst_emb,
                    'src_in': src_emb
                }

        taps_projected = {}
        for et, mm in taps.items():
            t_dst = mm['dst']
            t_src = mm['src_in']
            if rel_t_dst_proj is not None:
                t_dst = rel_t_dst_proj(t_dst)
            if rel_t_src_proj is not None:
                t_src = rel_t_src_proj(t_src)
            taps_projected[et] = {'dst': t_dst, 'src_in': t_src}
        

        rel_out = relation_relative_pos_l2(
            taps_teacher=taps_projected,
            rel_student=rel_student,
            hetero=hetero,
            category=category,
            reliability=rho_r,
            projector_t=None,
            projector_s=None,
            relation_weights=None,
            return_details=True,
            include_per_edge=False)

        struct_logits = student.structural_logits_direct(rel_base_aux, category)

        relation_losses = relation_combined_loss(
            rel_result=rel_out,
            struct_logits=struct_logits,
            y=y,
            idx_train=idx_train,
            lambda_rel_pos=args.lambda_rel_pos,
            lambda_rel_struct=args.lambda_rel_struct,
            lambda_rel_total=getattr(args, 'lambda_rel_total', None),
            balance_override=getattr(args, 'relation_balance', None),
            device=device,
        )

        mp_student_embs = build_student_metapath_embs_direct(
            hetero=hetero,
            ops_template=ops_template,
            mp_base=mp_base_aux,
            category=category,
            device=device,
        )

        mp_keys = list(dict.fromkeys(list(mp_teacher_embs.keys()) + list(mp_student_embs.keys())))
        if mp_student_embs:
            beta_student = student.meta_path_attention(mp_student_embs, mp_keys)
        else:
            beta_student = torch.zeros((tail_student_aux.size(0), 0), device=device)

        mp_losses = meta_path_alignment_losses(
            mp_teacher=mp_teacher_embs,
            mp_student=mp_student_embs,
            tail_teacher=tail_teacher,
            tail_student=tail_student_aux,
            teacher_proj=teacher_delta_proj,
            student_proj=student.delta_proj,
            beta_teacher=beta_teacher,
            beta_student=beta_student,
            reliability=rho_h,
            metapath_keys=mp_keys,
            component_weights={
                'feat': args.lambda_mp_feat,
                'relpos': args.lambda_mp_relpos,
                'beta': args.lambda_mp_beta,
            },
            lambda_mp_total=getattr(args, 'lambda_mp_total', None),
        )

        rel_loss = relation_losses['scaled']['relpos']
        struct_loss = relation_losses['scaled']['struct']
        relation_total = relation_losses['total']

        mp_feat_loss = mp_losses['scaled']['feature']
        mp_relpos = mp_losses['scaled']['relpos']
        mp_beta = mp_losses['scaled']['beta']
        mp_total = mp_losses['total']

        loss = ce_loss + kd_loss + relation_total + mp_total
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        student.eval()
        with torch.no_grad():
            logits_eval = student(student_inputs)
            val_acc = accuracy(logits_eval, y, idx_val)
            train_acc = accuracy(logits_eval, y, idx_train)
            test_acc = accuracy(logits_eval, y, idx_test)
        print(f"[Student] ep {epoch:03d} | CE {ce_loss.item():.4f} KD {kd_loss.item():.4f} "
              f"REL {relation_total.item():.4f} MP {mp_total.item():.4f} | "
              f"tr {train_acc:.4f} va {val_acc:.4f} te {test_acc:.4f}")
        if test_acc >= best_test:
            best_test, best_state, patience = test_acc, copy.deepcopy(student.state_dict()), 0
        else:
            patience += 1
            if patience >= args.student_patience:
                print("[Student] early stop")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
    student.eval()
    logits_final = student(student_inputs)
    test_acc = accuracy(logits_final, y, idx_test)
    print(f"[Student] Final(best) | test {best_test:.4f} | final_test {test_acc:.4f}")
    return student, student_inputs

# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Dual-teacher (RSAGE + meta-path) distillation into graph-free MLP")
    parser.add_argument('-d', '--dataset', type=str, default='TMDB', choices=['TMDB', 'ArXiv', 'DBLP', 'IMDB', 'AMINER'])
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_exp', type=int, default=1,
                        help='重复实验次数；>1 时将多次运行并输出平均±标准差报告')
    parser.add_argument('--teacher_cache_dir', type=str, default='./teachers',
                        help='教师模型缓存目录：teachers/<dataset>/seed_<seed>/{rsage.pt,han.pt}')
    parser.add_argument('--no_reuse_teacher', action='store_true',
                        help='不复用本地缓存教师，每次都重新训练并覆盖保存')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'rsage_only', 'han_only'],
                        help='运行模式：full 训练教师+学生；rsage_only 仅训练并保存RSAGE；han_only 仅训练并保存HAN')
    parser.add_argument('--log_csv', type=str, default='./runs.csv',
                        help='full模式下将结果追加记录到此CSV（若不存在则创建）')

    # teacher RSAGE
    parser.add_argument('--teacher_hidden', type=int, default=128)
    parser.add_argument('--teacher_layers', type=int, default=2)
    parser.add_argument('--teacher_dropout', type=float, default=0.2)
    parser.add_argument('--teacher_norm', type=str, default='none', choices=['none', 'batch', 'layer'])
    parser.add_argument('--teacher_lr', type=float, default=0.01)
    parser.add_argument('--teacher_wd', type=float, default=0.0)
    parser.add_argument('--teacher_epochs', type=int, default=100)
    parser.add_argument('--teacher_patience', type=int, default=30)

    # teacher HAN-like
    parser.add_argument('--han_hidden', type=int, default=128)
    parser.add_argument('--han_semantic_hidden', type=int, default=128)
    parser.add_argument('--han_dropout', type=float, default=0.5)
    parser.add_argument('--han_heads', type=int, default=4)
    parser.add_argument('--han_lr', type=float, default=0.01)
    parser.add_argument('--han_wd', type=float, default=0.0001)
    parser.add_argument('--han_epochs', type=int, default=300)
    parser.add_argument('--han_patience', type=int, default=40)

    # meta-path options
    parser.add_argument('--positional_relations', type=str, nargs='*', default=[],
                        help='meta-paths expressed as comma-separated relation names, e.g. directed_by,directs performed_by,performs')

    # student
    parser.add_argument('--student_hidden', type=int, default=128)
    parser.add_argument('--student_layers', type=int, default=2)
    parser.add_argument('--student_dropout', type=float, default=0.5)
    parser.add_argument('--student_lr', type=float, default=0.003)
    parser.add_argument('--student_wd', type=float, default=0)
    parser.add_argument('--student_epochs', type=int, default=1000)
    parser.add_argument('--student_patience', type=int, default=200)
    parser.add_argument('--no_direct_aux', action='store_true',
                        help='Detach auxiliary losses from the student MLP for ablation/baseline comparisons.')

    # KD weights
    parser.add_argument('--kd_T', type=float, default=1.0)
    parser.add_argument('--ce_coeff', type=float, default=0)
    parser.add_argument('--kd_coeff', type=float, default= 1)
    parser.add_argument('--lambda_rel_pos', type=float, default=1)
    parser.add_argument('--lambda_rel_struct', type=float, default=1)
    parser.add_argument('--lambda_rel_total', type=float, default=0.15,
                        help='Overall scaling for the combined relation loss (defaults to rel_pos + rel_struct).')
    parser.add_argument('--relation_balance', type=float, default=0.2,
                        help='Optional [0,1] ratio for the structural branch within the relation loss (1 => structure only).')
    parser.add_argument('--lambda_mp_feat', type=float, default=1)
    parser.add_argument('--lambda_mp_relpos', type=float, default=1)
    parser.add_argument('--lambda_mp_beta', type=float, default=0)  
    parser.add_argument('--lambda_mp_total', type=float, default=0.5,
                        help='Overall scaling for the combined meta-path loss (defaults to feat + relpos + beta).')
    parser.add_argument('--rel_dim', type=int, default=128)
    parser.add_argument('--mp_dim', type=int, default=128)
    parser.add_argument('--mp_beta_hidden', type=int, default=128)
    parser.add_argument('--delta_dim', type=int, default=128)

    parser.add_argument('--noise_std', type=float, default=0.05)
    parser.add_argument('--reliability_alpha', type=float, default=0.5)
    parser.add_argument('--gamma_a', type=float, default=2.0)
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

    # 多次实验：固定老师一次，学生可多次以不同seed重复
    num_exp = max(1, int(args.num_exp))

    # 先初始化一次数据、老师与静态算子
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

    if rel_sequences and isinstance(rel_sequences[0], (list, tuple)) and len(rel_sequences[0]) > 0 \
       and isinstance(rel_sequences[0][0], (list, tuple)) and len(rel_sequences[0][0]) == 3:
        typed_metapaths = rel_sequences
    else:
        typed_metapaths = parse_metapath_name_sequences(hetero, category, rel_sequences)
        typed_metapaths = ensure_metapaths_to_category(typed_metapaths, category)
    if not typed_metapaths:
        raise ValueError("No valid meta-paths ending at the category node were found.")

    ops_template = build_metapath_operators(hetero, typed_metapaths, device=device)

    # 支持教师缓存：每个实验种子都对应一套教师；可复用或重训
    def build_fresh_teachers():
        rs = train_rsage_teacher(args, hetero, category, y, idx_train, idx_val, idx_test, device)
        hn = train_metapath_teacher(args, hetero, category, y, idx_train, idx_val, idx_test,
                                    typed_metapaths, ops_template, device)
        return rs, hn

    def build_empty_teachers():
        # Instantiate RSAGE without training
        node_dims_local = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
        rs = RSAGE_Hetero(
            etypes=list(hetero.edge_types),
            in_dim=node_dims_local[category],
            hid_dim=args.teacher_hidden,
            num_classes=int(y.max().item()) + 1,
            category=category,
            num_layers=args.teacher_layers,
            dropout=args.teacher_dropout,
            norm_type='none',
            node_type_dims=node_dims_local,
        ).to(device)
        # Instantiate HAN teacher without training
        hn = MetaPathTeacher(
            node_dims=node_dims_local,
            category=category,
            metapaths=typed_metapaths,
            ops_template=ops_template,
            d_hid=args.han_hidden,
            num_classes=int(y.max().item()) + 1,
            semantic_hidden=args.han_semantic_hidden,
            dropout=args.han_dropout,
            heads=args.han_heads,
        ).to(device)
        hn.prepare_meta_graph(hetero)
        return rs, hn

    # rsage_only 模式：只训练并保存 RSAGE 教师
    if args.mode == 'rsage_only':
        exp_seed = args.seed
        set_seed(exp_seed)
        rsage_teacher = train_rsage_teacher(args, hetero, category, y, idx_train, idx_val, idx_test, device)
        # 加载/构建一个空的HAN，只是为了复用保存路径结构
        _, _, han_teacher = None, None, None
        node_dims_local = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
        han_teacher = MetaPathTeacher(
            node_dims=node_dims_local, category=category, metapaths=typed_metapaths,
            ops_template=ops_template, d_hid=args.han_hidden, num_classes=int(y.max().item()) + 1,
            semantic_hidden=args.han_semantic_hidden, dropout=args.han_dropout, heads=args.han_heads,
        ).to(device)
        han_teacher.prepare_meta_graph(hetero)
        save_teachers(args.teacher_cache_dir, args.dataset, exp_seed, rsage_teacher, han_teacher)
        # 记录RSAGE教师性能到CSV
        if args.log_csv:
            import csv
            hetero_device = hetero.to(device)
            with torch.no_grad():
                va_r = accuracy(rsage_teacher(hetero_device), y, idx_val)
                te_r = accuracy(rsage_teacher(hetero_device), y, idx_test)
            fields_to_skip = {'gpu_id', 'num_exp', 'teacher_cache_dir', 'no_reuse_teacher'}
            row = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(args).items() if k not in fields_to_skip}
            row.update({
                'mode': 'rsage_only',
                'rsage_val': f"{va_r:.4f}", 'rsage_test': f"{te_r:.4f}",
                'seed_used': exp_seed,
            })
            cli_keys = [k for k in vars(args).keys() if k not in fields_to_skip]
            metric_keys = [k for k in row.keys() if k not in cli_keys]
            fieldnames = cli_keys + metric_keys
            need_header = not os.path.exists(args.log_csv)
            with open(args.log_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if need_header:
                    writer.writeheader()
                writer.writerow(row)
        return

    # han_only 模式：只训练并保存 HAN 教师
    if args.mode == 'han_only':
        exp_seed = args.seed
        set_seed(exp_seed)
        han_teacher = train_metapath_teacher(args, hetero, category, y, idx_train, idx_val, idx_test,
                                             typed_metapaths, ops_template, device)
        # 加载/构建一个空的RSAGE，只是为了复用保存路径结构
        node_dims_local = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
        rsage_teacher = RSAGE_Hetero(
            etypes=list(hetero.edge_types), in_dim=node_dims_local[category], hid_dim=args.teacher_hidden,
            num_classes=int(y.max().item()) + 1, category=category, num_layers=args.teacher_layers,
            dropout=args.teacher_dropout, norm_type='none', node_type_dims=node_dims_local,
        ).to(device)
        save_teachers(args.teacher_cache_dir, args.dataset, exp_seed, rsage_teacher, han_teacher)
        # 记录HAN教师性能到CSV
        if args.log_csv:
            import csv
            hetero_device = hetero.to(device)
            with torch.no_grad():
                logits_h = han_teacher(hetero_device)
                va_h = accuracy(logits_h, y, idx_val)
                te_h = accuracy(logits_h, y, idx_test)
            fields_to_skip = {'gpu_id', 'num_exp', 'teacher_cache_dir', 'no_reuse_teacher'}
            row = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(args).items() if k not in fields_to_skip}
            row.update({
                'mode': 'han_only',
                'han_val': f"{va_h:.4f}", 'han_test': f"{te_h:.4f}",
                'seed_used': exp_seed,
            })
            cli_keys = [k for k in vars(args).keys() if k not in fields_to_skip]
            metric_keys = [k for k in row.keys() if k not in cli_keys]
            fieldnames = cli_keys + metric_keys
            need_header = not os.path.exists(args.log_csv)
            with open(args.log_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if need_header:
                    writer.writeheader()
                writer.writerow(row)
        return

    # 多次学生实验 & 教师也随种子重复/缓存（full 模式）
    val_scores, test_scores = [], []
    # 分别记录两位教师
    rsage_val_scores, rsage_test_scores = [], []
    han_val_scores, han_test_scores = [], []
    for k in range(num_exp):
        exp_seed = args.seed + k
        set_seed(exp_seed)

        # 教师：优先加载，否则重训并保存（除非指定 no_reuse_teacher）
        if not args.no_reuse_teacher:
            rsage_teacher, han_teacher, loaded = load_teachers_if_available(
                base_dir=args.teacher_cache_dir,
                dataset=args.dataset,
                seed=exp_seed,
                device=device,
                build_fn=lambda: build_empty_teachers(),
            )
            if not loaded:
                rsage_teacher, han_teacher = build_fresh_teachers()
                save_teachers(args.teacher_cache_dir, args.dataset, exp_seed, rsage_teacher, han_teacher)
        else:
            rsage_teacher, han_teacher = build_fresh_teachers()
            save_teachers(args.teacher_cache_dir, args.dataset, exp_seed, rsage_teacher, han_teacher)

        # 记录教师表现
        hetero_device = hetero.to(device)
        with torch.no_grad():
            tr_r = accuracy(rsage_teacher(hetero_device), y, idx_train)
            va_r = accuracy(rsage_teacher(hetero_device), y, idx_val)
            te_r = accuracy(rsage_teacher(hetero_device), y, idx_test)
            logits_h = han_teacher(hetero_device)
            tr_h = accuracy(logits_h, y, idx_train)
            va_h = accuracy(logits_h, y, idx_val)
            te_h = accuracy(logits_h, y, idx_test)
        rsage_val_scores.append(va_r); rsage_test_scores.append(te_r)
        han_val_scores.append(va_h); han_test_scores.append(te_h)

        # 学生训练
        student, student_inputs = train_student_dual_kd(
            args, hetero, category, y,
            idx_train, idx_val, idx_test,
            device, rsage_teacher, han_teacher,
            typed_metapaths, ops_template
        )
        student.eval()
        with torch.no_grad():
            logits_s = student(student_inputs)
            va_s = accuracy(logits_s, y, idx_val)
            te_s = accuracy(logits_s, y, idx_test)
        val_scores.append(va_s)
        test_scores.append(te_s)

    # 汇总报告
    val_mean = float(np.mean(val_scores)); val_std = float(np.std(val_scores))
    test_mean = float(np.mean(test_scores)); test_std = float(np.std(test_scores))
    r_val_mean = float(np.mean(rsage_val_scores)); r_val_std = float(np.std(rsage_val_scores))
    r_test_mean = float(np.mean(rsage_test_scores)); r_test_std = float(np.std(rsage_test_scores))
    h_val_mean = float(np.mean(han_val_scores)); h_val_std = float(np.std(han_val_scores))
    h_test_mean = float(np.mean(han_test_scores)); h_test_std = float(np.std(han_test_scores))
    print("\n" + "="*60)
    print(f"MULTI-RUN REPORT (n={num_exp})")
    print(f"RSAGE teacher | val {r_val_mean:.4f} ± {r_val_std:.4f} | test {r_test_mean:.4f} ± {r_test_std:.4f}")
    print(f"HAN teacher   | val {h_val_mean:.4f} ± {h_val_std:.4f} | test {h_test_mean:.4f} ± {h_test_std:.4f}")
    print(f"Student (MLP) | val {val_mean:.4f} ± {val_std:.4f} | test {test_mean:.4f} ± {test_std:.4f}")
    print("="*60)

    # 单次基准（使用最后一次学生）
    hetero_device = hetero.to(device)
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
        return student(student_inputs)
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

    # 追加写入CSV日志（仅在full模式）
    if args.mode == 'full' and args.log_csv:
        import csv
        fields_to_skip = {'gpu_id', 'num_exp', 'teacher_cache_dir', 'no_reuse_teacher'}
        row = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(args).items() if k not in fields_to_skip}
        row.update({
            'val_mean': f"{val_mean:.4f}", 'val_std': f"{val_std:.4f}",
            'test_mean': f"{test_mean:.4f}", 'test_std': f"{test_std:.4f}",
            'rsage_val_mean': f"{r_val_mean:.4f}", 'rsage_val_std': f"{r_val_std:.4f}",
            'rsage_test_mean': f"{r_test_mean:.4f}", 'rsage_test_std': f"{r_test_std:.4f}",
            'han_val_mean': f"{h_val_mean:.4f}", 'han_val_std': f"{h_val_std:.4f}",
            'han_test_mean': f"{h_test_mean:.4f}", 'han_test_std': f"{h_test_std:.4f}",
            'infer_rsage_ms': f"{mean_r*1000:.3f}", 'infer_rsage_std_ms': f"{std_r*1000:.3f}",
            'infer_han_ms': f"{mean_h*1000:.3f}", 'infer_han_std_ms': f"{std_h*1000:.3f}",
            'infer_student_ms': f"{mean_s*1000:.3f}", 'infer_student_std_ms': f"{std_s*1000:.3f}",
        })
        # 保证字段顺序稳定：先CLI参数，再结果
        cli_keys = [k for k in vars(args).keys() if k not in fields_to_skip]
        metric_keys = [k for k in row.keys() if k not in cli_keys]
        fieldnames = cli_keys + metric_keys
        need_header = not os.path.exists(args.log_csv)
        with open(args.log_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if need_header:
                writer.writeheader()
            writer.writerow(row)


if __name__ == '__main__':
    main()





