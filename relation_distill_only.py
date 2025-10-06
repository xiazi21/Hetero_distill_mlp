# -*- coding: utf-8 -*-
"""
Single-file: Relation-Type Distillation (+ logits KD) with Reliability
- Teacher: RSAGE_Hetero (heterogeneous GraphSAGE)
- Student: Plain MLP classifier (graph-free at inference)
- Train-time student relation adapter (RelKDAdapter) for relation-level KD
- Total loss: CE + KD(logits) + λ_pos * Relation-Relative-Position L2 (all KD terms reliability-weighted)
"""

import os
import math
import copy
import time
import random
import argparse
import hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn.models import MetaPath2Vec
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_sparse import SparseTensor

# ---- dataset loader (reuse from repo) ----
from dataloader import load_data


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

# ---- your dataset loader (reuse from your repo) ----
from dataloader import load_data  # must return (hetero, (idx_train, idx_val, idx_test), gen_node_feats, metapaths)


# =========================
# Utilities
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
    print(f"[INFO] Seed set to {seed} (deterministic)")

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    pred = logits[idx].argmax(dim=-1)
    return (pred == y[idx]).float().mean().item()

def kd_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 2.0, reduce: bool = True):
    """Standard temperature-scaled KL divergence (teacher || student)."""
    ps = F.log_softmax(student_logits / T, dim=-1)
    pt = F.log_softmax(teacher_logits / T, dim=-1)
    kl = torch.sum(torch.exp(pt) * (pt - ps), dim=-1) * (T * T)
    return kl.mean() if reduce else kl  # [N] if reduce=False


# =========================
# Teacher: RSAGE_Hetero
# =========================
class RSAGE_Hetero(nn.Module):
    """
    Heterogeneous GraphSAGE teacher.
    - Each layer: HeteroConv with per-relation SAGEConv(aggr='mean')
    - Returns logits for category node type
    - forward_with_relation_taps(): exposes per-relation 'dst' (last-layer out) and 'src_in' (last-layer in)
    """
    def __init__(self,
                 etypes: List[Tuple[str, str, str]],
                 in_dim: int, hid_dim: int, num_classes: int,
                 category: str, num_layers: int = 2, dropout: float = 0.2,
                 norm_type: str = "none",
                 node_type_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.category = category
        self.etypes = list(etypes)
        self.num_layers = int(num_layers)
        self.hid_dim = int(hid_dim)
        self.num_classes = int(num_classes)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        def conv_dict(in_channels, out_channels):
            return {etype: SAGEConv(in_channels, out_channels, aggr='mean') for etype in self.etypes}

        # Per-node-type projector to unify feature dims to in_dim (handles datasets like DBLP)
        self.node_type_dims = node_type_dims
        self.projectors = nn.ModuleDict()
        if node_type_dims is not None:
            for nt, dim in node_type_dims.items():
                if dim != in_dim:
                    self.projectors[nt] = nn.Linear(dim, in_dim, bias=False)
                else:
                    self.projectors[nt] = nn.Identity()

        if self.num_layers <= 1:
            self.layers.append(HeteroConv(conv_dict(in_dim, num_classes), aggr='mean'))
        else:
            self.layers.append(HeteroConv(conv_dict(in_dim, hid_dim), aggr='mean'))
            for _ in range(self.num_layers - 2):
                self.layers.append(HeteroConv(conv_dict(hid_dim, hid_dim), aggr='mean'))
            self.layers.append(HeteroConv(conv_dict(hid_dim, num_classes), aggr='mean'))

        self.use_norm = False
        if norm_type == "batch":
            self.use_norm = True
            norm_cls = nn.BatchNorm1d
        elif norm_type == "layer":
            self.use_norm = True
            norm_cls = nn.LayerNorm
        else:
            norm_cls = None

        if len(self.layers) > 1 and self.use_norm:
            for _ in range(len(self.layers) - 1):
                self.norms.append(norm_cls(hid_dim))

    def _apply_each(self, xdict: Dict[str, torch.Tensor], fn):
        return {k: fn(v) for k, v in xdict.items()}

    def forward(self, data: HeteroData, feats_override: Optional[torch.Tensor] = None):
        x_dict = {k: data[k].x for k in data.node_types}
        if feats_override is not None:
            x_dict[self.category] = feats_override

        # Apply projectors if defined to unify all node-type feature dims
        if len(self.projectors) > 0:
            x_dict = {k: self.projectors[k](v) if k in self.projectors else v for k, v in x_dict.items()}

        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        h = x_dict
        for l, layer in enumerate(self.layers):
            h = layer(h, edge_index_dict)
            h = {k: v.view(v.shape[0], -1) for k, v in h.items()}
            if l != len(self.layers) - 1:
                if len(self.norms) > 0 and l < len(self.norms):
                    h = {k: self.norms[l](v) for k, v in h.items()}
                h = self._apply_each(h, self.activation)
                h = self._apply_each(h, self.dropout)
        logits_cat = h[self.category]
        return logits_cat

    @torch.no_grad()
    def forward_with_relation_taps(self, data: HeteroData, feats_override: Optional[torch.Tensor] = None):
        x_dict = {k: data[k].x for k in data.node_types}
        if feats_override is not None:
            x_dict[self.category] = feats_override

        # Apply projectors if defined to unify all node-type feature dims
        if len(self.projectors) > 0:
            x_dict = {k: self.projectors[k](v) if k in self.projectors else v for k, v in x_dict.items()}

        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        # up to last layer
        h = x_dict
        for l, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index_dict)
            h = {k: v.view(v.shape[0], -1) for k, v in h.items()}
            if len(self.norms) > 0 and l < len(self.norms):
                h = {k: self.norms[l](v) for k, v in h.items()}
            h = self._apply_each(h, self.activation)
            h = self._apply_each(h, self.dropout)

        h_in_dict = {k: v.view(v.shape[0], -1) for k, v in h.items()}

        # last heteroconv per relation
        last_layer: HeteroConv = self.layers[-1]
        taps: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        for (s, r, d), conv in last_layer.convs.items():
            x_src = h_in_dict[s]
            x_dst = h_in_dict[d]
            ei = edge_index_dict[(s, r, d)]
            emb_dst = conv((x_src, x_dst), ei)
            emb_dst = emb_dst.view(emb_dst.shape[0], -1)
            taps[(s, r, d)] = {
                'dst': emb_dst,     # last-layer relation-specific dst output
                'src_in': h_in_dict[s]  # last-layer input for src type
            }

        # full last-layer aggregation for logits
        h_out = last_layer(h_in_dict, edge_index_dict)
        h_out = {k: v.view(v.shape[0], -1) for k, v in h_out.items()}
        logits_cat = h_out[self.category]
        return logits_cat, taps


# =========================
# Student: plain MLP
# =========================
class MLPClassifier(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 256, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        dims = [d_in] + [hidden] * (num_layers - 1) + [n_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers += [nn.ReLU(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================
# Train-time Student Relation Adapter
# =========================
class RelKDAdapter(nn.Module):
    """
    Train-only adapter for student to build per-relation embeddings:
      - Per node type linear projection to d_rel
      - For relation (s, r, d): mean-aggregate projected src -> dst
      - Returns dict[et] -> {'dst': [N_d, d_rel], 'src_in': [N_s, d_rel]}
    """
    def __init__(self, node_dims: Dict[str, int], d_rel: int):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(node_dims[nt], d_rel, bias=False) for nt in node_dims})
        self.d_rel = d_rel

    @torch.no_grad()
    def _build_adj(self, hetero: HeteroData, et: Tuple[str, str, str]):
        s, r, d = et
        ei = hetero[et].edge_index.to(hetero[s].x.device)
        ns = hetero[s].num_nodes
        nd = hetero[d].num_nodes
        deg = torch.bincount(ei[1], minlength=nd).clamp_min(1).float()
        val = torch.ones(ei.size(1), device=ei.device, dtype=torch.float32)
        # dst x src
        A = SparseTensor(row=ei[1], col=ei[0], value=val, sparse_sizes=(nd, ns)).coalesce()
        return A, deg

    def forward(self, hetero: HeteroData, etypes: List[Tuple[str, str, str]],
                node_overrides: Optional[Dict[str, torch.Tensor]] = None):
        rel_embs: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        x_proj = {}
        for nt in hetero.node_types:
            feats = node_overrides[nt] if node_overrides and nt in node_overrides else hetero[nt].x
            x_proj[nt] = self.proj[nt](feats)
        for et in etypes:
            s, r, d = et
            A, deg = self._build_adj(hetero, et)  # dst x src
            dst = (A @ x_proj[s]) / deg.unsqueeze(-1)
            rel_embs[et] = {'dst': dst, 'src_in': x_proj[s]}
        return rel_embs


class RelationStructuralHead(nn.Module):
    """Turns relation-specific embeddings (dst) into logits for the category nodes."""

    def __init__(self,
                 relations: List[Tuple[str, str, str]],
                 category: str,
                 rel_dim: int,
                 num_classes: int):
        super().__init__()
        self.category = category
        self.rel_list = [et for et in relations if et[2] == category]
        self.rel_heads = nn.ModuleDict()
        for et in self.rel_list:
            key = self._rel_key(et)
            self.rel_heads[key] = nn.Linear(rel_dim, num_classes, bias=False)

    @staticmethod
    def _rel_key(et: Tuple[str, str, str]) -> str:
        return f"{et[0]}__{et[1]}__{et[2]}"

    @property
    def has_relations(self) -> bool:
        return len(self.rel_list) > 0

    def forward(self, rel_embs: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]]) -> Optional[torch.Tensor]:
        if not self.has_relations:
            return None
        logits = None
        for et in self.rel_list:
            if et not in rel_embs:
                continue
            dst_feat = rel_embs[et]['dst']
            head = self.rel_heads[self._rel_key(et)]
            contrib = head(dst_feat)
            logits = contrib if logits is None else logits + contrib
        return logits


# =========================
# Reliability from teacher
# =========================
@torch.no_grad()
def run_teacher_and_reliability(teacher: nn.Module,
                                hetero: HeteroData,
                                category: str,
                                device: torch.device,
                                enhanced_features: Optional[torch.Tensor] = None,
                                noise_std: float = 0.05,
                                alpha: float = 0.5):
    teacher.eval()
    hetero = hetero.to(device)

    logits = teacher(hetero, enhanced_features)
    if isinstance(logits, tuple):  # safety
        _, logits = logits
    probs = torch.softmax(logits, dim=-1)

    if enhanced_features is not None:
        x_clean = enhanced_features.clone()
        x_noise = x_clean + noise_std * torch.randn_like(x_clean)
        logits_noise = teacher(hetero, x_noise)
    else:
        x_clean = hetero[category].x.clone()
        hetero[category].x = x_clean + noise_std * torch.randn_like(x_clean)
        logits_noise = teacher(hetero, None)
        hetero[category].x = x_clean

    if isinstance(logits_noise, tuple):
        _, logits_noise = logits_noise
    probs_noise = torch.softmax(logits_noise, dim=-1)

    # confidence (normalized entropy)
    ent = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    ent = (ent / math.log(probs.size(1))).clamp(0, 1)
    conf = 1.0 - ent
    # stability (agreement under noise)
    stab = torch.exp(- (probs - probs_noise).abs().sum(dim=-1) / 0.5)

    reliability = (alpha * conf + (1 - alpha) * stab).clamp(0, 1)
    return logits.detach(), probs.detach(), reliability.detach()


# =========================
# Relation-Type Distillation Loss (relative position only)
# =========================
def _edge_relpos(emb_dst: torch.Tensor, emb_src_in: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
    src, dst = ei[0].long(), ei[1].long()
    return emb_dst[dst] - emb_src_in[src]  # [E, D]

def relation_relative_pos_l2(
    taps_teacher: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
    rel_embs_student: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]],
    hetero: HeteroData,
    category: str,
    reliability: Optional[torch.Tensor] = None,
    projector_t: Optional[nn.Module] = None,
    projector_s: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Mean L2 over edges (dst == category), optionally reliability-weighted by dst node reliability."""
    l2_all = []
    device = next(iter(taps_teacher.values()))['dst'].device if taps_teacher else hetero[category].x.device
    for et in hetero.edge_types:
        s, r, d = et
        if d != category:  # only edges whose dst are category nodes to align reliability & indexing
            continue
        if (et not in taps_teacher) or (et not in rel_embs_student):
            continue

        ei = hetero[et].edge_index.to(device)  # [2, E]
        emb_t_dst = taps_teacher[et]['dst']
        emb_t_src = taps_teacher[et]['src_in']
        emb_s_dst = rel_embs_student[et]['dst'].to(device)
        emb_s_src = rel_embs_student[et]['src_in'].to(device)

        if projector_t is not None:
            emb_t_dst = projector_t(emb_t_dst)
            emb_t_src = projector_t(emb_t_src)
        if projector_s is not None:
            emb_s_dst = projector_s(emb_s_dst)
            emb_s_src = projector_s(emb_s_src)

        if emb_t_dst.size(1) != emb_s_dst.size(1):
            raise ValueError(f"[RelKD] dim mismatch on {et}: T={emb_t_dst.size(1)} vs S={emb_s_dst.size(1)}")

        rel_t = _edge_relpos(emb_t_dst, emb_t_src, ei)  # [E, D]
        rel_s = _edge_relpos(emb_s_dst, emb_s_src, ei)  # [E, D]
        l2_e = (rel_t - rel_s).pow(2).sum(dim=-1) / rel_t.size(1)  # [E] normalized by D

        if reliability is not None:
            l2_e = l2_e * reliability[ei[1].long()]  # weight by dst-node reliability

        l2_all.append(l2_e.mean())

    if len(l2_all) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(l2_all).mean()


def metapath2vec_category_embeddings(hetero: HeteroData,
                                     metapaths: List[List[Tuple[str, str, str]]],
                                     category: str,
                                     emb_dim: int = 128,
                                     walk_length: int = 40,
                                     context_size: int = 5,
                                     walks_per_node: int = 10,
                                     epochs: int = 50,
                                     device: str = "cpu",
                                     cache_dir: Optional[str] = None,
                                     seed: Optional[int] = None) -> torch.Tensor:
    if not metapaths:
        return torch.zeros((hetero[category].num_nodes, emb_dim), dtype=torch.float32)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_key_repr = {
            "metapaths": metapaths,
            "emb_dim": emb_dim,
            "walk_length": walk_length,
            "context_size": context_size,
            "walks_per_node": walks_per_node,
            "epochs": epochs,
        }
        cache_key = hashlib.md5(repr(cache_key_repr).encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"mp2v_{category}_{cache_key}.pt")
        if os.path.exists(cache_path):
            print(f"[PE] Loading cached MetaPath2Vec embeddings from {cache_path}")
            cached = torch.load(cache_path, map_location=device)
            if cached.size(0) == hetero[category].num_nodes and cached.size(1) == emb_dim:
                return cached.to(device)
            print("[PE] Cached embeddings shape mismatch; rebuilding.")
    else:
        cache_path = None

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    print(f"Training MetaPath2Vec for positional encoding with {epochs} epochs...")
    edge_index_dict = {et: hetero[et].edge_index for et in hetero.edge_types}
    per_path_dim = max(8, emb_dim // max(1, len(metapaths)))
    all_embs = []
    for mp_idx, mp in enumerate(metapaths):
        print(f"Training MetaPath2Vec for metapath {mp_idx + 1}/{len(metapaths)}: {mp}")
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
        opt = torch.optim.SparseAdam(mp2v.parameters(), lr=0.01)
        mp2v.train()
        for ep in range(epochs):
            total = 0.0
            for pos_rw, neg_rw in mp2v.loader(batch_size=512, shuffle=True):
                opt.zero_grad()
                loss = mp2v.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                opt.step()
                total += float(loss)
            print(f"MetaPath2Vec epoch {ep:03d} loss: {total:.4f}")
        with torch.no_grad():
            out = mp2v(category)
            W = out if isinstance(out, torch.Tensor) else out.weight
            W = F.normalize(W, p=2, dim=1)
            mu, std = W.mean(0, keepdim=True), W.std(0, keepdim=True).clamp_min(1e-6)
            W = (W - mu) / std
            all_embs.append(W.detach().cpu())
        mp2v.cpu()
    if not all_embs:
        pe = torch.zeros((hetero[category].num_nodes, emb_dim), dtype=torch.float32)
    else:
        pe = torch.cat(all_embs, dim=1)
        if pe.size(1) < emb_dim:
            pad = torch.zeros(pe.size(0), emb_dim - pe.size(1), dtype=pe.dtype)
            pe = torch.cat([pe, pad], dim=1)
        elif pe.size(1) > emb_dim:
            pe = pe[:, :emb_dim]

    if cache_path:
        torch.save(pe, cache_path)
        print(f"[PE] Saved MetaPath2Vec embeddings to {cache_path}")

    return pe.to(device)


# =========================
# Train / Eval
# =========================
@torch.no_grad()
def evaluate_teacher(teacher: nn.Module,
                     hetero: HeteroData, category: str, y: torch.Tensor,
                     idx_train: torch.Tensor, idx_val: torch.Tensor, idx_test: torch.Tensor,
                     device: torch.device):
    teacher.eval()
    logits = teacher(hetero.to(device), None)
    if isinstance(logits, tuple):
        _, logits = logits
    return (accuracy(logits, y, idx_train),
            accuracy(logits, y, idx_val),
            accuracy(logits, y, idx_test))

def train_teacher(args, hetero, category, y, idx_train, idx_val, idx_test, device):
    # Collect per-node-type feature dims to build projectors when needed (e.g., DBLP)
    node_type_dims = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
    teacher = RSAGE_Hetero(
        etypes=list(hetero.edge_types),
        in_dim=hetero[category].x.size(1),
        hid_dim=args.hid_dim,
        num_classes=int(y.max().item()) + 1,
        category=category,
        num_layers=args.teacher_layers,
        dropout=args.teacher_dropout,
        norm_type='none',
        node_type_dims=node_type_dims,
    ).to(device)

    opt = torch.optim.Adam(teacher.parameters(), lr=args.teacher_lr, weight_decay=args.teacher_wd)
    best_state, best_val, es = None, -1.0, 0

    for ep in range(args.teacher_epochs):
        teacher.train()
        logits = teacher(hetero.to(device), None)
        if isinstance(logits, tuple): _, logits = logits
        loss = F.cross_entropy(logits[idx_train], y[idx_train])
        opt.zero_grad(); loss.backward(); opt.step()

        tr, va, te = evaluate_teacher(teacher, hetero, category, y, idx_train, idx_val, idx_test, device)
        print(f"[Teacher] ep {ep:03d} | loss {loss.item():.4f} | tr {tr:.4f} va {va:.4f} te {te:.4f}")
        if va >= best_val:
            best_val, best_state, es = va, copy.deepcopy(teacher.state_dict()), 0
        else:
            es += 1
            if es >= args.teacher_patience:
                print("[Teacher] early stop")
                break

    if best_state is not None:
        teacher.load_state_dict(best_state)
    tr, va, te = evaluate_teacher(teacher, hetero, category, y, idx_train, idx_val, idx_test, device)
    print(f"[Teacher] Final(best) | tr {tr:.4f} va {va:.4f} te {te:.4f}")
    return teacher


def train_student_relation_kd(args, hetero, category, y, idx_train, idx_val, idx_test, device, metapaths_relnames=None):
    if args.reseed_student:
        seed_offset = args.seed + args.student_seed_offset
        random.seed(seed_offset)
        np.random.seed(seed_offset)
        torch.manual_seed(seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_offset)
            torch.cuda.manual_seed_all(seed_offset)
    # 1) Teacher
    teacher = train_teacher(args, hetero, category, y, idx_train, idx_val, idx_test, device)

    # 2) Teacher logits, taps, reliability (one-shot, full graph)
    teacher.eval()
    hetero_device = hetero.to(device)
    with torch.no_grad():
        t_logits_full, taps = teacher.forward_with_relation_taps(hetero_device, None)
        _, _, reliability = run_teacher_and_reliability(teacher, hetero_device, category, device, None)
        reliability = reliability.detach()

    # detach teacher logits to avoid grads flowing into teacher
    t_logits_full = t_logits_full.detach()
    taps = {et: {k: v.detach() for k, v in tap.items()} for et, tap in taps.items()}

    # 3) Student features (optional structural aug)
    base_feat = hetero_device[category].x
    student_feats = base_feat
    student_node_overrides: Dict[str, torch.Tensor] = {}

    mp_tuples = []
    if args.use_positional_encoding:
        rel_sequences = args.metapaths if args.metapaths else (metapaths_relnames or [])
        if rel_sequences:
            for rel_seq in rel_sequences:
                if not rel_seq:
                    continue
                # Support either relation-name sequences or explicit (src, rel, dst) tuples
                first = rel_seq[0]
                if isinstance(first, (list, tuple)) and len(first) == 3:
                    cur = category
                    tup = []
                    valid = True
                    for edge in rel_seq:
                        if not (isinstance(edge, (list, tuple)) and len(edge) == 3):
                            valid = False
                            break
                        s, rr, d = edge
                        if s != cur or (s, rr, d) not in hetero.edge_types:
                            valid = False
                            break
                        tup.append((s, rr, d))
                        cur = d
                    if valid and cur == category:
                        mp_tuples.append(tup)
                else:
                    tup = []
                    cur = category
                    for r in rel_seq:
                        cand = None
                        for (s, rr, d) in hetero.edge_types:
                            if rr == r and s == cur:
                                cand = (s, rr, d)
                                cur = d
                                break
                        if cand is None:
                            tup = []
                            break
                        tup.append(cand)
                    if tup and tup[-1][2] == category:
                        mp_tuples.append(tup)
            if not mp_tuples:
                print("[PE] No valid metapath tuples found; skipping MetaPath2Vec.")
        else:
            print("[PE] Dataset does not provide metapaths and none were specified; skipping MetaPath2Vec.")

    if mp_tuples:
        pe = metapath2vec_category_embeddings(
            hetero_device, mp_tuples, category,
            emb_dim=args.mp_pe_dim,
            walk_length=args.mp_walk_length,
            context_size=args.mp_context_size,
            walks_per_node=args.mp_walks_per_node,
            epochs=args.mp_epochs,
            device=str(device),
            cache_dir=args.mp_cache_dir,
            seed=args.seed
        ).to(device)
        student_feats = torch.cat([student_feats, pe], dim=1)
        print(f"[PE] MetaPath2Vec embeddings shape: {pe.shape}")

    student_node_overrides[category] = student_feats

    # 4) Student model (graph-free)
    x = student_feats
    num_classes = int(y.max().item()) + 1
    student = MLPClassifier(d_in=x.size(1),
                            n_classes=num_classes,
                            hidden=args.student_hidden,
                            num_layers=args.student_layers,
                            dropout=args.student_dropout).to(device)

    # 5) Train-time student relation adapter + projectors for dim alignment
    node_dims = {nt: hetero_device[nt].x.size(1) for nt in hetero.node_types}
    for nt, feats in student_node_overrides.items():
        node_dims[nt] = feats.size(1)

    rel_adapter = RelKDAdapter(node_dims, d_rel=args.rel_dim).to(device)
    rel_struct_head = RelationStructuralHead(list(hetero.edge_types), category,
                                             args.rel_dim, num_classes).to(device)

    # infer teacher/student relation dims and build projectors (teacher -> target_dim; student is Identity)
    # pick the first relation to read dims (fallback safe)
    if len(taps) > 0:
        first_et = next(iter(taps))
        teach_dst_dim = taps[first_et]['dst'].size(1)
        teach_src_dim = taps[first_et]['src_in'].size(1)
        target_dim = args.rel_dim  # unify to rel_dim

        class TProjector(nn.Module):
            def __init__(self, dst_dim, src_dim, target_dim):
                super().__init__()
                self.dst_proj = nn.Linear(dst_dim, target_dim, bias=False) if dst_dim != target_dim else nn.Identity()
                self.src_proj = nn.Linear(src_dim, target_dim, bias=False) if src_dim != target_dim else nn.Identity()
            def forward_dst(self, x): return self.dst_proj(x)
            def forward_src(self, x): return self.src_proj(x)

        t_proj = TProjector(teach_dst_dim, teach_src_dim, target_dim).to(device)
        s_proj = nn.Identity().to(device)
    else:
        t_proj = None
        s_proj = None

    # 6) Joint training: CE + KD(logits) + λ_pos * rel-pos L2 + structural CE
    params = [
        {'params': student.parameters()},
        {'params': rel_adapter.parameters()},
    ]
    if rel_struct_head.has_relations:
        params.append({
            'params': rel_struct_head.parameters(),
            'lr': args.struct_head_lr,
            'weight_decay': args.struct_head_wd,
        })
    if t_proj is not None:
        params.append({'params': t_proj.parameters()})

    opt = torch.optim.AdamW(params, lr=args.student_lr, weight_decay=args.student_wd)
    best_va, best_state, best_te, es = -1.0, None, 0.0, 0

    for ep in range(args.student_epochs):
        student.train()
        # forward
        s_logits = student(x)  # [N, C]

        # CE on train
        ce = F.cross_entropy(s_logits[idx_train], y[idx_train])

        # KD (reliability-weighted)
        kl_vec = kd_kl(s_logits, t_logits_full, T=args.kd_T, reduce=False)  # [N]
        kd = (kl_vec * reliability).mean()

        # Relation relative position L2 (teacher taps vs student adapter)
        rel_embs_student = rel_adapter(hetero_device, list(hetero.edge_types),
                                       node_overrides=student_node_overrides)
        structural_logits = rel_struct_head(rel_embs_student) if rel_struct_head.has_relations else None
        # wrap teacher projectors if defined
        def _pt(x, kind="dst"):
            if t_proj is None: return x
            return t_proj.forward_dst(x) if kind == "dst" else t_proj.forward_src(x)
        def _ps(x):
            return x  # student proj is identity here

        taps_proj = {}
        for et, mm in taps.items():
            taps_proj[et] = {
                'dst': _pt(mm['dst'], "dst"),
                'src_in': _pt(mm['src_in'], "src"),
            }

        rel_l2 = relation_relative_pos_l2(
            taps_teacher=taps_proj,
            rel_embs_student=rel_embs_student,
            hetero=hetero_device,
            category=category,
            reliability=reliability,
            projector_t=None,   # already projected
            projector_s=None
        )

        struct_ce = torch.tensor(0.0, device=device)
        if structural_logits is not None:
            struct_ce = F.cross_entropy(structural_logits[idx_train], y[idx_train])

        # total loss
        loss = (args.ce_coeff * ce
                + args.kd_coeff * kd
                + args.lambda_rel_pos * rel_l2
                + args.lambda_rel_struct * struct_ce)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # eval
        student.eval()
        with torch.no_grad():
            logits_eval = student(x)
            tr = accuracy(logits_eval, y, idx_train)
            va = accuracy(logits_eval, y, idx_val)
            te = accuracy(logits_eval, y, idx_test)
        print(f"[Student] ep {ep:03d} | CE {ce:.4f} KD {kd:.4f} REL {rel_l2:.4f} STRUCT {struct_ce:.4f} | tr {tr:.4f} va {va:.4f} te {te:.4f}")

        if va >= best_va:
            best_va, best_state, best_te, es = va, copy.deepcopy(student.state_dict()), te, 0
        else:
            es += 1
            if es >= args.student_patience:
                print("[Student] early stop")
                break

    if best_state is not None:
        student.load_state_dict(best_state)
        with torch.no_grad():
            logits_eval = student(x)
            tr = accuracy(logits_eval, y, idx_train)
            va = accuracy(logits_eval, y, idx_val)
            te = accuracy(logits_eval, y, idx_test)
        print(f"[Student] Final(best) | tr {tr:.4f} va {va:.4f} te {te:.4f} (best_te {best_te:.4f})")

    return teacher, student, student_feats


# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser("Relation-only KD (logits + relation relative position, reliability-weighted)")
    p.add_argument("-d", "--dataset", type=str, default="TMDB", choices=["TMDB", "CroVal", "ArXiv", "IGB-tiny-549K-19", "IGB-small-549K-2983", "DBLP", "IMDB"])
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)

    # teacher
    p.add_argument("--hid_dim", type=int, default=128)
    p.add_argument("--teacher_layers", type=int, default=2)
    p.add_argument("--teacher_dropout", type=float, default=0.2)
    p.add_argument("--teacher_lr", type=float, default=1e-2)
    p.add_argument("--teacher_wd", type=float, default=0.0)
    p.add_argument("--teacher_epochs", type=int, default=100)
    p.add_argument("--teacher_patience", type=int, default=20)

    # student
    p.add_argument("--student_hidden", type=int, default=128)
    p.add_argument("--student_layers", type=int, default=2)
    p.add_argument("--student_dropout", type=float, default=0.5)
    p.add_argument("--student_lr", type=float, default=0.01)
    p.add_argument("--student_wd", type=float, default=0)
    p.add_argument("--student_epochs", type=int, default=1000)
    p.add_argument("--student_patience", type=int, default=10)

    # KD + relation loss
    p.add_argument("--kd_T", type=float, default=1.0)
    p.add_argument("--ce_coeff", type=float, default=1, help="weight on CE")
    p.add_argument("--kd_coeff", type=float, default=1, help="weight on KD (reliability-weighted)")
    p.add_argument("--lambda_rel_pos", type=float, default=1, help="weight on relation relative-position L2")
    p.add_argument("--rel_dim", type=int, default=256, help="relation adapter projection dim for student")
    p.add_argument("--lambda_rel_struct", type=float, default=10, help="weight on relation structural CE")
    p.add_argument("--struct_head_lr", type=float, default=2e-3, help="lr for relation structural head")
    p.add_argument("--struct_head_wd", type=float, default=1e-4, help="weight decay for relation structural head")
    p.add_argument("--use_positional_encoding", action="store_true", help="augment student inputs with MetaPath2Vec positional encodings")
    p.add_argument("--positional_relations", type=str, nargs='*', default=[],
                   help="metapaths expressed as comma-separated relation names, e.g. performed_by,performs")
    p.add_argument("--mp_pe_dim", type=int, default=128)
    p.add_argument("--mp_walk_length", type=int, default=40)
    p.add_argument("--mp_context_size", type=int, default=5)
    p.add_argument("--mp_walks_per_node", type=int, default=10)
    p.add_argument("--mp_epochs", type=int, default=50)
    p.add_argument("--mp_cache_dir", type=str, default="./mp2v_cache",
                   help="directory to cache MetaPath2Vec embeddings")
    p.add_argument("--benchmark_warmup", type=int, default=3,
                   help="warmup iterations before measuring inference time (0 to disable)")
    p.add_argument("--benchmark_runs", type=int, default=10,
                   help="number of timed inference runs (0 to disable)")
    p.add_argument("--reseed_student", action="store_true",
                   help="reseed student-specific randomness using seed + student_seed_offset")
    p.add_argument("--student_seed_offset", type=int, default=0,
                   help="offset added to seed when reseeding student components")

    args = p.parse_args()
    args.metapaths = []
    rel_args = getattr(args, 'positional_relations', []) or []
    for item in rel_args:
        seq = [tok.strip() for tok in item.split(',') if tok.strip()]
        if seq:
            args.metapaths.append(seq)
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # load data
    hetero, splits, gen_node_feats, metapaths_relnames = load_data(dataset=args.dataset, return_mp=True)
    hetero = gen_node_feats(hetero)
    category = hetero.category
    idx_train, idx_val, idx_test = [t.to(device) for t in splits]
    y = hetero[category].y.long().to(device)

    if args.use_positional_encoding and not args.metapaths and metapaths_relnames:
        args.metapaths = metapaths_relnames

    # train pipeline
    teacher, student, student_feats = train_student_relation_kd(
        args, hetero, category, y, idx_train, idx_val, idx_test, device, metapaths_relnames
    )

    # Print teacher performance
    print("\n" + "="*60)
    print("TEACHER MODEL PERFORMANCE")
    print("="*60)
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(hetero.to(device), None)
        if isinstance(teacher_logits, tuple):
            _, teacher_logits = teacher_logits
        tr_acc = accuracy(teacher_logits, y, idx_train)
        va_acc = accuracy(teacher_logits, y, idx_val)
        te_acc = accuracy(teacher_logits, y, idx_test)
        print(f"Training Accuracy:   {tr_acc:.4f}")
        print(f"Validation Accuracy: {va_acc:.4f}")
        print(f"Test Accuracy:       {te_acc:.4f}")
    print("="*60)

    # graph-free inference demo
    student.eval()
    with torch.no_grad():
        logits = student(student_feats)
        pred = logits.argmax(dim=-1)
        print(f"[Inference] demo logits shape: {logits.shape}, preds shape: {pred.shape}")

    # Inference benchmark
    teacher.eval()
    warmup = max(0, args.benchmark_warmup)
    runs = max(0, args.benchmark_runs)
    if runs > 0:
        hetero_device = hetero.to(device)
        student_input = student_feats

        def teacher_forward():
            out = teacher(hetero_device, None)
            return out if isinstance(out, torch.Tensor) else out[1]

        def student_forward():
            out = student(student_input)
            return out if isinstance(out, torch.Tensor) else out[1]

        mean_t, std_t = _benchmark_forward(teacher_forward, warmup, runs, device)
        mean_s, std_s = _benchmark_forward(student_forward, warmup, runs, device)
        print(f"[Benchmark] Teacher: {mean_t * 1000:.3f} ± {std_t * 1000:.3f} ms")
        print(f"[Benchmark] Student: {mean_s * 1000:.3f} ± {std_s * 1000:.3f} ms")
        if mean_s > 0:
            print(f"[Benchmark] Speedup (teacher/student): {mean_t / mean_s:.2f}x")


if __name__ == "__main__":
    main()


# python relation_distill_only.py -d TMDB --gpu_id 0 --use_positional_encoding