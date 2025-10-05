# -*- coding: utf-8 -*-
"""
Two-Teacher KD:
- Teacher #1: RSAGE_Hetero (relation KD + relative-position L2)  [unchanged]
- Teacher #2: FullHANTeacher (full HAN with node-level and semantic-level attention)
- Student: MLP (graph-free at inference) + train-time adapters:
    * RelKDAdapter (relation)
    * MetaPathAdapterS (metapath)
Loss = CE
     + RSAGE: KD(logits) + λ_rel_pos * relation relative-position L2 + λ_rel_struct * relation structural CE
     + HAN:   KD_mp(logits) + λ_feat_mp * feature MSE
              + λ_attn_node * node-attn KL + λ_attn_sem * semantic-attn KL
              + λ_lap * Laplacian smoothing (closed metapaths only)
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

from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import HeteroConv, SAGEConv, HANConv
from torch_geometric.nn.models import MetaPath2Vec
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected

# ---- dataset loader (reuse from your repo) ----
from dataloader import load_data
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP


def load_dataset_like_han_test(dataset_name='DBLP'):
    """加载与han_test.py相同的数据集"""
    import os.path as osp
    
    if dataset_name == 'DBLP':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
        # DBLP元路径: author -> paper -> author, author -> paper -> conference -> paper -> author
        metapaths = [[('author', 'to', 'paper'), ('paper', 'to', 'author')],
                     [('author', 'to', 'paper'), ('paper', 'to', 'conference'), 
                      ('conference', 'to', 'paper'), ('paper', 'to', 'author')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        dataset = DBLP(path, transform=transform)
        data = dataset[0]
        category = 'author'
        
        # 构建训练/验证/测试分割
        idx_train = data[category].train_mask.nonzero(as_tuple=True)[0].long()
        idx_val = data[category].val_mask.nonzero(as_tuple=True)[0].long()
        idx_test = data[category].test_mask.nonzero(as_tuple=True)[0].long()
        
        return data, (idx_train, idx_val, idx_test), category
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_forward(fn, warmup: int, runs: int, device: torch.device):
    _sync_if_cuda(device)
    for _ in range(max(0, warmup)):
        fn(); _sync_if_cuda(device)
    timings = []
    for _ in range(max(0, runs)):
        start = time.perf_counter()
        fn(); _sync_if_cuda(device)
        timings.append(time.perf_counter() - start)
    if not timings:
        return 0.0, 0.0
    t = np.array(timings, dtype=np.float64)
    return float(t.mean()), float(t.std())

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
    ps = F.log_softmax(student_logits / T, dim=-1)
    pt = F.log_softmax(teacher_logits / T, dim=-1)
    kl = torch.sum(torch.exp(pt) * (pt - ps), dim=-1) * (T * T)
    return kl.mean() if reduce else kl  # [N] if reduce=False

# =========================
# Teacher #1: RSAGE_Hetero (original)
# =========================
class RSAGE_Hetero(nn.Module):
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

        # 添加投影层：将不同节点类型的特征投影到统一维度 in_dim
        self.node_type_dims = node_type_dims
        self.projectors = nn.ModuleDict()
        if node_type_dims is not None:
            for nt, dim in node_type_dims.items():
                if dim != in_dim:
                    self.projectors[nt] = nn.Linear(dim, in_dim, bias=False)
                else:
                    self.projectors[nt] = nn.Identity()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        def conv_dict(in_channels, out_channels):
            return {etype: SAGEConv(in_channels, out_channels, aggr='mean') for etype in self.etypes}

        if self.num_layers <= 1:
            self.layers.append(HeteroConv(conv_dict(in_dim, num_classes), aggr='mean'))
        else:
            self.layers.append(HeteroConv(conv_dict(in_dim, hid_dim), aggr='mean'))
            for _ in range(self.num_layers - 2):
                self.layers.append(HeteroConv(conv_dict(hid_dim, hid_dim), aggr='mean'))
            self.layers.append(HeteroConv(conv_dict(hid_dim, num_classes), aggr='mean'))

        self.use_norm = False
        if norm_type == "batch":
            self.use_norm = True; norm_cls = nn.BatchNorm1d
        elif norm_type == "layer":
            self.use_norm = True; norm_cls = nn.LayerNorm
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

        # 应用投影层，将所有节点类型投影到统一维度
        if len(self.projectors) > 0:
            x_dict = {k: self.projectors[k](v) if k in self.projectors else v 
                     for k, v in x_dict.items()}

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
        
        # 应用投影层
        if len(self.projectors) > 0:
            x_dict = {k: self.projectors[k](v) if k in self.projectors else v 
                     for k, v in x_dict.items()}
        
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        h = x_dict
        for l, layer in enumerate(self.layers[:-1]):
            h = layer(h, edge_index_dict)
            h = {k: v.view(v.shape[0], -1) for k, v in h.items()}
            if len(self.norms) > 0 and l < len(self.norms):
                h = {k: self.norms[l](v) for k, v in h.items()}
            h = self._apply_each(h, self.activation)
            h = self._apply_each(h, self.dropout)
        h_in_dict = {k: v.view(v.shape[0], -1) for k, v in h.items()}
        last_layer: HeteroConv = self.layers[-1]
        taps: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}
        for (s, r, d), conv in last_layer.convs.items():
            x_src = h_in_dict[s]; x_dst = h_in_dict[d]
            ei = edge_index_dict[(s, r, d)]
            emb_dst = conv((x_src, x_dst), ei).view(-1, conv.out_channels)
            taps[(s, r, d)] = {'dst': emb_dst, 'src_in': h_in_dict[s]}
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
# Train-time Student Relation Adapter (original)
# =========================
class RelKDAdapter(nn.Module):
    def __init__(self, node_dims: Dict[str, int], d_rel: int):
        super().__init__()
        self.proj = nn.ModuleDict({nt: nn.Linear(node_dims[nt], d_rel, bias=False) for nt in node_dims})
        self.d_rel = d_rel

    @torch.no_grad()
    def _build_adj(self, hetero: HeteroData, et: Tuple[str, str, str]):
        s, r, d = et
        ei = hetero[et].edge_index.to(hetero[s].x.device)
        ns = hetero[s].num_nodes; nd = hetero[d].num_nodes
        deg = torch.bincount(ei[1], minlength=nd).clamp_min(1).float()
        val = torch.ones(ei.size(1), device=ei.device, dtype=torch.float32)
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
    def __init__(self, relations, category, rel_dim, num_classes):
        super().__init__()
        self.category = category
        self.rel_list = [et for et in relations if et[2] == category]
        self.rel_heads = nn.ModuleDict()
        for et in self.rel_list:
            self.rel_heads[self._rel_key(et)] = nn.Linear(rel_dim, num_classes, bias=False)
    @staticmethod
    def _rel_key(et): return f"{et[0]}__{et[1]}__{et[2]}"
    @property
    def has_relations(self): return len(self.rel_list) > 0
    def forward(self, rel_embs):
        if not self.has_relations: return None
        logits = None
        for et in self.rel_list:
            if et not in rel_embs: continue
            dst_feat = rel_embs[et]['dst']
            contrib = self.rel_heads[self._rel_key(et)](dst_feat)
            logits = contrib if logits is None else logits + contrib
        return logits

# =========================
# Reliability util (used for both teachers)
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
    logits = teacher(hetero, enhanced_features) if hasattr(teacher, '__call__') else None
    if isinstance(logits, tuple):
        logits = logits[0] if isinstance(logits[0], torch.Tensor) else logits[1]
    probs = torch.softmax(logits, dim=-1)
    # noise robustness
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
        logits_noise = logits_noise[0] if isinstance(logits_noise[0], torch.Tensor) else logits_noise[1]
    probs_noise = torch.softmax(logits_noise, dim=-1)
    ent = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
    ent = (ent / math.log(probs.size(1))).clamp(0, 1)
    conf = 1.0 - ent
    stab = torch.exp(- (probs - probs_noise).abs().sum(dim=-1) / 0.5)
    reliability = (alpha * conf + (1 - alpha) * stab).clamp(0, 1)
    return logits.detach(), probs.detach(), reliability.detach()

# =========================
# Relation Relative-Position loss (original)
# =========================
def _edge_relpos(emb_dst: torch.Tensor, emb_src_in: torch.Tensor, ei: torch.Tensor):
    src, dst = ei[0].long(), ei[1].long()
    return emb_dst[dst] - emb_src_in[src]  # [E, D]

def relation_relative_pos_l2(
    taps_teacher, rel_embs_student, hetero, category,
    reliability: Optional[torch.Tensor] = None,
    projector_t: Optional[nn.Module] = None, projector_s: Optional[nn.Module] = None,
) -> torch.Tensor:
    l2_all = []
    device = next(iter(taps_teacher.values()))['dst'].device if taps_teacher else hetero[category].x.device
    for et in hetero.edge_types:
        s, r, d = et
        if d != category:  # only align edges that end at category nodes
            continue
        if (et not in taps_teacher) or (et not in rel_embs_student):
            continue
        ei = hetero[et].edge_index.to(device)
        emb_t_dst = taps_teacher[et]['dst']; emb_t_src = taps_teacher[et]['src_in']
        emb_s_dst = rel_embs_student[et]['dst'].to(device); emb_s_src = rel_embs_student[et]['src_in'].to(device)
        if projector_t is not None:
            emb_t_dst = projector_t(emb_t_dst); emb_t_src = projector_t(emb_t_src)
        if projector_s is not None:
            emb_s_dst = projector_s(emb_s_dst); emb_s_src = projector_s(emb_s_src)
        if emb_t_dst.size(1) != emb_s_dst.size(1):
            raise ValueError(f"[RelKD] dim mismatch on {et}: T={emb_t_dst.size(1)} vs S={emb_s_dst.size(1)}")
        rel_t = _edge_relpos(emb_t_dst, emb_t_src, ei)
        rel_s = _edge_relpos(emb_s_dst, emb_s_src, ei)
        l2_e = (rel_t - rel_s).pow(2).sum(dim=-1) / rel_t.size(1)
        if reliability is not None:
            l2_e = l2_e * reliability[ei[1].long()]
        l2_all.append(l2_e.mean())
    if len(l2_all) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(l2_all).mean()

# =========================
# MetaPath2Vec Positional Encoding (optional)
# =========================
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
            "metapaths": metapaths, "emb_dim": emb_dim, "walk_length": walk_length,
            "context_size": context_size, "walks_per_node": walks_per_node, "epochs": epochs,
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
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    print(f"Training MetaPath2Vec for positional encoding with {epochs} epochs...")
    edge_index_dict = {et: hetero[et].edge_index for et in hetero.edge_types}
    per_path_dim = max(8, emb_dim // max(1, len(metapaths)))
    all_embs = []
    for mp_idx, mp in enumerate(metapaths):
        print(f"Training MetaPath2Vec for metapath {mp_idx + 1}/{len(metapaths)}: {mp}")
        mp2v = MetaPath2Vec(
            edge_index_dict=edge_index_dict, embedding_dim=per_path_dim, metapath=mp,
            walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node,
            num_negative_samples=5, sparse=True,
        ).to(device)
        opt = torch.optim.SparseAdam(mp2v.parameters(), lr=0.01)
        mp2v.train()
        for ep in range(epochs):
            total = 0.0
            for pos_rw, neg_rw in mp2v.loader(batch_size=512, shuffle=True):
                opt.zero_grad()
                loss = mp2v.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward(); opt.step(); total += float(loss)
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
# Metapath helpers
# =========================
def parse_metapath_name_sequences(hetero: HeteroData, category: str,
                                  rel_name_seqs: List[List[str]]) -> List[List[Tuple[str, str, str]]]:
    """Map relation-name-only sequences to typed (src, rel, dst) sequences by walking from category."""
    etypes = list(hetero.edge_types)
    # Build quick index: (src, rel) -> dst
    nxt = {}
    for (s, r, d) in etypes:
        nxt.setdefault((s, r), set()).add(d)
    mp_typed = []
    for names in rel_name_seqs:
        cur = category
        this = []
        ok = True
        for r in names:
            found = None
            for (s, rr, d) in etypes:
                if rr == r and s == cur:
                    found = (s, rr, d); break
            if found is None:
                ok = False; break
            this.append(found); cur = found[2]
        if ok: mp_typed.append(this)
    return mp_typed

@torch.no_grad()
def build_metapath_AA(hetero: HeteroData, mp: List[Tuple[str, str, str]], category: str, device: torch.device):
    """Build A(category,category) adjacency (SparseTensor) for a metapath by chaining."""
    # Chain SparseTensor multiplications: A_m[k] = (dst_k x src_k)
    cur_type = category
    A = None
    for (s, r, d) in mp:
        ei = hetero[(s, r, d)].edge_index.to(device)
        ns = hetero[s].num_nodes; nd = hetero[d].num_nodes
        val = torch.ones(ei.size(1), device=device, dtype=torch.float32)
        A_rel = SparseTensor(row=ei[1], col=ei[0], value=val, sparse_sizes=(nd, ns)).coalesce()
        if A is None:
            A = A_rel
        else:
            A = A_rel @ A  # chain: note shapes align: (nd x ns) @ (prev_dst x prev_src)
        cur_type = d
    # We only return when final dst == category (closed metapath). Otherwise None.
    if cur_type != category:
        return None
    return A

def to_edge_index_from_sparse(A: SparseTensor):
    A = A.coalesce()
    row, col, _ = A.coo()
    return torch.stack([col, row], dim=0)  # src=col, dst=row

# =========================
# Teacher #2: HAN using PyG HANConv (rewritten)
# =========================
class FullHANTeacher(nn.Module):
    def __init__(self, hetero: HeteroData, category: str,
                 metapaths_typed: List[List[Tuple[str, str, str]]],
                 in_dim: int, hidden: int, num_classes: int,
                 heads: int = 4, dropout: float = 0.2, device='cpu',
                 node_type_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.category = category
        self.metapaths = metapaths_typed
        self.hidden = hidden
        self.device = torch.device(device)
        
        # 添加投影层：将不同节点类型的特征投影到统一维度 in_dim
        self.node_type_dims = node_type_dims
        self.projectors = nn.ModuleDict()
        if node_type_dims is not None:
            for nt, dim in node_type_dims.items():
                if dim != in_dim:
                    self.projectors[nt] = nn.Linear(dim, in_dim, bias=False)
                else:
                    self.projectors[nt] = nn.Identity()
        
        # Prepare input dimensions for HANConv
        # HANConv expects a dictionary of node types to input dimensions
        if hetero is not None:
            in_channels = {}
            for node_type in hetero.node_types:
                # 使用统一的 in_dim（投影后的维度）
                in_channels[node_type] = in_dim
            
            # Create HANConv layer - 使用与han_test.py相同的参数
            # 对于使用AddMetaPaths的数据，HANConv会自动处理元路径
            self.han_conv = HANConv(
                in_channels=in_channels,
                out_channels=hidden,
                heads=heads,
                dropout=dropout,  # 使用han_test.py中的dropout=0.6
                metadata=hetero.metadata()  # PyG metadata for heterogeneous graphs
            )
            
            # Store metadata for forward pass
            self.metadata = hetero.metadata()
        else:
            # 如果hetero为None，延迟初始化（用于加载checkpoint）
            self.han_conv = None
            self.metadata = None
        
        # Classification layer - 使用与han_test.py相同的结构
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, hetero: HeteroData, feats_override: Optional[torch.Tensor] = None):
        # Prepare input features
        x_dict = {node_type: hetero[node_type].x for node_type in hetero.node_types}
        
        # Override features if provided
        if feats_override is not None:
            x_dict[self.category] = feats_override
        
        # 应用投影层，将所有节点类型投影到统一维度
        if len(self.projectors) > 0:
            x_dict = {k: self.projectors[k](v) if k in self.projectors else v 
                     for k, v in x_dict.items()}
        
        # Prepare edge indices
        edge_index_dict = {edge_type: hetero[edge_type].edge_index for edge_type in hetero.edge_types}
        
        # Forward pass through HANConv
        # HANConv returns a dictionary of node types to features
        out_dict = self.han_conv(x_dict, edge_index_dict)
        
        # Extract features for the target category
        h_fused = out_dict[self.category]  # [N, hidden]
        
        # Apply dropout and classification - 使用与han_test.py相同的结构
        logits = self.lin(h_fused)  # [N, num_classes]
        
        # For compatibility with the original interface, we need to return
        # additional information that the original implementation provided
        # Since PyG's HANConv doesn't expose intermediate attention weights,
        # we'll return None for those
        
        # Create dummy outputs for compatibility
        N = h_fused.size(0)
        M = len(self.metapaths) if self.metapaths else 1
        
        # Per-metapath features (simplified - just repeat the fused features)
        H = h_fused.unsqueeze(1).repeat(1, M, 1)  # [N, M, hidden]
        
        # Node attention (not available from PyG HANConv)
        node_attn = {i: None for i in range(M)}
        
        # Semantic attention (not available from PyG HANConv)
        alpha_sem = torch.ones(N, M, device=h_fused.device) / M  # [N, M] - uniform attention
        
        return logits, H, node_attn, alpha_sem
# =========================
# Student metapath adapter (features + attention proxies)
# =========================
class MetaPathAdapterS(nn.Module):
    """Student-side train-time adapter to build per-metapath embeddings (category nodes).
       Also produces edge-attention logits on AA edges (when available) by dot product."""
    def __init__(self, hetero: HeteroData, category: str,
                 node_dims: Dict[str, int], d_rel: int,
                 metapaths_typed: List[List[Tuple[str, str, str]]],
                 edge_index_mp: List[Optional[torch.Tensor]], device: torch.device):
        super().__init__()
        self.category = category
        self.metapaths = metapaths_typed
        self.edge_index_mp = edge_index_mp
        self.device = device
        self.proj = nn.ModuleDict({nt: nn.Linear(node_dims[nt], d_rel, bias=False) for nt in node_dims})
        self.sem_q = nn.Parameter(torch.Tensor(d_rel))
        nn.init.xavier_uniform_(self.sem_q.unsqueeze(0))
        self.W_sem = nn.Linear(d_rel, d_rel, bias=True)
        nn.init.xavier_uniform_(self.W_sem.weight)
        nn.init.zeros_(self.W_sem.bias)

    @torch.no_grad()
    def _agg_step(self, src_x, edge_index, num_dst):
        # mean aggregate src->dst
        row, col = edge_index  # src, dst
        msg = src_x[row]
        out = torch.zeros((num_dst, src_x.size(1)), device=src_x.device)
        out.index_add_(0, col, msg)
        deg = torch.bincount(col, minlength=num_dst).clamp_min(1).float().to(out.device)
        out = out / deg.unsqueeze(-1)
        return out

    def forward(self, hetero: HeteroData, node_overrides: Optional[Dict[str, torch.Tensor]] = None):
        x_proj = {}
        for nt in hetero.node_types:
            feats = node_overrides[nt] if node_overrides and nt in node_overrides else hetero[nt].x
            x_proj[nt] = self.proj[nt](feats.to(self.device))
        # Per-metapath category features
        N = hetero[self.category].num_nodes
        per_mp_feat = []
        for mp in self.metapaths:
            cur = self.category
            cur_feat = x_proj[cur]
            ok = True
            for (s, r, d) in mp:
                if s != cur:
                    ok = False; break
                ei = hetero[(s, r, d)].edge_index.to(self.device)
                dst_feat = self._agg_step(src_x=x_proj[s], edge_index=ei, num_dst=hetero[d].num_nodes)
                cur = d
                cur_feat = dst_feat if d == self.category else x_proj[d]  # maintain projected basis
            if not ok:
                per_mp_feat.append(x_proj[self.category])
            else:
                per_mp_feat.append(cur_feat if cur == self.category else x_proj[self.category])
        Hs = torch.stack(per_mp_feat, dim=1)  # [N, M, d_rel]

        # Semantic attention proxy from student features
        Zs = torch.tanh(self.W_sem(Hs))  # [N, M, d_rel]
        q = self.sem_q.view(1, 1, -1)
        scores = (Zs * q).sum(dim=-1)  # [N, M]
        alpha_sem_s = torch.softmax(scores, dim=1)  # [N, M]
        return Hs, alpha_sem_s

    def edge_attention_proxy(self, Hs_m: torch.Tensor, edge_index: torch.Tensor, heads: int = 1):
        """Produce per-edge attention distribution by softmax over incoming edges per dst.
           We use dot(dst, src) as attention logit. Returns (edge_index, alpha_s)"""
        if edge_index is None:
            return None
        src, dst = edge_index
        # logits per edge = <h_dst, h_src>
        logits = (Hs_m[dst] * Hs_m[src]).sum(dim=-1)  # [E]
        # softmax per dst
        with torch.no_grad():
            _, inv = torch.sort(dst)  # stable grouping (not optimal but simple)
        # compute softmax per dst via scatter
        logits_exp = torch.exp(logits - logits.max())  # numerical
        denom = torch.zeros_like(logits_exp)
        denom.index_add_(0, dst, logits_exp)
        attn = logits_exp / denom[dst].clamp_min(1e-12)
        return edge_index, attn  # [E]

# =========================
# Train/Eval for teachers
# =========================
@torch.no_grad()
def evaluate_teacher_logits(teacher_fn, hetero, category, y, idx_tr, idx_va, idx_te, device):
    out = teacher_fn(hetero.to(device), None)
    if isinstance(out, tuple):
        logits = out[0] if isinstance(out[0], torch.Tensor) else out[1]
    else:
        logits = out
    return (accuracy(logits, y, idx_tr),
            accuracy(logits, y, idx_va),
            accuracy(logits, y, idx_te))

def train_teacher_rsage(args, hetero, category, y, idx_tr, idx_va, idx_te, device, save_path=None):
    # 收集所有节点类型的特征维度
    node_type_dims = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
    # 使用统一的投影维度（选择category节点的维度或args指定的维度）
    unified_dim = hetero[category].x.size(1)
    
    teacher = RSAGE_Hetero(
        etypes=list(hetero.edge_types),
        in_dim=unified_dim,
        hid_dim=args.hid_dim,
        num_classes=int(y.max().item()) + 1,
        category=category, num_layers=args.teacher_layers,
        dropout=args.teacher_dropout, norm_type='none',
        node_type_dims=node_type_dims,
    ).to(device)

    opt = torch.optim.Adam(teacher.parameters(), lr=args.teacher_lr, weight_decay=args.teacher_wd)
    best_state, best_val, es = None, -1.0, 0
    for ep in range(args.teacher_epochs):
        teacher.train()
        logits = teacher(hetero.to(device), None)
        loss = F.cross_entropy(logits[idx_tr], y[idx_tr])
        opt.zero_grad(); loss.backward(); opt.step()
        tr, va, te = evaluate_teacher_logits(teacher, hetero, category, y, idx_tr, idx_va, idx_te, device)
        print(f"[RSAGE] ep {ep:03d} | loss {loss.item():.4f} | tr {tr:.4f} va {va:.4f} te {te:.4f}")
        if va >= best_val:
            best_val, best_state, es = va, copy.deepcopy(teacher.state_dict()), 0
        else:
            es += 1
            if es >= args.teacher_patience:
                print("[RSAGE] early stop"); break
    if best_state is not None: teacher.load_state_dict(best_state)
    tr, va, te = evaluate_teacher_logits(teacher, hetero, category, y, idx_tr, idx_va, idx_te, device)
    print(f"[RSAGE] Final(best) | tr {tr:.4f} va {va:.4f} te {te:.4f}")
    
    # Save teacher if path provided
    if save_path:
        torch.save({
            'model_state_dict': teacher.state_dict(),
            'args': args,
            'category': category,
            'etypes': list(hetero.edge_types),
            'in_dim': unified_dim,
            'node_type_dims': node_type_dims,
            'num_classes': int(y.max().item()) + 1,
            'performance': {'train': tr, 'val': va, 'test': te}
        }, save_path)
        print(f"[RSAGE] Model saved to {save_path}")
    
    return teacher


def train_teacher_han(args, hetero, category, y, idx_tr, idx_va, idx_te,
                      metapaths_typed, device, save_path=None):
    # 对于使用AddMetaPaths的数据，不需要额外的元路径处理
    if metapaths_typed is None or len(metapaths_typed) == 0:
        # 创建一个空的元路径列表，因为AddMetaPaths已经处理了元路径
        metapaths_typed = []
    # 收集所有节点类型的特征维度
    node_type_dims = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
    # 使用统一的投影维度
    unified_dim = hetero[category].x.size(1)
    C = int(y.max().item()) + 1
    teacher = FullHANTeacher(
        hetero=hetero.to(device), category=category, metapaths_typed=metapaths_typed,
        in_dim=unified_dim, hidden=args.han_dim, num_classes=C,
        heads=args.han_heads, dropout=args.han_dropout, device=str(device),
        node_type_dims=node_type_dims,
    ).to(device)
    # 使用与han_test.py相同的优化器：Adam而不是AdamW
    opt = torch.optim.Adam(teacher.parameters(), lr=args.han_lr, weight_decay=args.han_wd)
    best_state, best_val, es = None, -1.0, 0
    for ep in range(args.han_epochs):
        teacher.train()
        logits, _, _, _ = teacher(hetero.to(device), None)
        loss = F.cross_entropy(logits[idx_tr], y[idx_tr])
        opt.zero_grad(); loss.backward()
        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(teacher.parameters(), args.grad_clip_norm)
        opt.step()
        with torch.no_grad():
            logits, _, _, _ = teacher(hetero.to(device), None)
            tr = accuracy(logits, y, idx_tr); va = accuracy(logits, y, idx_va); te = accuracy(logits, y, idx_te)
        print(f"[HAN] ep {ep:03d} | loss {loss.item():.4f} | tr {tr:.4f} va {va:.4f} te {te:.4f}")
        if va >= best_val:
            best_val, best_state, es = va, copy.deepcopy(teacher.state_dict()), 0
        else:
            es += 1
            if es >= args.han_patience:
                print("[HAN] early stop"); break
    if best_state is not None: teacher.load_state_dict(best_state)
    with torch.no_grad():
        logits, H, node_attn, alpha_sem = teacher(hetero.to(device), None)
        tr = accuracy(logits, y, idx_tr); va = accuracy(logits, y, idx_va); te = accuracy(logits, y, idx_te)
    print(f"[HAN] Final(best) | tr {tr:.4f} va {va:.4f} te {te:.4f}")
    
    # Save teacher if path provided
    if save_path:
        torch.save({
            'model_state_dict': teacher.state_dict(),
            'args': args,
            'category': category,
            'metapaths_typed': metapaths_typed,
            'in_dim': unified_dim,
            'node_type_dims': node_type_dims,
            'num_classes': C,
            'performance': {'train': tr, 'val': va, 'test': te}
        }, save_path)
        print(f"[HAN] Model saved to {save_path}")
    
    return teacher


def load_teacher_rsage(checkpoint_path, device):
    """Load RSAGE teacher from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 兼容旧版本checkpoint（没有node_type_dims字段）
    node_type_dims = checkpoint.get('node_type_dims', None)
    teacher = RSAGE_Hetero(
        etypes=checkpoint['etypes'],
        in_dim=checkpoint['in_dim'],
        hid_dim=checkpoint['args'].hid_dim,
        num_classes=checkpoint['num_classes'],
        category=checkpoint['category'],
        num_layers=checkpoint['args'].teacher_layers,
        dropout=checkpoint['args'].teacher_dropout,
        norm_type='none',
        node_type_dims=node_type_dims,
    ).to(device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    print(f"[RSAGE] Loaded from {checkpoint_path}")
    print(f"[RSAGE] Performance: Train {checkpoint['performance']['train']:.4f}, "
          f"Val {checkpoint['performance']['val']:.4f}, Test {checkpoint['performance']['test']:.4f}")
    return teacher

def load_teacher_han(checkpoint_path, device, hetero=None):
    """Load HAN teacher from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 兼容旧版本checkpoint（没有node_type_dims字段）
    node_type_dims = checkpoint.get('node_type_dims', None)
    teacher = FullHANTeacher(
        hetero=hetero.to(device) if hetero is not None else None,  # 需要hetero来获取metadata
        category=checkpoint['category'],
        metapaths_typed=checkpoint['metapaths_typed'],
        in_dim=checkpoint['in_dim'],
        hidden=checkpoint['args'].han_dim,
        num_classes=checkpoint['num_classes'],
        heads=checkpoint['args'].han_heads,
        dropout=checkpoint['args'].han_dropout,
        device=str(device),
        node_type_dims=node_type_dims,
    ).to(device)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    print(f"[HAN] Loaded from {checkpoint_path}")
    print(f"[HAN] Performance: Train {checkpoint['performance']['train']:.4f}, "
          f"Val {checkpoint['performance']['val']:.4f}, Test {checkpoint['performance']['test']:.4f}")
    return teacher

# =========================
# Training: student with dual-teacher KD
# =========================
def train_student_dual_kd(args, hetero, category, y, idx_tr, idx_va, idx_te, device, metapaths_relnames=None, 
                          t_rel=None, t_han=None):
    # 0) Parse metapaths
    rel_sequences = args.metapaths if args.metapaths else (metapaths_relnames or [])
    
    # 检查metapaths_relnames是否已经是完整的元路径元组（来自dataloader.py）
    if metapaths_relnames and len(metapaths_relnames) > 0 and isinstance(metapaths_relnames[0][0], tuple):
        # 如果已经是元组格式，直接使用
        mp_typed = metapaths_relnames
    else:
        # 否则使用原来的解析函数
        mp_typed = parse_metapath_name_sequences(hetero, category, rel_sequences)

    # 1) Train or load RSAGE teacher
    if t_rel is None:
        t_rel = train_teacher_rsage(args, hetero, category, y, idx_tr, idx_va, idx_te, device)
    else:
        print("[RSAGE] Using provided teacher")

    # 2) Train or load HAN teacher (optional)
    if t_han is None and args.use_han_teacher and len(mp_typed) > 0:
        t_han = train_teacher_han(args, hetero, category, y, idx_tr, idx_va, idx_te, mp_typed, device)
    elif t_han is not None:
        print("[HAN] Using provided teacher")

    # 3) Precompute RSAGE taps & reliability
    hetero_device = hetero.to(device)
    with torch.no_grad():
        t_logits_full, taps_rel = t_rel.forward_with_relation_taps(hetero_device, None)
        _, _, reliability_rel = run_teacher_and_reliability(t_rel, hetero_device, category, device, None)
        reliability_rel = reliability_rel.detach()
    t_logits_full = t_logits_full.detach()
    taps_rel = {et: {k: v.detach() for k, v in tap.items()} for et, tap in taps_rel.items()}

    # 4) If HAN: forward once to cache taps & reliability
    if t_han is not None:
        t_han.eval()
        with torch.no_grad():
            logits_han, H_mp, node_attn, alpha_sem = t_han(hetero_device, None)
            _, _, reliability_mp = run_teacher_and_reliability(t_han, hetero_device, category, device, None)
            reliability_mp = reliability_mp.detach()
    else:
        logits_han = None; H_mp = None; node_attn = None; alpha_sem = None; reliability_mp = None

    # 5) Student feature base (+ optional MetaPath2Vec)
    base_feat = hetero_device[category].x
    student_feats = base_feat
    student_node_overrides: Dict[str, torch.Tensor] = {}

    # For PE we can only use closed metapaths (AA graph exists)
    closed_mp = []
    if len(mp_typed) > 0:
        # identify closed
        for mp in mp_typed:
            if mp[-1][2] == category:
                closed_mp.append(mp)
    if args.use_positional_encoding and len(closed_mp) > 0:
        pe = metapath2vec_category_embeddings(
            hetero_device, closed_mp, category,
            emb_dim=args.mp_pe_dim, walk_length=args.mp_walk_length,
            context_size=args.mp_context_size, walks_per_node=args.mp_walks_per_node,
            epochs=args.mp_epochs, device=str(device), cache_dir=args.mp_cache_dir, seed=args.seed
        ).to(device)
        student_feats = torch.cat([student_feats, pe], dim=1)
        print(f"[PE] MetaPath2Vec embeddings shape: {pe.shape}")
    student_node_overrides[category] = student_feats

    # 6) Student main MLP
    x = student_feats
    C = int(y.max().item()) + 1
    student = MLPClassifier(
        d_in=x.size(1), n_classes=C, hidden=args.student_hidden,
        num_layers=args.student_layers, dropout=args.student_dropout
    ).to(device)

    # 7) Relation adapter & head
    node_dims = {nt: hetero_device[nt].x.size(1) for nt in hetero.node_types}
    for nt, feats in student_node_overrides.items():
        node_dims[nt] = feats.size(1)
    rel_adapter = RelKDAdapter(node_dims, d_rel=args.rel_dim).to(device)
    rel_struct_head = RelationStructuralHead(list(hetero.edge_types), category, args.rel_dim, C).to(device)

    # 8) HAN student adapter & per-metapath head
    if t_han is not None:
        # reuse HAN teacher metapath AA edge_index(s)
        edge_index_mp = t_han.edge_index_mp
        mp_adapter = MetaPathAdapterS(hetero_device, category, node_dims, args.rel_dim, mp_typed, edge_index_mp, device).to(device)
        mp_heads = nn.ModuleList([nn.Linear(args.rel_dim, C, bias=False) for _ in range(len(mp_typed))]).to(device)
    else:
        mp_adapter = None
        mp_heads = None

    # 9) Projectors
    # RSAGE taps projector -> rel_dim
    if len(taps_rel) > 0:
        first_et = next(iter(taps_rel))
        teach_dst_dim = taps_rel[first_et]['dst'].size(1)
        teach_src_dim = taps_rel[first_et]['src_in'].size(1)
        class TProjector(nn.Module):
            def __init__(self, dst_dim, src_dim, target_dim):
                super().__init__()
                self.dst_proj = nn.Linear(dst_dim, target_dim, bias=False) if dst_dim != target_dim else nn.Identity()
                self.src_proj = nn.Linear(src_dim, target_dim, bias=False) if src_dim != target_dim else nn.Identity()
            def forward_dst(self, x): return self.dst_proj(x)
            def forward_src(self, x): return self.src_proj(x)
        t_proj = TProjector(teach_dst_dim, teach_src_dim, args.rel_dim).to(device)
    else:
        t_proj = None

    # HAN feature projectors: han_dim -> rel_dim per metapath
    proj_T_mp = None
    if t_han is not None:
        proj_T_mp = nn.ModuleList([nn.Linear(args.han_dim, args.rel_dim, bias=False) for _ in range(len(mp_typed))]).to(device)

    # 10) Optimizer
    params = [{'params': student.parameters()},
              {'params': rel_adapter.parameters()}]
    if rel_struct_head.has_relations:
        params.append({'params': rel_struct_head.parameters(),
                       'lr': args.struct_head_lr, 'weight_decay': args.struct_head_wd})
    if t_proj is not None:
        params.append({'params': t_proj.parameters()})
    if mp_adapter is not None:
        params.append({'params': mp_adapter.parameters()})
        params.append({'params': mp_heads.parameters(),
                       'lr': args.struct_head_lr, 'weight_decay': args.struct_head_wd})
        if proj_T_mp is not None:
            params.append({'params': proj_T_mp.parameters()})
    opt = torch.optim.AdamW(params, lr=args.student_lr, weight_decay=args.student_wd)

    # 11) Train loop
    best_va, best_state, best_te, es = -1.0, None, 0.0, 0
    for ep in range(args.student_epochs):
        student.train()
        # forward main
        s_logits = student(x)

        # CE
        ce = F.cross_entropy(s_logits[idx_tr], y[idx_tr])

        # RSAGE KD
        kl_vec = kd_kl(s_logits, t_logits_full, T=args.kd_T, reduce=False)  # [N]
        kd_rel = (kl_vec * reliability_rel).mean()

        # Relation Relative-Position
        rel_embs_student = rel_adapter(hetero_device, list(hetero.edge_types), node_overrides=student_node_overrides)
        structural_logits_rel = rel_struct_head(rel_embs_student) if rel_struct_head.has_relations else None

        def _pt(x, kind="dst"):
            if t_proj is None: return x
            return t_proj.forward_dst(x) if kind == "dst" else t_proj.forward_src(x)

        taps_proj = {}
        for et, mm in taps_rel.items():
            taps_proj[et] = {'dst': _pt(mm['dst'], "dst"), 'src_in': _pt(mm['src_in'], "src")}

        rel_l2 = relation_relative_pos_l2(
            taps_teacher=taps_proj, rel_embs_student=rel_embs_student,
            hetero=hetero_device, category=category,
            reliability=reliability_rel, projector_t=None, projector_s=None
        )
        struct_ce_rel = torch.tensor(0.0, device=device)
        if structural_logits_rel is not None:
            struct_ce_rel = F.cross_entropy(structural_logits_rel[idx_tr], y[idx_tr])

        # HAN KD pieces (if enabled)
        kd_mp = torch.tensor(0.0, device=device)
        feat_mse_mp = torch.tensor(0.0, device=device)
        attn_node_kl = torch.tensor(0.0, device=device)
        attn_sem_kl = torch.tensor(0.0, device=device)
        lap = torch.tensor(0.0, device=device)

        if t_han is not None:
            # logits KD from HAN
            if logits_han is None:
                with torch.no_grad():
                    logits_han_, _, _, _ = t_han(hetero_device, None)
            else:
                logits_han_ = logits_han
            kl_mp_vec = kd_kl(s_logits, logits_han_.detach(), T=args.kd_T, reduce=False)
            if args.han_reliability_to_kd and reliability_mp is not None:
                kd_mp = (kl_mp_vec * reliability_mp).mean()
            else:
                kd_mp = kl_mp_vec.mean()

            # Student metapath features & semantic attention
            Hs, alpha_sem_s = mp_adapter(hetero_device, node_overrides=student_node_overrides)  # [N,M,D], [N,M]

            # Feature MSE per metapath on category nodes
            with torch.no_grad():
                _, H_mp_t, node_attn_t, alpha_sem_t = t_han(hetero_device, None)
            # project teacher features to rel_dim
            per_mp_mse = []
            for mi in range(len(mp_typed)):
                Ht = H_mp_t[:, mi, :]  # [N, han_dim]
                Ht_proj = proj_T_mp[mi](Ht.detach())
                diff = (Hs[:, mi, :] - Ht_proj)
                mse = (diff * diff).mean(dim=-1)  # [N]
                weight = 1.0
                if reliability_mp is not None and args.han_reliability_to_feat:
                    weight = weight * reliability_mp
                # optionally emphasize nodes by semantic attention
                if args.mp_attn_gamma > 0:
                    weight = weight * (alpha_sem_t[:, mi].detach() ** args.mp_attn_gamma)
                per_mp_mse.append((mse * weight).mean())
            feat_mse_mp = torch.stack(per_mp_mse).mean()

            # Node-level attention KL (per metapath, only when AA edges exist)
            per_mp_node_kl = []
            for mi in range(len(mp_typed)):
                if node_attn_t[mi] is None or t_han.edge_index_mp[mi] is None:
                    continue
                eix_t, alpha_t = node_attn_t[mi]  # alpha over edges*heads
                # average over heads to [E]
                # alpha_t shape: [E, heads] or [E*heads]? GATConv returns [num_edges, heads]
                if alpha_t.dim() == 2:
                    alpha_t_e = alpha_t.mean(dim=1)
                else:
                    alpha_t_e = alpha_t  # [E]
                # student edge attention proxy on same edges using Hs[:, mi, :]
                Hm = Hs[:, mi, :]  # [N, D]
                eix = eix_t  # [2, E]
                _, alpha_s = mp_adapter.edge_attention_proxy(Hm, eix)
                # KL over edges, optionally weight by dst degree / semantic attn
                # safeguard shapes
                minE = min(alpha_t_e.numel(), alpha_s.numel())
                pt = (alpha_t_e[:minE] + 1e-12); ps = (alpha_s[:minE] + 1e-12)
                kl = (pt * (pt.log() - ps.log())).mean()
                if args.han_reliability_to_struct and reliability_mp is not None:
                    kl = kl * reliability_mp.mean()
                per_mp_node_kl.append(kl)
            if len(per_mp_node_kl) > 0:
                attn_node_kl = torch.stack(per_mp_node_kl).mean()

            # Semantic attention KL per node across metapaths
            pt = (alpha_sem_t + 1e-12); ps = (alpha_sem_s + 1e-12)
            sem_kl_vec = (pt * (pt.log() - ps.log())).sum(dim=1)  # [N]
            if args.han_reliability_to_struct and reliability_mp is not None:
                attn_sem_kl = (sem_kl_vec * reliability_mp).mean()
            else:
                attn_sem_kl = sem_kl_vec.mean()

            # Laplacian smoothing on closed metapath AA graphs (student predictions)
            if args.lambda_lap > 0:
                probs_s = torch.softmax(s_logits, dim=-1).detach()  # stop-grad inside laplacian? keep as regularization
                per_mp_lap = []
                for mi in range(len(mp_typed)):
                    ei = t_han.edge_index_mp[mi]
                    if ei is None: continue
                    # simple Dirichlet: sum ||p_u - p_v||^2 over edges (u->v), averaged
                    pu = probs_s[ei[0]]; pv = probs_s[ei[1]]
                    diff2 = (pu - pv).pow(2).sum(dim=-1)
                    w = alpha_sem_t[:, mi].mean()  # scalar weight: avg semantic importance of this mp
                    per_mp_lap.append(diff2.mean() * w)
                if len(per_mp_lap) > 0:
                    lap = torch.stack(per_mp_lap).mean()

        # Total loss
        loss = (args.ce_coeff * ce
                + args.kd_coeff * kd_rel
                + args.lambda_rel_pos * rel_l2
                + args.lambda_rel_struct * struct_ce_rel
                + (args.kd_coeff_mp * kd_mp if t_han is not None else 0)
                + (args.lambda_feat_mp * feat_mse_mp if t_han is not None else 0)
                + (args.lambda_attn_node * attn_node_kl if t_han is not None else 0)
                + (args.lambda_attn_sem * attn_sem_kl if t_han is not None else 0)
                + (args.lambda_lap * lap if t_han is not None else 0)
                )

        opt.zero_grad()
        loss.backward()
        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip_norm)
        opt.step()

        # Eval
        student.eval()
        with torch.no_grad():
            logits_eval = student(x)
            tr = accuracy(logits_eval, y, idx_tr)
            va = accuracy(logits_eval, y, idx_va)
            te = accuracy(logits_eval, y, idx_te)
        print(f"[Student] ep {ep:03d} | "
              f"CE {ce:.4f} KDrel {kd_rel:.4f} REL {rel_l2:.4f} RStruct {struct_ce_rel:.4f} "
              f"| KDmp {kd_mp:.4f} FMSE {feat_mse_mp:.4f} ANode {attn_node_kl:.4f} ASem {attn_sem_kl:.4f} LAP {lap:.4f} "
              f"| tr {tr:.4f} va {va:.4f} te {te:.4f}")

        if va >= best_va:
            best_va, best_state, best_te, es = va, {
                'student': copy.deepcopy(student.state_dict()),
                'rel_adapter': copy.deepcopy(rel_adapter.state_dict()),
                'rel_head': copy.deepcopy(rel_struct_head.state_dict()),
                't_proj': copy.deepcopy(t_proj.state_dict()) if t_proj is not None else None,
                'mp_adapter': copy.deepcopy(mp_adapter.state_dict()) if mp_adapter is not None else None,
                'mp_heads': copy.deepcopy([h.state_dict() for h in mp_heads]) if mp_heads is not None else None,
            }, te, 0
        else:
            es += 1
            if es >= args.student_patience:
                print("[Student] early stop"); break

    # Restore best
    if best_state is not None:
        student.load_state_dict(best_state['student'])
        rel_adapter.load_state_dict(best_state['rel_adapter'])
        rel_struct_head.load_state_dict(best_state['rel_head'])
        if t_proj is not None and best_state['t_proj'] is not None:
            t_proj.load_state_dict(best_state['t_proj'])
        if mp_adapter is not None and best_state['mp_adapter'] is not None:
            mp_adapter.load_state_dict(best_state['mp_adapter'])
        if mp_heads is not None and best_state['mp_heads'] is not None:
            for h, sd in zip(mp_heads, best_state['mp_heads']):
                h.load_state_dict(sd)
        with torch.no_grad():
            logits_eval = student(x)
            tr = accuracy(logits_eval, y, idx_tr)
            va = accuracy(logits_eval, y, idx_va)
            te = accuracy(logits_eval, y, idx_te)
        print(f"[Student] Final(best) | tr {tr:.4f} va {va:.4f} te {te:.4f} (best_te {best_te:.4f})")

    return t_rel, t_han, student, student_feats

# =========================
# Main
# =========================
def main():
    p = argparse.ArgumentParser("Dual-Teacher KD: RSAGE + Full HAN (attn & semantics) for metapath distillation")
    p.add_argument("-d", "--dataset", type=str, default="TMDB", choices=["TMDB", "CroVal", "ArXiv", "IGB-tiny-549K-19", "IGB-small-549K-2983", "DBLP"])
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_clip_norm", type=float, default=0.0)
    
    # Training mode selection
    p.add_argument("--train_mode", type=str, default="all", 
                   choices=["all", "rsage_only", "han_only", "student_only"],
                   help="Training mode: all (default), rsage_only, han_only, student_only")
    p.add_argument("--teacher_checkpoint", type=str, default=None,
                   help="Path to saved teacher checkpoint for student_only mode")

    # RSAGE teacher
    p.add_argument("--hid_dim", type=int, default=128)
    p.add_argument("--teacher_layers", type=int, default=2)
    p.add_argument("--teacher_dropout", type=float, default=0.2)
    p.add_argument("--teacher_lr", type=float, default=1e-2)
    p.add_argument("--teacher_wd", type=float, default=0.0)
    p.add_argument("--teacher_epochs", type=int, default=100)
    p.add_argument("--teacher_patience", type=int, default=20)

    # HAN teacher
    p.add_argument("--use_han_teacher", action="store_true")
    p.add_argument("--han_dim", type=int, default=128)
    p.add_argument("--han_heads", type=int, default=8)
    p.add_argument("--han_dropout", type=float, default=0.6)
    p.add_argument("--han_lr", type=float, default=0.005)
    p.add_argument("--han_wd", type=float, default=0.001)
    p.add_argument("--han_epochs", type=int, default=200)
    p.add_argument("--han_patience", type=int, default=40)
    p.add_argument("--han_reliability_to_kd", action="store_true")
    p.add_argument("--han_reliability_to_feat", action="store_true")
    p.add_argument("--han_reliability_to_struct", action="store_true")

    # Student
    p.add_argument("--student_hidden", type=int, default=128)
    p.add_argument("--student_layers", type=int, default=2)
    p.add_argument("--student_dropout", type=float, default=0.5)
    p.add_argument("--student_lr", type=float, default=2e-3)
    p.add_argument("--student_wd", type=float, default=5e-4)
    p.add_argument("--student_epochs", type=int, default=1000)
    p.add_argument("--student_patience", type=int, default=500)

    # KD & relation loss
    p.add_argument("--kd_T", type=float, default=1.0)
    p.add_argument("--ce_coeff", type=float, default=1.0)
    p.add_argument("--kd_coeff", type=float, default=1.0)                 # RSAGE KD
    p.add_argument("--lambda_rel_pos", type=float, default=1.0)           # RSAGE relative-position
    p.add_argument("--rel_dim", type=int, default=256)
    p.add_argument("--lambda_rel_struct", type=float, default=1.0)
    p.add_argument("--struct_head_lr", type=float, default=2e-3)
    p.add_argument("--struct_head_wd", type=float, default=1e-4)

    # Metapath KD weights (HAN)
    p.add_argument("--kd_coeff_mp", type=float, default=1.0)              # HAN logits KD
    p.add_argument("--lambda_feat_mp", type=float, default=1.0)           # feature MSE
    p.add_argument("--lambda_attn_node", type=float, default=1.0)         # node-level attention KL
    p.add_argument("--lambda_attn_sem", type=float, default=1.0)          # semantic-level attention KL
    p.add_argument("--lambda_lap", type=float, default=1.0)               # Laplacian smoothing
    p.add_argument("--mp_attn_gamma", type=float, default=1.0)            # weight by semantic attention^gamma

    # Positional encoding (optional)
    p.add_argument("--use_positional_encoding", action="store_true")
    p.add_argument("--positional_relations", type=str, nargs='*', default=[],
                   help="metapaths as comma-separated relation names; e.g. directed_by,directs performed_by,performs")
    p.add_argument("--mp_pe_dim", type=int, default=128)
    p.add_argument("--mp_walk_length", type=int, default=40)
    p.add_argument("--mp_context_size", type=int, default=5)
    p.add_argument("--mp_walks_per_node", type=int, default=10)
    p.add_argument("--mp_epochs", type=int, default=50)
    p.add_argument("--mp_cache_dir", type=str, default="./mp2v_cache")

    # Benchmark
    p.add_argument("--benchmark_warmup", type=int, default=3)
    p.add_argument("--benchmark_runs", type=int, default=10)

    args = p.parse_args()
    args.metapaths = []
    rel_args = getattr(args, 'positional_relations', []) or []
    for item in rel_args:
        seq = [tok.strip() for tok in item.split(',') if tok.strip()]
        if seq: args.metapaths.append(seq)

    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # load data
    if args.dataset == 'DBLP':
        # 对于DBLP，使用与han_test.py相同的数据加载方式
        hetero, splits, category = load_dataset_like_han_test('DBLP')
        idx_train, idx_val, idx_test = [t.to(device) for t in splits]
        y = hetero[category].y.long().to(device)
        metapaths_relnames = None  # 不需要元路径，因为已经通过AddMetaPaths处理
    else:
        # 其他数据集使用原来的方式
        hetero, splits, gen_node_feats, metapaths_relnames = load_data(dataset=args.dataset, return_mp=True)
        hetero = gen_node_feats(hetero)
        category = hetero.category
        idx_train, idx_val, idx_test = [t.to(device) for t in splits]
        y = hetero[category].y.long().to(device)

    if args.use_positional_encoding and not args.metapaths and metapaths_relnames:
        args.metapaths = metapaths_relnames

    # train pipeline based on mode
    if args.train_mode == "rsage_only":
        print("Training RSAGE teacher only...")
        save_path = f"rsage_teacher_{args.dataset}_{args.seed}.pt"
        t_rel = train_teacher_rsage(args, hetero, category, y, idx_train, idx_val, idx_test, device, save_path)
        t_han, student, student_feats = None, None, None
        
    elif args.train_mode == "han_only":
        print("Training HAN teacher only...")
        
        if args.dataset == 'DBLP':
            # 对于DBLP，使用AddMetaPaths处理的数据，不需要额外的元路径
            mp_typed = None
        else:
            # 其他数据集使用原来的元路径处理
            rel_sequences = args.metapaths if args.metapaths else (metapaths_relnames or [])
            
            # 检查metapaths_relnames是否已经是完整的元路径元组（来自dataloader.py）
            if metapaths_relnames and len(metapaths_relnames) > 0 and isinstance(metapaths_relnames[0][0], tuple):
                # 如果已经是元组格式，直接使用
                mp_typed = metapaths_relnames
            else:
                # 否则使用原来的解析函数
                mp_typed = parse_metapath_name_sequences(hetero, category, rel_sequences)
            
            if len(mp_typed) == 0:
                print("No valid metapaths found for HAN training!")
                return
        
        save_path = f"han_teacher_{args.dataset}_{args.seed}.pt"
        t_han = train_teacher_han(args, hetero, category, y, idx_train, idx_val, idx_test, mp_typed, device, save_path)
        t_rel, student, student_feats = None, None, None
        
    elif args.train_mode == "student_only":
        print("Training student only with pre-trained teachers...")
        if args.teacher_checkpoint is None:
            print("Error: teacher_checkpoint must be provided for student_only mode!")
            return
        
        # Load teacher(s) from checkpoint
        checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
        if 'etypes' in checkpoint:  # RSAGE checkpoint
            t_rel = load_teacher_rsage(args.teacher_checkpoint, device)
            t_han = None
        elif 'metapaths_typed' in checkpoint:  # HAN checkpoint
            t_rel = None
            t_han = load_teacher_han(args.teacher_checkpoint, device, hetero=hetero)
        else:
            print("Error: Unknown checkpoint format!")
            return
            
        t_rel, t_han, student, student_feats = train_student_dual_kd(
            args, hetero, category, y, idx_train, idx_val, idx_test, device, metapaths_relnames, t_rel, t_han
        )
        
    else:  # train_mode == "all"
        print("Training all models (RSAGE + HAN + Student)...")
        t_rel, t_han, student, student_feats = train_student_dual_kd(
            args, hetero, category, y, idx_train, idx_val, idx_test, device, metapaths_relnames
        )

    # Print teacher performance
    print("\n" + "="*60)
    print("TEACHER MODEL PERFORMANCE")
    print("="*60)
    if t_rel is not None:
        t_rel.eval()
        with torch.no_grad():
            logits_rel = t_rel(hetero.to(device), None)
            tr_acc = accuracy(logits_rel, y, idx_train)
            va_acc = accuracy(logits_rel, y, idx_val)
            te_acc = accuracy(logits_rel, y, idx_test)
            print(f"[RSAGE] Train {tr_acc:.4f} | Val {va_acc:.4f} | Test {te_acc:.4f}")
    if t_han is not None:
        t_han.eval()
        with torch.no_grad():
            logits_han, _, _, _ = t_han(hetero.to(device), None)
            tr_acc = accuracy(logits_han, y, idx_train)
            va_acc = accuracy(logits_han, y, idx_val)
            te_acc = accuracy(logits_han, y, idx_test)
            print(f"[HAN]   Train {tr_acc:.4f} | Val {va_acc:.4f} | Test {te_acc:.4f}")

    # Graph-free inference demo (only if student exists)
    if student is not None:
        student.eval()
        with torch.no_grad():
            logits = student(student_feats)
            pred = logits.argmax(dim=-1)
            print(f"[Inference] demo logits shape: {logits.shape}, preds shape: {pred.shape}")

        # Inference benchmark
        warmup = max(0, args.benchmark_warmup); runs = max(0, args.benchmark_runs)
        if runs > 0:
            hetero_device = hetero.to(device)
            student_input = student_feats
            def teacher_forward():
                out = t_rel(hetero_device, None)
                return out if isinstance(out, torch.Tensor) else out[1]
            def student_forward():
                out = student(student_input)
                return out if isinstance(out, torch.Tensor) else out[1]
            mean_t, std_t = _benchmark_forward(teacher_forward, warmup, runs, device)
            mean_s, std_s = _benchmark_forward(student_forward, warmup, runs, device)
            print(f"[Benchmark] Teacher(RSAGE): {mean_t * 1000:.3f} ± {std_t * 1000:.3f} ms")
            print(f"[Benchmark] Student:        {mean_s * 1000:.3f} ± {std_s * 1000:.3f} ms")
            if mean_s > 0:
                print(f"[Benchmark] Speedup (teacher/student): {mean_t / mean_s:.2f}x")


if __name__ == "__main__":
    main()

#python two_teacher_kd.py -d TMDB --gpu_id 0 --seed 0 --use_han_teacher --positional_relations directed_by,directs performed_by,performs --han_dim 128 --han_heads 4 --han_dropout 0.2 --han_lr 0.002 --han_wd 0.0005 --han_epochs 200 --han_patience 40 --han_reliability_to_kd --han_reliability_to_feat --han_reliability_to_struct --student_hidden 128 --student_layers 2 --student_dropout 0.5 --student_lr 0.002 --student_wd 0.0005 --student_epochs 1000 --student_patience 500 --kd_T 2.0 --ce_coeff 1 --kd_coeff 1 --lambda_rel_pos 1 --rel_dim 256 --lambda_rel_struct 1 --struct_head_lr 0.002 --struct_head_wd 0.0001 --kd_coeff_mp 1 --lambda_feat_mp 0.5 --lambda_attn_node 0.5 --lambda_attn_sem 0.5 --lambda_lap 0.2 --mp_attn_gamma 1.0 --use_positional_encoding --mp_pe_dim 128 --mp_walk_length 40 --mp_context_size 5 --mp_walks_per_node 10 --mp_epochs 50 --mp_cache_dir ./mp2v_cache
