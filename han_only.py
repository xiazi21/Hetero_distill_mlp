# -*- coding: utf-8 -*-
"""
纯 HAN 训练/评测脚本（无 RSAGE、无学生、无蒸馏）
支持数据集：TMDB / ArXiv / DBLP

用法示例：
  - TMDB：
    python han_only.py -d TMDB --gpu_id 0 --epochs 300 --patience 80 --hidden 128 --heads 8 --layers 2 --dropout 0.6 --lr 0.002 --wd 0.001 --positional_relations directed_by,directs performed_by,performs
  - ArXiv：
    python han_only.py -d ArXiv --gpu_id 0 --epochs 300 --patience 80 --hidden 128 --heads 8 --layers 2 --dropout 0.6 --lr 0.002 --wd 0.001 --positional_relations cites written_by,writes
  - DBLP（内置 APA/APCPA 元路径）：
    python han_only.py -d DBLP --gpu_id 0 --epochs 300 --patience 80 --hidden 128 --heads 8 --layers 2 --dropout 0.6 --lr 0.002 --wd 0.001
"""

import os
import os.path as osp
import argparse
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import HANConv
from torch_geometric.datasets import DBLP as PYG_DBLP

from dataloader import load_data


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Seed set to {seed} (deterministic)")


def accuracy(logits: torch.Tensor, y: torch.Tensor, idx: torch.Tensor) -> float:
    pred = logits[idx].argmax(dim=-1)
    return (pred == y[idx]).float().mean().item()


def parse_metapath_name_sequences(hetero: HeteroData, category: str,
                                  rel_name_seqs: List[List[str]]) -> List[List[Tuple[str, str, str]]]:
    """将仅关系名的序列解析为带类型的 (src, rel, dst) 序列，从 category 出发逐步匹配。"""
    etypes = list(hetero.edge_types)
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
                    found = (s, rr, d)
                    break
            if found is None:
                ok = False
                break
            this.append(found)
            cur = found[2]
        if ok:
            mp_typed.append(this)
    return mp_typed


class HANStack(nn.Module):
    """多层 HANConv + 最终线性分类。包含可选的按类型特征对齐投影层。"""
    def __init__(self, hetero: HeteroData, category: str,
                 in_dim: int, hidden: int, num_classes: int,
                 heads: int = 8, dropout: float = 0.6, num_layers: int = 2,
                 node_type_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.category = category
        self.hidden = int(hidden)
        self.dropout_val = float(dropout)
        self.num_layers = int(max(1, num_layers))

        # 将不同节点类型特征统一到 in_dim
        self.projectors = nn.ModuleDict()
        if node_type_dims is not None:
            for nt, dim in node_type_dims.items():
                if dim != in_dim:
                    self.projectors[nt] = nn.Linear(dim, in_dim, bias=False)
                else:
                    self.projectors[nt] = nn.Identity()

        metadata = hetero.metadata()
        self.han_convs = nn.ModuleList()
        # 第一层：in_dim -> hidden
        self.han_convs.append(HANConv(
            in_channels={nt: in_dim for nt in hetero.node_types},
            out_channels=hidden,
            heads=heads,
            dropout=self.dropout_val,
            metadata=metadata,
        ))
        # 其余层：hidden -> hidden
        for _ in range(self.num_layers - 1):
            self.han_convs.append(HANConv(
                in_channels={nt: hidden for nt in hetero.node_types},
                out_channels=hidden,
                heads=heads,
                dropout=self.dropout_val,
                metadata=metadata,
            ))

        self.lin = nn.Linear(hidden, num_classes)
        self.activation = nn.ReLU()

    def forward(self, hetero: HeteroData):
        x_dict = {nt: hetero[nt].x for nt in hetero.node_types}
        # 类型特征投影
        if len(self.projectors) > 0:
            x_dict = {k: self.projectors[k](v) if k in self.projectors else v for k, v in x_dict.items()}
        edge_index_dict = {et: hetero[et].edge_index for et in hetero.edge_types}

        out_dict = x_dict
        for i, conv in enumerate(self.han_convs):
            out_dict = conv(out_dict, edge_index_dict)
            if i < len(self.han_convs) - 1:
                out_dict = {k: self.activation(v) for k, v in out_dict.items()}
                out_dict = {k: F.dropout(v, p=self.dropout_val, training=self.training) for k, v in out_dict.items()}
        h = out_dict[self.category]
        logits = self.lin(h)
        return logits


def load_dblp_with_metapaths() -> Tuple[HeteroData, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], str]:
    """按 PyG DBLP + AddMetaPaths 方式加载 DBLP（与 two_teacher_kd 中一致逻辑）。"""
    import torch_geometric.transforms as T
    path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/DBLP')
    metapaths = [[('author', 'to', 'paper'), ('paper', 'to', 'author')],
                 [('author', 'to', 'paper'), ('paper', 'to', 'conference'),
                  ('conference', 'to', 'paper'), ('paper', 'to', 'author')]]
    transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, drop_unconnected_node_types=True)
    dataset = PYG_DBLP(path, transform=transform)
    data = dataset[0]
    category = 'author'
    idx_train = data[category].train_mask.nonzero(as_tuple=True)[0].long()
    idx_val = data[category].val_mask.nonzero(as_tuple=True)[0].long()
    idx_test = data[category].test_mask.nonzero(as_tuple=True)[0].long()
    return data, (idx_train, idx_val, idx_test), category


def main():
    p = argparse.ArgumentParser("HAN only training (TMDB / ArXiv / DBLP)")
    p.add_argument("-d", "--dataset", type=str, default="TMDB", choices=["TMDB", "ArXiv", "DBLP"])
    p.add_argument("--gpu_id", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)

    # HAN 超参
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--wd", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=100)

    # 可选：通过关系名传入元路径（仅 TMDB/ArXiv 使用；DBLP 内置）
    p.add_argument("--positional_relations", type=str, nargs='*', default=[],
                   help="metapaths expressed as comma-separated relation names, e.g. directed_by,directs performed_by,performs")

    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    # 加载数据
    if args.dataset == 'DBLP':
        hetero, (idx_train, idx_val, idx_test), category = load_dblp_with_metapaths()
        y = hetero[category].y.long()
        # 统一维度：选用 category 维度作为输入维度
        node_type_dims = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
        in_dim = hetero[category].x.size(1)
        hetero_for_han = hetero  # 已经通过 AddMetaPaths 处理
    else:
        # 其余数据集使用仓库 dataloader
        hetero, (idx_train, idx_val, idx_test), gen_node_feats, metapaths_names = load_data(dataset=args.dataset, return_mp=True)
        hetero = gen_node_feats(hetero)
        category = hetero.category
        y = hetero[category].y.long()
        node_type_dims = {nt: hetero[nt].x.size(1) for nt in hetero.node_types}
        in_dim = hetero[category].x.size(1)

        # 解析元路径（优先使用 CLI 指定，否则使用 dataloader 提供）
        rel_sequences = []
        rel_args = getattr(args, 'positional_relations', []) or []
        for item in rel_args:
            seq = [tok.strip() for tok in item.split(',') if tok.strip()]
            if seq:
                rel_sequences.append(seq)
        if not rel_sequences and metapaths_names:
            rel_sequences = metapaths_names

        mp_typed = parse_metapath_name_sequences(hetero, category, rel_sequences) if rel_sequences else []

        # 将元路径作为新关系添加到图中，供 HANConv 做语义聚合
        hetero_for_han = hetero
        if len(mp_typed) > 0:
            try:
                transform = T.AddMetaPaths(metapaths=mp_typed, drop_orig_edge_types=False, drop_unconnected_node_types=True)
                hetero_for_han = transform(hetero)
            except Exception as _:
                hetero_for_han = hetero

    # 准备训练
    hetero_for_han = hetero_for_han.to(device)
    y = y.to(device)
    idx_train, idx_val, idx_test = [t.to(device) for t in (idx_train, idx_val, idx_test)]

    num_classes = int(y.max().item()) + 1
    model = HANStack(
        hetero=hetero_for_han,
        category=category,
        in_dim=in_dim,
        hidden=args.hidden,
        num_classes=num_classes,
        heads=args.heads,
        dropout=args.dropout,
        num_layers=args.layers,
        node_type_dims={nt: hetero_for_han[nt].x.size(1) for nt in hetero_for_han.node_types},
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val, best_state, es = -1.0, None, 0
    print(f"[INFO] Training HAN on {args.dataset} | category={category} | classes={num_classes}")
    for ep in range(1, args.epochs + 1):
        model.train()
        logits = model(hetero_for_han)
        loss = F.cross_entropy(logits[idx_train], y[idx_train])
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(hetero_for_han)
            tr = accuracy(logits, y, idx_train)
            va = accuracy(logits, y, idx_val)
            te = accuracy(logits, y, idx_test)
        print(f"[HAN] ep {ep:03d} | loss {loss.item():.4f} | tr {tr:.4f} va {va:.4f} te {te:.4f}")

        if va >= best_val:
            best_val, best_state, es = va, { 'model': model.state_dict() }, 0
        else:
            es += 1
            if es >= args.patience:
                print("[HAN] early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state['model'])
        model.eval()
        with torch.no_grad():
            logits = model(hetero_for_han)
            tr = accuracy(logits, y, idx_train)
            va = accuracy(logits, y, idx_val)
            te = accuracy(logits, y, idx_test)
        print(f"[HAN] Final(best) | tr {tr:.4f} va {va:.4f} te {te:.4f}")


if __name__ == "__main__":
    main()


