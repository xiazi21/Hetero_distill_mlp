# han_tmdb.py
# ------------------------------------------------------------
# Pure HAN on TMDB (PyTorch Geometric), robust to local schema:
# - Loads TMDB data using dataloader.py
# - Maps node types and creates appropriate metapaths
# - Applies AddMetaPaths and trains HAN
# Target node type: 'movie'
# ------------------------------------------------------------
import os.path as osp
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv

from dataloader import load_data


def add_reverse_if_missing(data: HeteroData, src: str, dst: str):
    """Ensure there is a directed edge type from src->dst.
    If none exists but a dst->src exists, add a reverse edge."""
    # Check any canonical edge type with these endpoints exists:
    has_forward = any(s == src and d == dst for (s, _, d) in data.edge_types)
    if has_forward:
        return

    # Try to find any reverse edge
    for (s, rel, d) in list(data.edge_types):
        if s == dst and d == src:
            ei = data[(s, rel, d)].edge_index
            rev_rel = rel + "_rev"
            # Avoid name collision
            suffix_id = 1
            while (src, rev_rel, dst) in data.edge_types:
                suffix_id += 1
                rev_rel = f"{rel}_rev{suffix_id}"
            data[(src, rev_rel, dst)].edge_index = ei.flip(0).contiguous()
            print(f"[INFO] Added reverse relation for ({dst} --{rel}--> {src}) as ({src} --{rev_rel}--> {dst})")
            return

    # If we get here, there was no way to construct src->dst
    raise RuntimeError(f"No edge type found to support direction {src} -> {dst}. Available edge types: {data.edge_types}")


def main():
    print("[INFO] Starting TMDB HAN training...")
    # 0) Load TMDB data using dataloader
    print("[INFO] Loading TMDB data...")
    data, (idx_train, idx_val, idx_test), generate_node_features, metapaths = load_data(dataset='TMDB', return_mp=True)
    print("[INFO] TMDB data loaded successfully")
    
    # Apply feature generation
    data = generate_node_features(data)

    # 1) Inspect schema
    print("[DEBUG] Node types:", data.node_types)
    print("[DEBUG] Edge types:", data.edge_types)
    print("[DEBUG] Category:", data.category)

    # 2) Create train/val/test masks for movie nodes
    category = data.category  # Should be 'movie'
    num_movies = data[category].num_nodes
    
    # Create masks
    train_mask = torch.zeros(num_movies, dtype=torch.bool)
    val_mask = torch.zeros(num_movies, dtype=torch.bool)
    test_mask = torch.zeros(num_movies, dtype=torch.bool)
    
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    
    data[category].train_mask = train_mask
    data[category].val_mask = val_mask
    data[category].test_mask = test_mask

    # 3) Define metapaths using *node-type pairs* expected by AddMetaPaths
    #    MAM: movie -> actor -> movie
    #    MDM: movie -> director -> movie
    # Create metapaths based on available edge types
    edge_types = data.edge_types
    print("[DEBUG] Available edge types:", edge_types)
    
    metapaths_node_pairs = []
    
    # Create metapaths based on available relationships
    if ('actor', 'performs', 'movie') in edge_types and ('movie', 'performed_by', 'actor') in edge_types:
        metapaths_node_pairs.append([('movie', 'actor'), ('actor', 'movie')])
    if ('director', 'directs', 'movie') in edge_types and ('movie', 'directed_by', 'director') in edge_types:
        metapaths_node_pairs.append([('movie', 'director'), ('director', 'movie')])
    
    print("[DEBUG] Using metapaths (node-type pairs):", metapaths_node_pairs)

    # 4) Ensure each directed pair used above exists as a canonical edge type.
    #    If not, but the reverse exists, we add a reverse edge.
    needed_pairs = set()
    for path in metapaths_node_pairs:
        for (s, d) in path:
            needed_pairs.add((s, d))
    for (s, d) in sorted(needed_pairs):
        # Probe whether an s->d edge type exists:
        forward_exists = any(ss == s and dd == d for (ss, _, dd) in data.edge_types)
        if not forward_exists:
            # Try to synthesize from a reverse
            add_reverse_if_missing(data, s, d)

    # 5) Apply AddMetaPaths (now that directions are guaranteed to exist)
    add_mp = T.AddMetaPaths(
        metapaths=metapaths_node_pairs,
        drop_orig_edge_types=True,        # only use meta-path induced edges
        drop_unconnected_node_types=False,  # Keep all node types for HAN
    )
    data = add_mp(data)
    print("[DEBUG] After AddMetaPaths - node types:", data.node_types)
    print("[DEBUG] After AddMetaPaths - edge types:", data.edge_index_dict.keys())

    # 6) Build HAN (2 layers) for 'movie' classification
    out_channels = int(data[category].y.max().item()) + 1

    class HAN(nn.Module):
        def __init__(self,
                     in_channels: Union[int, Dict[str, int]],
                     out_channels: int,
                     hidden_channels: int = 128,
                     heads: int = 8,
                     dropout: float = 0.6):
            super().__init__()
            self.h1 = HANConv(in_channels, hidden_channels, heads=heads,
                              dropout=dropout, metadata=data.metadata())
            self.h2 = HANConv(hidden_channels, hidden_channels, heads=heads,
                              dropout=dropout, metadata=data.metadata())
            self.lin = nn.Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            out = self.h1(x_dict, edge_index_dict)
            out = {k: F.elu(v) for k, v in out.items() if v is not None}
            out = self.h2(out, edge_index_dict)
            return self.lin(out[category])

    # 7) Device & init
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch_geometric.is_xpu_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    model = HAN(in_channels=-1, out_channels=out_channels).to(device)
    data = data.to(device)

    with torch.no_grad():  # initialize lazy modules
        _ = model(data.x_dict, data.edge_index_dict)

    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    def train() -> float:
        model.train()
        opt.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data[category].train_mask
        loss = F.cross_entropy(out[mask], data[category].y[mask])
        loss.backward()
        opt.step()
        return float(loss)

    @torch.no_grad()
    def test() -> List[float]:
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
        accs = []
        for split in ['train_mask', 'val_mask', 'test_mask']:
            mask = data[category][split]
            acc = (pred[mask] == data[category].y[mask]).sum() / mask.sum()
            accs.append(float(acc))
        return accs

    # 8) Train with patience
    best_val = best_test = best_train = 0.0
    best_epoch = 0
    start_patience = patience = 100

    for epoch in range(1, 201):
        loss = train()
        train_acc, val_acc, test_acc = test()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
              f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

        if val_acc >= best_val:
            best_val, best_test, best_train, best_epoch = val_acc, test_acc, train_acc, epoch
            patience = start_patience
        else:
            patience -= 1
            if patience <= 0:
                print(f"Early stop: no val improvement for {start_patience} epochs.")
                break

    print(f"Best@epoch {best_epoch:03d} -> "
          f"Val: {best_val:.4f}, Test: {best_test:.4f}, Train: {best_train:.4f}")


if __name__ == "__main__":
    main()
