import os.path as osp
from typing import Dict, List, Union, Optional
import argparse

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB, DBLP
from torch_geometric.nn import HANConv

# 支持多个数据集
def load_dataset(dataset_name='IMDB'):
    if dataset_name == 'IMDB':
        path = osp.join(osp.dirname(osp.realpath(__file__)), '../../data/IMDB')
        metapaths = [[('movie', 'actor'), ('actor', 'movie')],
                     [('movie', 'director'), ('director', 'movie')]]
        transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True,
                                   drop_unconnected_node_types=True)
        dataset = IMDB(path, transform=transform)
        data = dataset[0]
        category = 'movie'
    elif dataset_name == 'DBLP':
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
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return data, category

# 默认使用IMDB数据集以保持向后兼容
data, category = load_dataset('IMDB')


class HAN(nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8, category='movie', metadata=None,
                 node_type_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.category = category
        
        # 添加投影层：将不同节点类型的特征投影到统一维度
        self.node_type_dims = node_type_dims
        self.projectors = nn.ModuleDict()
        if node_type_dims is not None:
            for nt, dim in node_type_dims.items():
                if isinstance(in_channels, dict):
                    target_dim = in_channels[nt]
                else:
                    target_dim = in_channels
                if dim != target_dim:
                    self.projectors[nt] = nn.Linear(dim, target_dim, bias=False)
                else:
                    self.projectors[nt] = nn.Identity()
        
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=0.6, metadata=metadata)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # 应用投影层，将所有节点类型投影到统一维度
        if len(self.projectors) > 0:
            x_dict = {k: self.projectors[k](v) if k in self.projectors else v 
                     for k, v in x_dict.items()}
        
        out = self.han_conv(x_dict, edge_index_dict)
        out = self.lin(out[self.category])
        return out


def main():
    parser = argparse.ArgumentParser(description='HAN Test Script')
    parser.add_argument('--dataset', type=str, default='IMDB', choices=['IMDB', 'DBLP'],
                        help='Dataset to use (default: IMDB)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden channels')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    args = parser.parse_args()
    
    # 加载数据集
    data, category = load_dataset(args.dataset)
    
    # 确定输出类别数
    if args.dataset == 'IMDB':
        num_classes = 3
    elif args.dataset == 'DBLP':
        num_classes = 4  # DBLP有4个类别
    
    # 收集所有节点类型的特征维度
    node_type_dims = {nt: data[nt].x.size(1) for nt in data.node_types}
    # 使用统一的投影维度（选择category节点的维度）
    unified_dim = data[category].x.size(1)
    
    # 为HANConv准备输入维度字典
    in_channels = {nt: unified_dim for nt in data.node_types}
    
    model = HAN(in_channels=in_channels, out_channels=num_classes, 
                hidden_channels=args.hidden_channels, heads=args.heads, category=category, 
                metadata=data.metadata(), node_type_dims=node_type_dims)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch_geometric.is_xpu_available():
        device = torch.device('xpu')
    else:
        device = torch.device('cpu')
    
    data, model = data.to(device), model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def train() -> float:
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data[category].train_mask
        loss = F.cross_entropy(out[mask], data[category].y[mask])
        loss.backward()
        optimizer.step()
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

    best_val_acc = 0
    start_patience = patience = args.patience
    
    print(f"Training HAN on {args.dataset} dataset...")
    print(f"Category: {category}, Classes: {num_classes}")
    
    for epoch in range(1, args.epochs + 1):
        loss = train()
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        if best_val_acc <= val_acc:
            patience = start_patience
            best_val_acc = val_acc
        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                  f'for {start_patience} epochs')
            break

if __name__ == "__main__":
    main()