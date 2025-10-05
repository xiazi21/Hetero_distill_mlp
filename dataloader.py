# -*- coding: utf-8 -*-  # 指定源文件的编码格式，保证中文注释在不同环境下正常显示
"""
DGL→PyG 数据加载脚本迁移（逐行中文注释版）
================================================
目的：
- 将原始 DGL 版的 `load_data` 迁移为基于 PyTorch Geometric 的 HeteroData 实现；
- 保持函数签名与返回项尽量一致：`load_data(dataset='TMDB', return_mp=False)`；
- 仍返回：(hetero_data, (idx_train, idx_val, idx_test), generate_node_features[, metapaths])；
- 兼容 PyG 1.x/2.x；
- 用 PyG 惯例字段 `.x`/`.y`，同时保留 `.feat`/`.label` 副本以便旧代码平滑过渡。

注意：
- DGL 的 `update_all` 用于消息传递 + 聚合；在 PyG 中，这里用 `scatter_mean` 手工实现简单的“从 src 到 dst 的均值聚合”。
- DGL 的 `to_simple/remove_self_loop/add_self_loop` 在 PyG 中分别用 `coalesce/remove_self_loops/add_self_loops`（按关系逐一处理）。
- 异构图使用 `HeteroData` 管理不同类型节点与边；每条关系的 `edge_index` 形状为 [2, E]；
- 未提供节点特征的节点，若需要，可通过邻居聚合生成（与原脚本一致）。
"""

import os.path as osp  # 导入 os.path 别名以处理文件路径
import pickle  # 导入 pickle 以便加载 .pkl 数据
from typing import List, Tuple  # 导入类型注解，便于阅读与 IDE 智能提示

import numpy as np  # 导入 numpy 以处理数组
import torch  # 导入 PyTorch 作为张量与数值运算基础
from torch import Tensor  # 导入张量类型别名
try:
    from torch_scatter import scatter_mean  # 导入 scatter_mean 以实现按索引的均值聚合
except ImportError:
    # Fallback implementation when torch_scatter is not available
    def scatter_mean(src, index, dim=0, dim_size=None):
        """Fallback implementation of scatter_mean"""
        if dim_size is None:
            dim_size = index.max().item() + 1
        
        # Create output tensor
        out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
        
        # Use index_add_ for aggregation
        out.index_add_(dim, index, src)
        
        # Count occurrences for normalization
        count = torch.zeros(dim_size, device=src.device, dtype=torch.long)
        count.index_add_(dim, index, torch.ones_like(index))
        
        # Avoid division by zero
        count = count.clamp_min(1).float()
        
        return out / count.unsqueeze(-1)
from torch_geometric.data import HeteroData  # 导入 PyG 的异构图数据结构 HeteroData
from torch_geometric.utils import (  # 导入若干图操作工具函数
    coalesce,  # 合并重复边并排序
    remove_self_loops,  # 移除自环边（u==v）
    add_self_loops,  # 为同型关系添加自环
)
from torch_geometric.datasets import DBLP as PYG_DBLP, AMiner as PYG_AMINER  # PyG datasets
from torch_geometric.transforms import RandomNodeSplit


# =========================
# 工具函数：类型与索引处理
# =========================

def _to_long(x: Tensor) -> Tensor:
    """将张量强制转换为 int64（PyG 的索引类型约定），并保证内存连续。"""
    return x.to(dtype=torch.long).contiguous()  # 转换为 torch.long 并 contiguous，避免后续操作潜在报错


def _stack_edge_index(src: Tensor, dst: Tensor) -> Tensor:
    """将 src/dst 两个一维索引堆叠为 PyG 期望的形状 [2, E] 的 edge_index。"""
    src = _to_long(src)  # 确保源端索引为 int64
    dst = _to_long(dst)  # 确保目标端索引为 int64
    return torch.stack([src, dst], dim=0)  # 在第 0 维堆叠得到 [2, E]


def _infer_num_nodes_from_edges(edge_index: Tensor) -> int:
    """根据边索引推断节点数（最大索引 + 1）。"""
    return int(edge_index.max().item()) + 1  # 取最大索引并加 1，得到至少需要的节点计数


def _coalesce_relation(edge_index: Tensor, size: Tuple[int, int] = None) -> Tensor:
    """对某个关系的边做去重/排序；size 可指定 (num_src, num_dst)。"""
    # PyG coalesce 不支持 size 参数，改为使用 num_nodes 参数（单个整数）
    if size is not None:
        # 对于二分图，使用源节点数作为 num_nodes
        num_nodes = size[0]
        return coalesce(edge_index, num_nodes=num_nodes, is_sorted=False, sort_by_row=True)
    else:
        return coalesce(edge_index, is_sorted=False, sort_by_row=True)


def _remove_and_add_self_loops_homo(edge_index: Tensor, num_nodes: int) -> Tensor:
    """用于同型关系（src_type==dst_type）的移除自环并添加自环。"""
    ei, _ = remove_self_loops(edge_index)  # 先删除已有的自环，避免重复
    ei, _ = add_self_loops(ei, num_nodes=num_nodes)  # 再为每个节点添加一个自环
    return ei  # 返回处理后的边索引


def _neighbor_mean_aggregate(
    src_x: Tensor,  # 源节点特征矩阵 [N_src, F]
    edge_index: Tensor,  # 关系的边索引 [2, E]
    num_dst: int,  # 目标节点数量
) -> Tensor:
    """按关系从源节点到目标节点做均值聚合，返回目标节点特征 [N_dst, F]。"""
    row, col = edge_index  # 解包边索引，row 表示源端索引，col 表示目标端索引
    msg = src_x[row]  # 从源端取出消息（即特征）
    out = scatter_mean(msg, index=col, dim=0, dim_size=num_dst)  # 以目标端索引为组做均值，未出现的目标端填 0
    return out  # 返回聚合后的目标端特征


# =========================
# 主函数：数据加载（与原接口兼容）
# =========================

def load_data(dataset: str = 'TMDB', return_mp: bool = False):
    """加载指定数据集并构造 PyG 的 HeteroData；保持与原 DGL 版相同的返回项。"""
    if dataset == 'TMDB':  # 分支：TMDB 电影数据集
        with open('./data/tmdb.pkl', 'rb') as f:  # 打开 pkl 文件
            network = pickle.load(f)  # 反序列化得到字典结构
        num_movies = len(network['movie_labels'])  # 电影节点数由标签长度决定

        # 读取异构边：电影-演员、电影-导演（均为 numpy 数组）
        movie_actor_mid = torch.from_numpy(network['movie-actor'][0])  # 电影端索引（与演员相连）
        movie_actor_aid = torch.from_numpy(network['movie-actor'][1])  # 演员端索引（与电影相连）
        movie_director_mid = torch.from_numpy(network['movie-director'][0])  # 电影端索引（与导演相连）
        movie_director_did = torch.from_numpy(network['movie-director'][1])  # 导演端索引（与电影相连）

        # ------------------------ 构建 HeteroData ------------------------
        data = HeteroData()  # 新建异构图容器
        data['movie'].num_nodes = int(num_movies)  # 指定电影节点数（来自标签长度）
        data['actor'].num_nodes = int(movie_actor_aid.max().item() + 1)  # 演员节点数由最大索引+1推断
        data['director'].num_nodes = int(movie_director_did.max().item() + 1)  # 导演节点数同理

        # 添加自环关系（movie 自环）：与原 DGL 中 ('movie','self-loop','movie') 等价
        self_loop_ei = _stack_edge_index(torch.arange(num_movies), torch.arange(num_movies))  # 构造自环索引
        data[('movie', 'self-loop', 'movie')].edge_index = self_loop_ei  # 赋值到对应关系

        # 添加演员相关双向关系：actor→movie 与 movie→actor
        data[('actor', 'performs', 'movie')].edge_index = _stack_edge_index(movie_actor_aid, movie_actor_mid)  # 演员到电影
        data[('movie', 'performed_by', 'actor')].edge_index = _stack_edge_index(movie_actor_mid, movie_actor_aid)  # 电影到演员

        # 添加导演相关双向关系：director→movie 与 movie→director
        data[('director', 'directs', 'movie')].edge_index = _stack_edge_index(movie_director_did, movie_director_mid)  # 导演到电影
        data[('movie', 'directed_by', 'director')].edge_index = _stack_edge_index(movie_director_mid, movie_director_did)  # 电影到导演

        # 对所有关系做 coalesce（等价于 DGL 的 to_simple：去重/排序，不回写映射）
        for et in data.edge_types:  # 遍历每条关系
            st, _, dt = et  # 解包 src_type / rel / dst_type
            num_src = data[st].num_nodes  # 源节点数
            num_dst = data[dt].num_nodes  # 目标节点数
            ei = data[et].edge_index  # 取出关系边索引
            data[et].edge_index = _coalesce_relation(ei, size=(num_src, num_dst))  # coalesce 并回写

        # 加载电影节点特征
        movie_feat = torch.from_numpy(network['movie_feats']).to(torch.float32)  # 读取并转 float32
        data['movie'].x = movie_feat  # 按 PyG 惯例存入 x
        data['movie'].feat = movie_feat  # 同时保留 feat 字段，兼容旧代码

        # ------- 定义特征生成函数（与原 DGL generate_node_features 对齐） -------
        def generate_node_features(g: HeteroData) -> HeteroData:  # 传入/返回 HeteroData
            # 从 movie→actor（performed_by 的反向）聚合，得到演员特征
            ei = g[('movie', 'performed_by', 'actor')].edge_index  # 取出边索引（movie→actor）
            num_actor = g['actor'].num_nodes  # 演员节点数
            actor_x = _neighbor_mean_aggregate(g['movie'].x, ei, num_dst=num_actor)  # 将电影特征均值聚合到演员
            g['actor'].x = actor_x  # 存入演员 x
            g['actor'].feat = actor_x  # 兼容字段 feat

            # 从 movie→director（directed_by 的反向）聚合，得到导演特征
            ei = g[('movie', 'directed_by', 'director')].edge_index  # 取出边索引（movie→director）
            num_director = g['director'].num_nodes  # 导演节点数
            director_x = _neighbor_mean_aggregate(g['movie'].x, ei, num_dst=num_director)  # 聚合得到导演特征
            g['director'].x = director_x  # 存入导演 x
            g['director'].feat = director_x  # 兼容字段 feat
            return g  # 返回修改后的图

        # 加载标签并赋给 movie 节点
        labels = torch.tensor(network['movie_labels'], dtype=torch.long)  # 读取标签为 int64
        data['movie'].y = labels  # PyG 惯例：用 y 存标签
        data['movie'].label = labels  # 兼容旧字段名 label

        # 记录“中心类型”（与原 g.category 一致）
        data.category = 'movie'  # 为下游代码保留中心类别信息

        # 基于年份划分训练/验证/测试索引（与原脚本保持一致）
        movie_years = network['movie_years']  # 取年份数组（numpy）
        idx_train = torch.from_numpy((movie_years <= 2015).nonzero()[0]).long()  # 2015 及以前为训练集
        idx_val = torch.from_numpy(((movie_years >= 2016) & (movie_years <= 2018)).nonzero()[0]).long()  # 2016-2018 验证
        idx_test = torch.from_numpy((movie_years >= 2019).nonzero()[0]).long()  # 2019 及以后为测试

        # 定义元路径（与原返回一致）：用于上层可能的元路径操作
        metapaths: List[List[str]] = [['directed_by', 'directs'], ['performed_by', 'performs']]  # 两条长度-2 元路径

    elif dataset == 'CroVal':  # 分支：CroVal 问答数据集
        with open('./data/croval.pkl', 'rb') as f:  # 打开 pkl 文件
            network = pickle.load(f)  # 反序列化

        # 读取边（保持与原 DGL 脚本的方向与拼接逻辑一致）
        question_user_qid = torch.tensor(network['question-user'][0], dtype=torch.int32)  # 问题端索引
        question_user_uid = torch.tensor(network['question-user'][1], dtype=torch.int32)  # 用户端索引
        question_src_id = torch.tensor(network['question-question'][0], dtype=torch.int32)  # 问题-问题边源端
        question_dst_id = torch.tensor(network['question-question'][1], dtype=torch.int32)  # 问题-问题边目标端
        question_tag_qid = torch.tensor(network['question-tag'][0], dtype=torch.int32)  # 问题-标签边问题端
        question_tag_tid = torch.tensor(network['question-tag'][1], dtype=torch.int32)  # 问题-标签边标签端

        data = HeteroData()  # 新建异构图

        # 依据特征推断节点数（问题与标签提供特征，用户数量从边最大索引推断）
        num_questions = int(network['question_feats'].shape[0])  # 问题节点数由特征行数确定
        num_tags = int(network['tag_feats'].shape[0])  # 标签节点数由特征行数确定
        num_users = int(question_user_uid.max().item() + 1)  # 用户节点数由最大索引+1确定
        data['question'].num_nodes = num_questions  # 设定问题节点数
        data['tag'].num_nodes = num_tags  # 设定标签节点数
        data['user'].num_nodes = num_users  # 设定用户节点数

        # 构造并赋值关系边索引（含双向与同型关系）
        data[('user', 'asks', 'question')].edge_index = _stack_edge_index(question_user_uid, question_user_qid)  # 用户→问题
        data[('question', 'asked_by', 'user')].edge_index = _stack_edge_index(question_user_qid, question_user_uid)  # 问题→用户
        # 问题-问题：按原逻辑拼接双向（等价于无向），这里直接 concat 后堆叠
        links_src = torch.cat([question_src_id, question_dst_id], dim=0)  # 拼接源端
        links_dst = torch.cat([question_dst_id, question_src_id], dim=0)  # 拼接目标端
        data[('question', 'links', 'question')].edge_index = _stack_edge_index(links_src, links_dst)  # 赋值同型关系
        data[('question', 'contains', 'tag')].edge_index = _stack_edge_index(question_tag_qid, question_tag_tid)  # 问题→标签
        data[('tag', 'contained_by', 'question')].edge_index = _stack_edge_index(question_tag_tid, question_tag_qid)  # 标签→问题

        # 对所有关系做 coalesce（等价 DGL 的 to_simple）
        for et in data.edge_types:  # 遍历所有关系
            st, _, dt = et  # 解包类型
            num_src = data[st].num_nodes  # 源节点数
            num_dst = data[dt].num_nodes  # 目标节点数
            ei = data[et].edge_index  # 取边索引
            data[et].edge_index = _coalesce_relation(ei, size=(num_src, num_dst))  # 去重/排序

        # 对 question→question 的同型边执行“移除自环 + 添加自环”（与 DGL 行为对应）
        ei = data[('question', 'links', 'question')].edge_index  # 取出同型边
        data[('question', 'links', 'question')].edge_index = _remove_and_add_self_loops_homo(ei, num_nodes=num_questions)  # 处理自环

        # 加载问题与标签特征
        q_feat = torch.from_numpy(network['question_feats']).to(torch.float32)  # 问题特征
        t_feat = torch.from_numpy(network['tag_feats']).to(torch.float32)  # 标签特征
        data['question'].x = q_feat  # 按 PyG 惯例存入 x
        data['question'].feat = q_feat  # 兼容旧字段 feat
        data['tag'].x = t_feat  # 存入标签 x
        data['tag'].feat = t_feat  # 兼容字段 feat

        # 定义节点特征生成：从 question→user（asked_by 的方向）聚合生成用户特征
        def generate_node_features(g: HeteroData) -> HeteroData:  # 传入/返回 HeteroData
            ei = g[('question', 'asked_by', 'user')].edge_index  # 取出问题→用户的边
            num_user = g['user'].num_nodes  # 用户节点数
            user_x = _neighbor_mean_aggregate(g['question'].x, ei, num_dst=num_user)  # 聚合问题特征到用户
            g['user'].x = user_x  # 存入用户 x
            g['user'].feat = user_x  # 同步 feat 字段
            return g  # 返回修改后的图

        # 读取问题标签并赋值
        labels = torch.tensor(network['question_labels'], dtype=torch.long)  # 问题分类标签
        data['question'].y = labels  # 存入 y
        data['question'].label = labels  # 同步 label 字段
        data.category = 'question'  # 标记中心类型为 question

        # 基于年份划分集合（保持与原始 DGL 脚本一致的阈值）
        q_years = network['question_years']  # 年份数组
        idx_train = torch.from_numpy((q_years <= 2015).nonzero()[0]).long()  # 训练索引
        idx_val = torch.from_numpy(((q_years >= 2016) & (q_years <= 2018)).nonzero()[0]).long()  # 验证索引
        idx_test = torch.from_numpy((q_years >= 2019).nonzero()[0]).long()  # 测试索引

        # 返回的元路径（与原版一致）
        metapaths: List[List[str]] = [['contains', 'contained_by'], ['asked_by', 'asks']]  # 两条长度-2 元路径

    elif dataset == 'ArXiv':  # 分支：ArXiv 引文/写作网络
        with open('./data/arxiv.pkl', 'rb') as f:  # 打开 pkl 文件
            network = pickle.load(f)  # 反序列化

        # 读取边：论文-论文（引文，按原代码拼接双向），论文-作者双向
        paper_src_id = torch.from_numpy(network['paper-paper'][0])  # 论文-论文源端
        paper_dst_id = torch.from_numpy(network['paper-paper'][1])  # 论文-论文目标端
        paper_author_pid = torch.from_numpy(network['paper-author'][0])  # 论文端索引
        paper_author_aid = torch.from_numpy(network['paper-author'][1])  # 作者端索引

        data = HeteroData()  # 新建异构图

        # 由特征得到论文节点数；作者节点数由最大索引+1 推断
        num_papers = int(network['paper_feats'].shape[0])  # 论文节点数
        num_authors = int(paper_author_aid.max().item() + 1)  # 作者节点数
        data['paper'].num_nodes = num_papers  # 设定论文节点数
        data['author'].num_nodes = num_authors  # 设定作者节点数

        # 构造论文-论文边（拼接双向）
        cites_src = torch.cat([paper_src_id, paper_dst_id], dim=0)  # 源端拼接
        cites_dst = torch.cat([paper_dst_id, paper_src_id], dim=0)  # 目标端拼接
        data[('paper', 'cites', 'paper')].edge_index = _stack_edge_index(cites_src, cites_dst)  # 赋值同型边

        # 构造论文-作者双向关系
        data[('author', 'writes', 'paper')].edge_index = _stack_edge_index(paper_author_aid, paper_author_pid)  # 作者→论文
        data[('paper', 'written_by', 'author')].edge_index = _stack_edge_index(paper_author_pid, paper_author_aid)  # 论文→作者

        # 对所有关系做 coalesce
        for et in data.edge_types:  # 遍历关系
            st, _, dt = et  # 解包类型
            num_src = data[st].num_nodes  # 源节点数
            num_dst = data[dt].num_nodes  # 目标节点数
            ei = data[et].edge_index  # 取边索引
            data[et].edge_index = _coalesce_relation(ei, size=(num_src, num_dst))  # 去重/排序

        # 对 paper→paper 引文关系执行“移除自环 + 添加自环”（与原逻辑一致）
        ei = data[('paper', 'cites', 'paper')].edge_index  # 取引文边
        data[('paper', 'cites', 'paper')].edge_index = _remove_and_add_self_loops_homo(ei, num_nodes=num_papers)  # 处理自环

        # 加载论文特征
        paper_feat = torch.from_numpy(network['paper_feats']).to(torch.float32)  # 论文特征
        data['paper'].x = paper_feat  # 存入 x
        data['paper'].feat = paper_feat  # 同步 feat 字段

        # 定义作者特征生成：从 paper→author（written_by 的方向）均值聚合
        def generate_node_features(g: HeteroData) -> HeteroData:  # 传入/返回 HeteroData
            ei = g[('paper', 'written_by', 'author')].edge_index  # 取论文→作者边
            num_author = g['author'].num_nodes  # 作者节点数
            author_x = _neighbor_mean_aggregate(g['paper'].x, ei, num_dst=num_author)  # 聚合到作者
            g['author'].x = author_x  # 存入作者 x
            g['author'].feat = author_x  # 同步 feat 字段
            return g  # 返回图

        # 加载论文标签
        labels = torch.tensor(network['paper_labels'], dtype=torch.long)  # 论文标签
        data['paper'].y = labels  # 存入 y
        data['paper'].label = labels  # 同步 label 字段
        data.category = 'paper'  # 中心类型为 paper

        # 基于年份划分数据集（与原脚本阈值一致）
        paper_years = network['paper_years']  # 年份数组
        idx_train = torch.from_numpy((paper_years <= 2017).nonzero()[0]).long()  # 训练集
        idx_val = torch.from_numpy(((paper_years >= 2018) & (paper_years <= 2018)).nonzero()[0]).long()  # 验证集
        idx_test = torch.from_numpy((paper_years >= 2019).nonzero()[0]).long()  # 测试集

        # 元路径定义（与原返回一致）
        metapaths: List[List[str]] = [['cites'], ['written_by', 'writes']]  # cites 为长度-1，另一条长度-2

    elif 'IGB' in dataset:  # 分支：IGB（Industry Graph Benchmark，变体名中包含 IGB）
        # 解析数据规模与类别数（与原脚本一致）
        dataset_size = 'tiny' if dataset.split('-')[-2] == '549K' else 'small'  # 依据名称推断大小
        num_classes = 19 if dataset.split('-')[-1] == '19' else 2983  # 根据后缀决定类别数
        dir_path = osp.join('data', 'igb', dataset_size, 'processed')  # 构造数据根目录

        # 加载异构边（均为 numpy .npy 文件）
        paper_paper_edges = torch.from_numpy(
            np.load(osp.join(dir_path, 'paper__cites__paper', 'edge_index.npy'))
        )  # 论文-论文边（形如 [E, 2] 或 [2, E]，这里按原数据为 [E, 2] 假定）
        paper_author_edges = torch.from_numpy(
            np.load(osp.join(dir_path, 'paper__written_by__author', 'edge_index.npy'))
        )  # 论文-作者边
        author_institute_edges = torch.from_numpy(
            np.load(osp.join(dir_path, 'author__affiliated_to__institute', 'edge_index.npy'))
        )  # 作者-机构边
        paper_fos_edges = torch.from_numpy(
            np.load(osp.join(dir_path, 'paper__topic__fos', 'edge_index.npy'))
        )  # 论文-学科（fos）边

        # 若上述边为形状 [E, 2]，需转置为 [2, E]；若已为 [2, E] 则保持
        def _ensure_2E(e):  # 小工具：保证边索引为 [2, E]
            return e.t().contiguous() if e.dim() == 2 and e.shape[0] != 2 else e  # 判断首维是否为 2，否则转置

        paper_paper_ei = _ensure_2E(paper_paper_edges)  # 处理论文-论文边
        paper_author_ei = _ensure_2E(paper_author_edges)  # 处理论文-作者边
        author_inst_ei = _ensure_2E(author_institute_edges)  # 处理作者-机构边
        paper_fos_ei = _ensure_2E(paper_fos_edges)  # 处理论文-学科边

        data = HeteroData()  # 新建异构图

        # 从节点特征文件推断节点数（论文与学科），其他类型从边最大索引推断
        paper_features = torch.from_numpy(np.load(osp.join(dir_path, 'paper', 'node_feat.npy'))).to(torch.float32)  # 论文特征
        data['paper'].x = paper_features  # 存入论文 x
        data['paper'].feat = paper_features  # 同步 feat
        data['paper'].num_nodes = paper_features.size(0)  # 论文节点数

        fos_features = torch.from_numpy(np.load(osp.join(dir_path, 'fos', 'node_feat.npy'))).to(torch.float32)  # 学科特征
        data['fos'].x = fos_features  # 存入学科 x
        data['fos'].feat = fos_features  # 同步 feat
        data['fos'].num_nodes = fos_features.size(0)  # 学科节点数

        # 依据边索引推断作者与机构节点数
        data['author'].num_nodes = max(_infer_num_nodes_from_edges(paper_author_ei[1:2, :]),  # 从论文→作者边的作者端推断
                                       _infer_num_nodes_from_edges(author_inst_ei[0:1, :]))  # 从作者→机构边的作者端推断
        data['institute'].num_nodes = _infer_num_nodes_from_edges(author_inst_ei[1:2, :])  # 机构节点数来自目标端

        # 赋值各关系的边索引（注意方向与原脚本一致）
        data[('paper', 'cites', 'paper')].edge_index = paper_paper_ei  # 论文→论文（引文）
        data[('author', 'writes', 'paper')].edge_index = torch.stack([paper_author_ei[1], paper_author_ei[0]], dim=0)  # 作者→论文
        data[('paper', 'written_by', 'author')].edge_index = paper_author_ei  # 论文→作者
        data[('author', 'affiliated_with', 'institute')].edge_index = author_inst_ei  # 作者→机构
        data[('institute', 'affiliates', 'author')].edge_index = torch.stack([author_inst_ei[1], author_inst_ei[0]], dim=0)  # 机构→作者
        data[('fos', 'topics', 'paper')].edge_index = torch.stack([paper_fos_ei[1], paper_fos_ei[0]], dim=0)  # 学科→论文
        data[('paper', 'has_topic', 'fos')].edge_index = paper_fos_ei  # 论文→学科

        # 对所有关系做 coalesce（去重/排序）
        for et in data.edge_types:  # 遍历关系
            st, _, dt = et  # 解包类型
            num_src = data[st].num_nodes  # 源节点数
            num_dst = data[dt].num_nodes  # 目标节点数
            ei = data[et].edge_index  # 取边索引
            data[et].edge_index = _coalesce_relation(ei, size=(num_src, num_dst))  # 应用 coalesce

        # 对论文-论文同型关系执行“移除自环 + 添加自环”（与原始脚本一致）
        ei = data[('paper', 'cites', 'paper')].edge_index  # 取引文边
        data[('paper', 'cites', 'paper')].edge_index = _remove_and_add_self_loops_homo(ei, num_nodes=data['paper'].num_nodes)  # 处理自环

        # 加载论文标签（19 类或 2K+ 类）
        if num_classes == 19:  # 分支：19 类
            paper_labels = torch.tensor(
                np.load(osp.join(dir_path, 'paper', 'node_label_19.npy')), dtype=torch.long
            )  # 读入 19 类标签
        else:  # 分支：2983 类
            paper_labels = torch.tensor(
                np.load(osp.join(dir_path, 'paper', 'node_label_2K.npy')), dtype=torch.long
            )  # 读入 2K+ 类标签
        data['paper'].y = paper_labels  # 存入 y
        data['paper'].label = paper_labels  # 同步 label
        data.category = 'paper'  # 中心类型为 paper

        # 年份划分训练/验证/测试
        paper_years = np.load(osp.join(dir_path, 'paper', 'paper_year.npy'))  # 读取年份数组
        idx_train = torch.from_numpy((paper_years <= 2016).nonzero()[0]).long()  # 训练索引
        idx_val = torch.from_numpy(((paper_years >= 2017) & (paper_years <= 2018)).nonzero()[0]).long()  # 验证索引
        idx_test = torch.from_numpy((paper_years >= 2019).nonzero()[0]).long()  # 测试索引

        # 元路径定义（与原返回一致）
        metapaths: List[List[str]] = [['cites'], ['written_by', 'writes']]  # cites + 写作双向



    elif dataset == 'DBLP':
        # --- Load from PyG (downloads on first run) ---
        pyg_ds = PYG_DBLP(root='./data/pyg_dblp')  # creates ./data/pyg_dblp/*
        data: HeteroData = pyg_ds[0]

        # DBLP task: author classification (4 classes). PyG ships masks for authors.
        assert 'author' in data.node_types, 'DBLP must have author node type'
        assert data['author'].y is not None, 'DBLP must provide author labels'
        assert 'train_mask' in data['author'] and 'val_mask' in data['author'] and 'test_mask' in data['author'], \
            'DBLP in PyG provides predefined masks for authors'

        # Build indices from masks:
        idx_train = data['author'].train_mask.nonzero(as_tuple=True)[0].long()
        idx_val   = data['author'].val_mask.nonzero(as_tuple=True)[0].long()
        idx_test  = data['author'].test_mask.nonzero(as_tuple=True)[0].long()

        # Keep consistent field aliases + category
        data['author'].label = data['author'].y
        for ntype in data.node_types:
            if 'x' in data[ntype]:
                data[ntype].feat = data[ntype].x
        data.category = 'author'

        # Optional self-loops for homogenous relations (DBLP has no paper-paper by default)
        if ('paper', 'cites', 'paper') in data.edge_types:
            ei = data[('paper', 'cites', 'paper')].edge_index
            data[('paper', 'cites', 'paper')].edge_index = _remove_and_add_self_loops_homo(ei, data['paper'].num_nodes)

        # Coalesce edges (keeps your original behavior)
        for et in data.edge_types:
            st, _, dt = et
            data[et].edge_index = _coalesce_relation(data[et].edge_index, (data[st].num_nodes, data[dt].num_nodes))

        # ---- Feature completion & (optional) projection will be done by generate_node_features() below ----
        def generate_node_features(g: HeteroData, proj_dim: int = 0) -> HeteroData:
            """
            1) 对缺失 .x 的节点类型，尝试通过"有特征节点 → 该类型"的边做均值聚合补齐特征；
            2) 若 proj_dim>0，则为所有类型用一层线性层投到同一维度（方便模型统一处理）。
               注意：这是数据级转换（固定权），若想让投影可学习，请用下文对 train_and_eval 的改动。
            """
            # 1) complete missing x via neighbor mean (单跳，从任意有 x 的源到 x 缺失的目标)
            need = [nt for nt in g.node_types if 'x' not in g[nt]]
            if len(need) > 0:
                # 对每条关系，若 src 有 x、dst 无 x，则从 src 聚合到 dst
                for (st, rel, dt) in g.edge_types:
                    if ('x' in g[st]) and ('x' not in g[dt]):
                        num_dst = g[dt].num_nodes
                        g[dt].x = _neighbor_mean_aggregate(g[st].x, g[(st, rel, dt)].edge_index, num_dst=num_dst)
                        g[dt].feat = g[dt].x

            # 2) optional: project all node features to same dim (非学习版；保持接口不变)
            if proj_dim and proj_dim > 0:
                with torch.no_grad():
                    for nt in g.node_types:
                        if 'x' in g[nt]:
                            x = g[nt].x
                            F = x.size(-1)
                            if F != proj_dim:
                                # 用固定随机矩阵（正交近似）将维度统一，避免下游模型不一致
                                W = torch.randn(F, proj_dim, device=x.device) / np.sqrt(max(1, F))
                                g[nt].x = x @ W
                                g[nt].feat = g[nt].x
            return g

        # 基于实际存在的 etype 构建以 author 为首尾的元路径（使用 etype 元组，避免 PyG DBLP 中 rel 名称均为 'to' 的歧义）
        metapaths: List[List[tuple]] = []
        has = set(data.edge_types)
        # APA: author -> paper -> author
        if ('author', 'to', 'paper') in has and ('paper', 'to', 'author') in has:
            metapaths.append([('author', 'to', 'paper'), ('paper', 'to', 'author')])
        # APCPA: author -> paper -> conference -> paper -> author
        if {('author', 'to', 'paper'), ('paper', 'to', 'conference'), ('conference', 'to', 'paper'), ('paper', 'to', 'author')}.issubset(has):
            metapaths.append([
                ('author', 'to', 'paper'),
                ('paper', 'to', 'conference'),
                ('conference', 'to', 'paper'),
                ('paper', 'to', 'author'),
            ])
        # APTPA: author -> paper -> term -> paper -> author
        if {('author', 'to', 'paper'), ('paper', 'to', 'term'), ('term', 'to', 'paper'), ('paper', 'to', 'author')}.issubset(has):
            metapaths.append([
                ('author', 'to', 'paper'),
                ('paper', 'to', 'term'),
                ('term', 'to', 'paper'),
                ('paper', 'to', 'author'),
            ])

    elif dataset == 'AMINER':
        # --- Load from PyG (downloads on first run) ---
        pyg_ds = PYG_AMINER(root='./data/pyg_aminer')
        data: HeteroData = pyg_ds[0]

        # Pick a labeled category: prefer author (research interests), else venue (categories)
        candidate_ntypes = []
        if 'author' in data.node_types and getattr(data['author'], 'y', None) is not None:
            candidate_ntypes.append('author')
        if 'venue' in data.node_types and getattr(data['venue'], 'y', None) is not None:
            candidate_ntypes.append('venue')
        assert len(candidate_ntypes) > 0, 'AMiner provides labels for a subset; author or venue should have y'

        category = 'author' if 'author' in candidate_ntypes else candidate_ntypes[0]
        data.category = category

        # Create a stratified split on labeled nodes only
        # (PyG docs: AMiner has labels for a subset)  -> we filter valid label indices (>=0) then split
        y = data[category].y
        labeled_mask = (y >= 0) if y.dtype == torch.long else torch.isfinite(y)
        # Apply RandomNodeSplit on that node type only
        splitter = RandomNodeSplit(
            split='train_rest', num_val=0.2, num_test=0.2, key='y',
            num_splits=1,  # single split
        )
        data = splitter(data)

        # Build indices restricted to labeled nodes
        idx_train = (data[category].train_mask & labeled_mask).nonzero(as_tuple=True)[0].long()
        idx_val   = (data[category].val_mask & labeled_mask).nonzero(as_tuple=True)[0].long()
        idx_test  = (data[category].test_mask & labeled_mask).nonzero(as_tuple=True)[0].long()

        # Aliases for compatibility
        if data[category].y is not None:
            data[category].label = data[category].y
        for ntype in data.node_types:
            if 'x' in data[ntype]:
                data[ntype].feat = data[ntype].x

        # Coalesce all edges (consistent with your other branches)
        for et in data.edge_types:
            st, _, dt = et
            data[et].edge_index = _coalesce_relation(data[et].edge_index, (data[st].num_nodes, data[dt].num_nodes))

        # Optional: add self-loop to any homo relation if present
        for et in data.edge_types:
            st, _, dt = et
            if st == dt:
                ei = data[et].edge_index
                data[et].edge_index = _remove_and_add_self_loops_homo(ei, data[st].num_nodes)

        # ---- Feature completion & (optional) projection done in generate_node_features() ----
        def generate_node_features(g: HeteroData, proj_dim: int = 0) -> HeteroData:
            # Same logic as DBLP branch
            need = [nt for nt in g.node_types if 'x' not in g[nt]]
            if len(need) > 0:
                for (st, rel, dt) in g.edge_types:
                    if ('x' in g[st]) and ('x' not in g[dt]):
                        num_dst = g[dt].num_nodes
                        g[dt].x = _neighbor_mean_aggregate(g[st].x, g[(st, rel, dt)].edge_index, num_dst=num_dst)
                        g[dt].feat = g[dt].x
            if proj_dim and proj_dim > 0:
                with torch.no_grad():
                    for nt in g.node_types:
                        if 'x' in g[nt]:
                            x = g[nt].x
                            F = x.size(-1)
                            if F != proj_dim:
                                W = torch.randn(F, proj_dim, device=x.device) / np.sqrt(max(1, F))
                                g[nt].x = x @ W
                                g[nt].feat = g[nt].x
            return g

        # Build metapaths from what exists
        metapaths: List[List[str]] = []
        if ('paper', 'written_by', 'author') in data.edge_types and ('author', 'writes', 'paper') in data.edge_types:
            metapaths.append(['written_by', 'writes'])
        if ('paper', 'published_in', 'venue') in data.edge_types and ('venue', 'publishes', 'paper') in data.edge_types:
            metapaths.append(['published_in', 'publishes'])

    else:  # 分支：未知数据集名
        raise ValueError(f'Unsupported dataset: {dataset}')  # 对未知名称抛出异常，提示调用方

    # 依据 return_mp 决定是否返回 metapaths（保持与原版一致）
    if return_mp:  # 若需要返回元路径
        return data, (idx_train, idx_val, idx_test), generate_node_features, metapaths  # 返回四元组

    return data, (idx_train, idx_val, idx_test), generate_node_features  # 默认返回三元组（不含元路径）
