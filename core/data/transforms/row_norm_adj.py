# core/data/transforms/row_norm_adj.py
from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_sparse import SparseTensor

class RowNormalizeAdjT(BaseTransform):
    """
    Row-normalize data.adj_t so that sum of weights in each row is 1.
    (data.adj_t is the transposed adjacency used in matmul(adj_t, x).)
    """
    def __call__(self, data: Data) -> Data:
        adj: SparseTensor = data.adj_t
        row, col, val = adj.coo()
        if val is None:
            # unweighted adjacency -> nothing to do
            return data

        # row sum
        row_sum = torch.zeros((data.num_nodes,), device=val.device, dtype=val.dtype)
        row_sum.scatter_add_(0, row, val.abs())
        inv = 1.0 / row_sum.clamp_min(1e-12)
        val = val * inv[row]

        data.adj_t = SparseTensor(
            row=row, col=col, value=val,
            sparse_sizes=(data.num_nodes, data.num_nodes)
        )
        return data