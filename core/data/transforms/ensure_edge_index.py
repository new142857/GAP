import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from torch_sparse import SparseTensor

class EnsureEdgeIndex(BaseTransform):
    """If data.edge_index is None but data.adj_t exists, reconstruct edge_index from adj_t."""
    def __call__(self, data: Data) -> Data:
        if getattr(data, "edge_index", None) is not None:
            return data

        adj_t = getattr(data, "adj_t", None)
        if adj_t is None:
            return data

        if not isinstance(adj_t, SparseTensor):
            return data

        # adj_t is transposed adjacency used in your code, so use adj_t.t() to get original direction
        row, col, _ = adj_t.t().coo()
        data.edge_index = torch.stack([row, col], dim=0)
        return data