# core/data/transforms/appr_gdc.py
from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class BuildAPPRByGDC(BaseTransform):
    """
    Use PyG GDC to build a PPR/APPR-like diffusion graph via:
    diffusion (ppr) -> sparsify (topk) -> (optional) keep weights or structure-only.

    This transform rewrites edge_index (+ edge_attr).
    Must run BEFORE ToSparseTensor().
    """
    def __init__(self, *, alpha: float = 0.15, k: int = 64, weighted: bool = False, exact: bool = True):
        self.alpha = float(alpha)
        self.k = int(k)
        self.weighted = bool(weighted)
        self.exact = bool(exact)

        from torch_geometric.transforms import GDC
        self.gdc = GDC(
            diffusion_kwargs=dict(method="ppr", alpha=self.alpha),
            sparsification_kwargs=dict(method="topk", k=self.k, dim=0),
            exact=self.exact,
        )

    def __call__(self, data: Data) -> Data:
        data = self.gdc(data)  # creates edge_attr weights typically

        # 结构版：把权重全部置 1，避免下游任何“权重假设/丢权重”问题
        if not self.weighted:
            if getattr(data, "edge_attr", None) is not None:
                data.edge_attr = torch.ones_like(data.edge_attr)

        return data