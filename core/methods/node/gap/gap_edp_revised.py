import numpy as np
import torch
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from core import console
from core.args.utils import ArgInfo
from core.methods.node import GAP
from core.privacy.algorithms import PMA
from core.modules.base import Metrics

import torch.nn.functional as F


class EdgePrivGAPRevised (GAP):
    """edge-private GAP method (revised)"""

    def __init__(self,
                 num_classes,
                 epsilon:       Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta:         Annotated[Union[Literal['auto'], float],
                                                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')] = 'auto',
                 sensitivity: Annotated[
                     float,
                     ArgInfo(help='L2 sensitivity for last-hop release')
                 ] = 1.0, # 这里 sensitivity 是指最后一步聚合的敏感度
                 **kwargs:      Annotated[dict,  ArgInfo(help='extra options passed to base class', bases=[GAP])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.num_edges = None  # will be used to set delta if it is 'auto'

    def calibrate(self):
        # 现在只在最终聚合结果上做一次 release
        self.pma_mechanism = PMA(noise_scale=0.0, hops=1)

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)

            self.noise_scale = self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges
            self.calibrate()

        return super().fit(data, prefix=prefix)

    def _aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        return matmul(adj_t, x) # 这里只做纯聚合

    def compute_aggregations(self, data: Data) -> Data:
        with console.status('computing aggregations'):
            x = F.normalize(data.x, p=2, dim=-1)
            x_list = [x]

            for _ in range(self.hops):
                x = self._aggregate(x, data.adj_t)
                x = self._normalize(x)
                x_list.append(x)
            # 聚合结束后统一加噪
            x = torch.stack(x_list, dim=-1)   # [N, d, K+1]
            x = self.pma_mechanism(x, sensitivity=self.sensitivity)

            data.x = x
        return data
    # 原版compute_aggregations代码：
    # def compute_aggregations(self, data: Data) -> Data:
    #     with console.status('computing aggregations'):
    #         x = F.normalize(data.x, p=2, dim=-1)
    #         x_list = [x]
    #
    #         for _ in range(self.hops):
    #             x = self._aggregate(x, data.adj_t)
    #             x = self._normalize(x)
    #             x_list.append(x)
    #
    #         data.x = torch.stack(x_list, dim=-1)
    #     return data