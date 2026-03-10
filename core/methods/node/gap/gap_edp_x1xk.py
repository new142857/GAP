import numpy as np
import torch
import torch.nn.functional as F
from typing import Annotated, Literal, Union

from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul

from core import console
from core.args.utils import ArgInfo
from core.methods.node import GAP
from core.privacy.algorithms import PMA
from core.modules.base import Metrics
from core.privacy.mechanisms.commons import GaussianMechanism


class EdgePrivGAPX1XK(GAP):
    """edge-private GAP method (last-hop only post-noise version)"""

    def __init__(self,
                 num_classes,
                 epsilon: Annotated[
                     float,
                     ArgInfo(help='DP epsilon parameter', option='-e')
                 ],
                 delta: Annotated[
                     Union[Literal['auto'], float],
                     ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)', option='-d')
                 ] = 'auto',
                 sensitivity: Annotated[
                     float,
                     ArgInfo(help='L2 sensitivity for the last-hop release')
                 ] = 1.0,
                 **kwargs: Annotated[
                     dict,
                     ArgInfo(help='extra options passed to base class', bases=[GAP])
                 ]):
        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.num_edges = None

    def calibrate(self):
        # 现在只在最终聚合结果上做一次 release
        self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)

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
        # 聚合阶段不加噪
        return matmul(adj_t, x)

    def compute_aggregations(self, data: Data) -> Data:
        with console.status('computing last-hop aggregations'):
            x = F.normalize(data.x, p=2, dim=-1)
            x_list = [x]

            for _ in range(self.hops):
                x = self._aggregate(x, data.adj_t)
                x_list.append(x)

            #只对x1到xk加噪
            x_list_noisy = [x_list[0]]  # x0 保留
            for xi in x_list[1:]:
                xi = self.pma_mechanism(xi, sensitivity=self.sensitivity)
                xi = self._normalize(xi)
                x_list_noisy.append(xi)

            data.x = torch.stack(x_list_noisy, dim=-1)
            # data.x = torch.stack([x_clean0,x_priv], dim=-1)  # [N,d,k+1]

        return data