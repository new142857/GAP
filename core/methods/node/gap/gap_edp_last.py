import numpy as np
import torch
import torch.nn.functional as F
from typing import Annotated, Literal, Union

from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul

from core import console
from core.args.utils import ArgInfo
from core.methods.node import GAP
from core.modules.base import Metrics
from core.privacy.mechanisms.commons import GaussianMechanism


class EdgePrivGAPLastHop(GAP):
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
        # 这里只做一次最终发布，直接用 GaussianMechanism 更清楚
        self.gaussian_mechanism = GaussianMechanism(noise_scale=0.0)

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)
            else:
                delta = self.delta

            self.noise_scale = self.gaussian_mechanism.calibrate(eps=self.epsilon, delta=delta)
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
                x = self._normalize(x)
                x_list.append(x)

            # 只对最后一个 hop 加噪
            x_last = x_list[-1]
            x_last = self.gaussian_mechanism(x_last, sensitivity=self.sensitivity)
            x_last = self._normalize(x_last)
            x_list[-1] = x_last
            data.x = torch.stack(x_list, dim=-1)   # [N, d, K+1]

        return data