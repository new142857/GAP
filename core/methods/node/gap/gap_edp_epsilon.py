import numpy as np
import torch
import torch.nn.functional as F
from typing import Annotated, Literal, Union
from torch_geometric.data import Data
from torch_sparse import SparseTensor, matmul
from core import console
from core.args.utils import ArgInfo
from core.methods.node import GAP
from core.privacy.algorithms import PMAE
from core.modules.base import Metrics


class EdgePrivGAPE(GAP):
    """edge-private GAP method"""

    def __init__(self,
                 num_classes,
                 epsilon: Annotated[float, ArgInfo(help='DP epsilon parameter', option='-e')],
                 delta: Annotated[Union[Literal['auto'], float],
                 ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)',
                         option='-d')] = 'auto',
                 hop_budget_weights: Annotated[
                     str,
                     ArgInfo(help='Comma-separated privacy budget weights over hops, '
                                  'e.g. "0.3,0.7". Larger weight means more privacy budget '
                                  'and smaller noise on that hop. Default: uniform.')
                 ] = 'uniform',
                 **kwargs: Annotated[dict, ArgInfo(help='extra options passed to base class', bases=[GAP])]
                 ):

        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.num_edges = None  # will be used to set delta if it is 'auto'
        self.hop_budget_weights = hop_budget_weights
        self.noise_scale = 0.0
        self.pma_mechanism = None

    def _parse_hop_budget_weights(self):
        if self.hops == 0:
            return []

        raw = self.hop_budget_weights
        if raw is None or str(raw).strip().lower() in {'', 'uniform', 'none'}:
            return [1.0 / self.hops] * self.hops

        weights = [float(x.strip()) for x in str(raw).split(',') if x.strip()]
        if len(weights) != self.hops:
            raise ValueError(
                f'hop_budget_weights length must equal hops, '
                f'got len={len(weights)}, hops={self.hops}'
            )
        if any(w <= 0 for w in weights):
            raise ValueError(f'all hop_budget_weights must be positive, got {weights}')

        s = sum(weights)
        return [w / s for w in weights]

    def _get_delta(self):
        if self.epsilon == float('inf'):
            return 0.0
        if self.delta == 'auto':
            return 1 / (10 ** len(str(self.num_edges)))
        return float(self.delta)

    def calibrate(self):
        hop_budget_weights = self._parse_hop_budget_weights()
        self.pma_mechanism = PMAE(noise_scale=0.0, hops=self.hops, hop_budget_weights=hop_budget_weights)

        delta = self._get_delta()
        console.info(f'delta = {delta}')
        console.info(f'hop budget weights = {hop_budget_weights}')

        self.noise_scale = self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)
        console.info(f'global noise scale: {self.noise_scale:.4f}')
        console.info(f'hop noise scales: {self.pma_mechanism.hop_noise_scales}')
        console.info(f'eps check: {self.pma_mechanism.get_approxDP(delta):.6f}')

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges
            self.calibrate()

        return super().fit(data, prefix=prefix)

    def compute_aggregations(self, data: Data) -> Data:
        with console.status('computing private aggregations'):
            x = F.normalize(data.x, p=2, dim=-1)
            x_list = [x]

            for hop in range(self.hops):
                x = matmul(data.adj_t, x)
                x = self.pma_mechanism(x, sensitivity=1, hop=hop)
                x = self._normalize(x)
                x_list.append(x)

            data.x = torch.stack(x_list, dim=-1)
            return data