import numpy as np
import torch
import torch.nn.functional as F
from typing import Annotated, Literal, Union

from torch_geometric.data import Data
from torch_sparse import matmul

from core import console
from core.args.utils import ArgInfo
from core.methods.node import GAP
from core.privacy.algorithms import PMAE2
from core.modules.base import Metrics


class EdgePrivGAPE2(GAP):
    """edge-private GAP method with hop-wise noise schedule"""

    def __init__(self,
                 num_classes,
                 epsilon: Annotated[
                     float,
                     ArgInfo(help='DP epsilon parameter', option='-e')
                 ],
                 delta: Annotated[
                     Union[Literal['auto'], float],
                     ArgInfo(help='DP delta parameter (if "auto", sets a proper value based on data size)',
                             option='-d')
                 ] = 'auto',
                 hop_noise_mode: Annotated[
                     str,
                     ArgInfo(help='hop noise schedule mode: uniform | exp | softmax | poly | sigmoid | manual')
                 ] = 'uniform',
                 hop_noise_lambda: Annotated[
                     float,
                     ArgInfo(help='lambda in exp schedule: factor_k = exp(k * lambda)')
                 ] = 0.0,
                 hop_noise_temp: Annotated[
                     float,
                     ArgInfo(help='temperature for softmax schedule')
                 ] = 3.0,
                 hop_noise_poly_power: Annotated[
                     float,
                     ArgInfo(help='power p for polynomial schedule: factor_k = k^p')
                 ] = 1.2,
                 hop_noise_sigmoid_alpha: Annotated[
                     float,
                     ArgInfo(help='alpha for sigmoid schedule')
                 ] = 1.0,
                 hop_noise_factors: Annotated[
                     str,
                     ArgInfo(help='manual hop noise factors, e.g. "1,2,4"')
                 ] = '',
                 **kwargs: Annotated[
                     dict,
                     ArgInfo(help='extra options passed to base class', bases=[GAP])
                 ]):
        super().__init__(num_classes, **kwargs)

        self.epsilon = epsilon
        self.delta = delta
        self.num_edges = None  # used for auto delta

        self.hop_noise_mode = hop_noise_mode
        self.hop_noise_lambda = hop_noise_lambda
        self.hop_noise_temp = hop_noise_temp
        self.hop_noise_poly_power = hop_noise_poly_power
        self.hop_noise_sigmoid_alpha = hop_noise_sigmoid_alpha
        self.hop_noise_factors = hop_noise_factors

        self.noise_scale = 0.0
        self.pma_mechanism = None

    def _build_hop_noise_factors(self):
        if self.hops == 0:
            return []

        mode = str(self.hop_noise_mode).strip().lower()
        x = np.arange(1, self.hops + 1, dtype=float)

        if mode in {'uniform', '', 'none'}:
            return [1.0] * self.hops

        if mode == 'exp':
            lam = float(self.hop_noise_lambda)
            # factor_k = exp(k * lambda), k = 1..K
            z = np.exp(x * lam)
            return z.tolist()

        if mode == 'softmax':
            T = float(self.hop_noise_temp)
            if T <= 0:
                raise ValueError(f'hop_noise_temp must be positive, got {T}')
            z = x / T
            z = z - np.max(z)  # 数值稳定
            z = np.exp(z)
            z = z / z.sum()
            return z.tolist()

        if mode == 'poly':
            p = float(self.hop_noise_poly_power)
            z = np.power(x, p)
            return z.tolist()

        if mode == 'sigmoid':
            alpha = float(self.hop_noise_sigmoid_alpha)
            center = (self.hops + 1) / 2.0
            z = 1.0 / (1.0 + np.exp(-alpha * (x - center)))
            return z.tolist()

        if mode == 'manual':
            raw = str(self.hop_noise_factors).strip()
            if not raw:
                raise ValueError('hop_noise_factors must be provided when hop_noise_mode=manual')
            factors = [float(v.strip()) for v in raw.split(',') if v.strip()]
            if len(factors) != self.hops:
                raise ValueError(
                    f'hop_noise_factors length must equal hops, got len={len(factors)}, hops={self.hops}'
                )
            if any(v <= 0 for v in factors):
                raise ValueError(f'all hop_noise_factors must be positive, got {factors}')
            return factors

        raise ValueError(f'unknown hop_noise_mode: {self.hop_noise_mode}')

    def _get_delta(self):
        if self.epsilon == float('inf'):
            return 0.0
        if self.delta == 'auto':
            return 1 / (10 ** len(str(self.num_edges)))
        return float(self.delta)

    def calibrate(self):
        hop_noise_factors = self._build_hop_noise_factors()

        self.pma_mechanism = PMAE2(
            noise_scale=0.0,
            hops=self.hops,
            hop_noise_factors=hop_noise_factors
        )

        delta = self._get_delta()
        console.info(f'delta = {delta}')
        console.info(f'hop noise mode = {self.hop_noise_mode}')
        console.info(f'hop noise factors = {hop_noise_factors}')

        self.noise_scale = self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)

        console.info(f'global noise scale: {self.noise_scale:.4f}')
        console.info(f'hop noise scales: {self.pma_mechanism.hop_noise_scales}')
        console.info(f'eps check: {self.pma_mechanism.get_approxDP(delta):.6f}')

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges
            self.calibrate()

        metrics = super().fit(data, prefix=prefix)
        metrics["noise_scale"] = float(self.noise_scale)
        return metrics

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