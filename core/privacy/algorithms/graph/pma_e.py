import torch
from autodp.mechanism_zoo import ExactGaussianMechanism
from autodp.transformer_zoo import ComposeGaussian
from core import console
from core.privacy.mechanisms.commons import GaussianMechanism, InfMechanism, ZeroMechanism
from core.privacy.mechanisms.composed import ComposedGaussianMechanism
from core.privacy.mechanisms.noisy import NoisyMechanism
import math
from typing import Optional, Sequence

class PMAE(NoisyMechanism):
    """
    Private Message Aggregation supporting per-hop privacy budget allocation.

    hop_budget_weights:
        w_1, ..., w_K, sum w_k = 1, w_k > 0
        Larger w_k means hop k gets more privacy budget -> smaller noise.
    """

    def __init__(self, noise_scale: float, hops: int, hop_budget_weights=None):
        super().__init__(noise_scale)

        if hops < 0:
            raise ValueError(f'hops must be non-negative, got {hops}')

        # 1) 解析 hop 权重
        if hops == 0:
            weights = []
        elif hop_budget_weights is None:
            weights = [1.0 / hops] * hops
        else:
            weights = [float(w) for w in hop_budget_weights]
            if len(weights) != hops:
                raise ValueError(
                    f'len(hop_budget_weights) must equal hops, '
                    f'got len={len(weights)}, hops={hops}'
                )
            if any(w <= 0 for w in weights):
                raise ValueError(f'all hop_budget_weights must be positive, got {weights}')
            s = sum(weights)
            if s <= 0:
                raise ValueError(f'sum(hop_budget_weights) must be positive, got {s}')
            weights = [w / s for w in weights]

        self.hops = hops
        self.hop_budget_weights = weights

        self.name = 'PMA'
        self.params = {
            'noise_scale': noise_scale,
            'hops': hops,
            'hop_budget_weights': self.hop_budget_weights,
        }

        # 2) 根据预算权重生成每一跳实际 noise scale
        #    sigma_k = noise_scale / sqrt(hops * w_k)
        #    这样总 RDP 与原始 PMA 保持同一语义
        self.hop_noise_scales = []
        self.gm_list = []

        if hops > 0:
            for w in self.hop_budget_weights:
                sigma_k = noise_scale / math.sqrt(hops * w) if noise_scale > 0 else 0.0
                self.hop_noise_scales.append(sigma_k)
                self.gm_list.append(GaussianMechanism(noise_scale=sigma_k))

        # 3) 构造 accountant 用的组合机制
        if hops == 0:
            mech = ZeroMechanism()
            self.params['noise_scale'] = 0.0
        elif noise_scale == 0.0:
            mech = InfMechanism()
        else:
            exact_list = [ExactGaussianMechanism(sigma=s) for s in self.hop_noise_scales]
            mech = ComposeGaussian()(exact_list, [1] * hops)

        self.set_all_representation(mech)

    def __call__(self, x: torch.Tensor, sensitivity: float, hop: Optional[int] = None) -> torch.Tensor:
        if self.hops == 0:
            return x

        if hop is None:
            raise ValueError('PMA.__call__ requires hop index when hops > 0')

        if hop < 0 or hop >= self.hops:
            raise ValueError(f'hop index out of range: {hop}, hops={self.hops}')
        # 临时调试
        console.info(f'apply hop={hop}, sigma={self.gm_list[hop].params["noise_scale"]}')

        return self.gm_list[hop](x, sensitivity=sensitivity)
