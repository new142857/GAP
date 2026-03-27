import torch
from autodp.mechanism_zoo import ExactGaussianMechanism
from autodp.transformer_zoo import ComposeGaussian
from core import console
from core.privacy.mechanisms.commons import GaussianMechanism, InfMechanism, ZeroMechanism
from core.privacy.mechanisms.composed import ComposedGaussianMechanism
from core.privacy.mechanisms.noisy import NoisyMechanism
import math
from typing import Optional, Sequence

class PMAE2(NoisyMechanism):
    def __init__(self, noise_scale: float, hops: int, hop_noise_factors=None):
        super().__init__(noise_scale)
        self.name = 'PMAE2'

        self.hops = hops
        self.hop_noise_factors = hop_noise_factors
        self.hop_noise_scales = []
        self.gm_list = []
        self.params = {
            'noise_scale': noise_scale,
            'hops': hops,
            'hop_noise_factors': hop_noise_factors,
        }

        # 1) 检查参数
        if hops < 0:
            raise ValueError(f'hops must be non-negative, got {hops}')

        if hops == 0:
            factors = []
        elif hop_noise_factors is None:
            factors = [1.0] * hops
        else:
            factors = [float(v) for v in hop_noise_factors]
            if len(factors) != hops:
                raise ValueError(
                    f'len(hop_noise_factors) must equal hops, got len={len(factors)}, hops={hops}'
                )
            if any(v <= 0 for v in factors):
                raise ValueError(f'all hop_noise_factors must be positive, got {factors}')



        # 2) 根据预算权重生成每一跳实际 noise scale : sigma_k = noise_scale * factor_k
        if hops > 0:
            for factor_k in self.hop_noise_factors:
                sigma_k = noise_scale * factor_k if noise_scale > 0 else 0.0
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
        # console.info(f'apply hop={hop}, sigma={self.gm_list[hop].params["noise_scale"]}')

        return self.gm_list[hop](x, sensitivity=sensitivity)
