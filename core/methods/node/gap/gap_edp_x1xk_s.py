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


class EdgePrivGAPX1XKS(GAP):
    """edge-private GAP method (post-noise on x1...xk, keep x0 clean)"""

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
                 auto_sensitivity: Annotated[
                     bool,
                     ArgInfo(help='automatically estimate per-hop local sensitivity by enumerating edge deletions')
                 ] = True,
                 sensitivity_undirected: Annotated[
                     bool,
                     ArgInfo(help='if True, remove both (u,v) and (v,u) when estimating local sensitivity')
                 ] = False,
                 sensitivity_max_edges: Annotated[
                     int,
                     ArgInfo(
                         help='max number of stored edges to enumerate when estimating local sensitivity; <=0 means all')
                 ] = 2000,
                 sensitivity_verbose_every: Annotated[
                     int,
                     ArgInfo(help='print progress every N checked edges when estimating local sensitivity')
                 ] = 200,
                 **kwargs: Annotated[
                     dict,
                     ArgInfo(help='extra options passed to base class', bases=[GAP])
                 ]):
        super().__init__(num_classes, **kwargs)
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.num_edges = None

        self.auto_sensitivity = auto_sensitivity
        self.sensitivity_undirected = sensitivity_undirected
        self.sensitivity_max_edges = sensitivity_max_edges
        self.sensitivity_verbose_every = sensitivity_verbose_every
        self.sensitivity_per_hop = [float(sensitivity)] * self.hops

    @torch.no_grad()
    def _compute_clean_hop_queries(self, data: Data, adj_t: SparseTensor = None):
        """
        返回 [q1, q2, ..., qK]
        其中
            z0 = normalize(x)
            qk = A z_{k-1}
            zk = normalize(qk)

        注意：这里返回的是“每一跳加噪前的干净发布量” qk，
        与当前 compute_aggregations 中 x_list[1:] 的语义一致。
        """
        if adj_t is None:
            adj_t = data.adj_t

        z = F.normalize(data.x, p=2, dim=-1)
        q_list = []

        for _ in range(self.hops):
            q = self._aggregate(z, adj_t)  # q_k
            q_list.append(q)
            z = self._normalize(q)  # z_k

        return q_list

    @torch.no_grad()
    def _remove_one_edge_from_coo(self,
                                  row: torch.Tensor,
                                  col: torch.Tensor,
                                  val: torch.Tensor,
                                  sparse_sizes,
                                  edge_idx: int,
                                  undirected: bool = False) -> SparseTensor:
        mask = torch.ones(row.numel(), dtype=torch.bool, device=row.device)

        r = row[edge_idx]
        c = col[edge_idx]
        mask[edge_idx] = False

        if undirected and r != c:
            rev_mask = (row == c) & (col == r)
            mask = mask & (~rev_mask)

        kwargs = {
            'row': row[mask],
            'col': col[mask],
            'sparse_sizes': sparse_sizes,
        }
        if val is not None:
            kwargs['value'] = val[mask]

        return SparseTensor(**kwargs)

    @torch.no_grad()
    def estimate_local_sensitivity_per_hop(self,
                                           data: Data,
                                           undirected: bool = False,
                                           max_edges: int = 0,
                                           verbose_every: int = 1000):
        """
        枚举删边，计算 deletion-only 的 per-hop local sensitivity:

            Delta_k_local(G) = max_{e in E} || q_k(G) - q_k(G-e) ||_F
        其中 hop1 不枚举，直接用理论值：
            - undirected=False: Delta_1 = 1
        参数
        ----
        undirected:
            若 True，则删除一条存储边 (u,v) 时，同时删除 (v,u)
        max_edges:
            只检查前 max_edges 条存储边；<=0 表示检查全部
        verbose_every:
            每检查多少条边打印一次进度；<=0 表示不打印

        返回
        ----
        deltas_cpu: shape [hops]
            每个 hop 的 local sensitivity
        """
        assert self.hops >= 1, 'hops must be >= 1'

        with console.status('estimating per-hop local sensitivity by edge enumeration'):
            base_q_list = self._compute_clean_hop_queries(data, data.adj_t)

            row, col, _ = data.adj_t.coo()
            num_stored_edges = row.numel()

            if max_edges is None or max_edges <= 0 or max_edges >= num_stored_edges:
                edge_indices = torch.arange(num_stored_edges, device=row.device)
            else:
                perm = torch.randperm(num_stored_edges, device=row.device)
                edge_indices = perm[:max_edges]

            deltas = torch.zeros(self.hops, dtype=torch.float32, device=data.x.device)
            best_edge_idx = torch.full((self.hops,), -1, dtype=torch.long, device=row.device)

            total = edge_indices.numel()
            console.info(
                f'estimating local sensitivity on {total}/{num_stored_edges} stored edges '
                f'(undirected={undirected})'
            )

            row, col, val = data.adj_t.coo()
            sparse_sizes = data.adj_t.sparse_sizes()

            for t, eidx in enumerate(edge_indices.tolist(), start=1):
                adj_drop = self._remove_one_edge_from_coo(
                    row=row,
                    col=col,
                    val=val,
                    sparse_sizes=sparse_sizes,
                    edge_idx=eidx,
                    undirected=undirected,
                )
                drop_q_list = self._compute_clean_hop_queries(data, adj_drop)

                for k in range(1, self.hops):
                    diff = base_q_list[k] - drop_q_list[k]
                    d = diff.reshape(-1).norm(p=2)  # Frobenius norm

                    if d > deltas[k]:
                        deltas[k] = d
                        best_edge_idx[k] = eidx

                if verbose_every and verbose_every > 0 and (t % verbose_every == 0 or t == total):
                    current_vals = ['hop1=1.000000']
                    for k in range(1, self.hops):
                        current_vals.append(f'hop{k + 1}={deltas[k].item():.6f}')
                    current = ', '.join(current_vals)
                    console.info(f'checked {t}/{total} edges, current max: {current}')

            deltas[0] = 1.0
            best_edge_idx[0] = -1

            deltas_cpu = deltas.detach().cpu()
            best_edge_idx_cpu = best_edge_idx.detach().cpu()

            console.info('per-hop local sensitivity (deletion-only):')
            for k in range(self.hops):
                eidx = int(best_edge_idx_cpu[k].item())
                if k == 0:
                    console.info(
                        f'  hop 1: {deltas_cpu[0].item():.6f} '
                        f'(fixed theoretical value, undirected={undirected})'
                    )
                elif eidx >= 0:
                    u = int(row[eidx].item())
                    v = int(col[eidx].item())
                    console.info(
                        f'  hop {k + 1}: {deltas_cpu[k].item():.6f} '
                        f'(attained by removing stored edge idx={eidx}, edge=({u}, {v}))'
                    )
                else:
                    console.info(f'  hop {k + 1}: {deltas_cpu[k].item():.6f}')

            console.info(f'global choice for self.sensitivity = max_k Delta_k = {deltas_cpu.max().item():.6f}\n')

        return deltas_cpu

    @torch.no_grad()
    def _get_sensitivity_for_hop(self, hop_idx: int) -> float:
        """
        hop_idx: 0-based, 对应 x1...xk 的第几个 hop
        """
        if self.sensitivity_per_hop is None:
            return float(self.sensitivity)

        if isinstance(self.sensitivity_per_hop, torch.Tensor):
            return float(self.sensitivity_per_hop[hop_idx].item())

        return float(self.sensitivity_per_hop[hop_idx])

    def calibrate(self):
        # 对 x1~xk 分别加噪
        self.pma_mechanism = PMA(noise_scale=0.0, hops=self.hops)

        with console.status('calibrating noise to privacy budget'):
            if self.delta == 'auto':
                delta = 0.0 if np.isinf(self.epsilon) else 1. / (10 ** len(str(self.num_edges)))
                console.info('delta = %.0e' % delta)

            self.noise_scale = self.pma_mechanism.calibrate(eps=self.epsilon, delta=delta)
            console.info(f'noise scale: {self.noise_scale:.4f}\n')

            if self.sensitivity_per_hop is not None:
                msg = ', '.join([
                    f'hop{i + 1}: sens={float(s):.6f}, std={self.noise_scale * float(s):.6f}'
                    for i, s in enumerate(self.sensitivity_per_hop)
                ])
                console.info(f'per-hop Gaussian std: {msg}\n')
            else:
                console.info(f'unified sensitivity = {self.sensitivity:.6f}, '
                             f'std = {self.noise_scale * self.sensitivity:.6f}\n')

    def fit(self, data: Data, prefix: str = '') -> Metrics:
        if data.num_edges != self.num_edges:
            self.num_edges = data.num_edges

            if self.auto_sensitivity:
                deltas = self.estimate_local_sensitivity_per_hop(
                    data,
                    undirected=self.sensitivity_undirected,
                    max_edges=self.sensitivity_max_edges,
                    verbose_every=self.sensitivity_verbose_every,
                )
                self.sensitivity_per_hop = [float(x) for x in deltas.tolist()]
                self.sensitivity = max(self.sensitivity_per_hop)  # 仅保留给兼容逻辑/打印用

                sens_msg = ', '.join([f'x{i + 1}={s:.6f}' for i, s in enumerate(self.sensitivity_per_hop)])
                console.info(f'auto-set per-hop sensitivity: {sens_msg}')
                console.info(f'max sensitivity (for compatibility) = {self.sensitivity:.6f}')

            else:
                # 没开自动估计时，所有 hop 默认共用同一个 sensitivity
                self.sensitivity_per_hop = [float(self.sensitivity)] * self.hops

            self.calibrate()

        return super().fit(data, prefix=prefix)

    def _aggregate(self, x: torch.Tensor, adj_t: SparseTensor) -> torch.Tensor:
        # 聚合阶段不加噪
        return matmul(adj_t, x)

    def compute_aggregations(self, data: Data) -> Data:
        with console.status('computing aggregations'):
            x = F.normalize(data.x, p=2, dim=-1)
            x_list = [x]
            x_list_noisy = [x_list[0]]
            for _ in range(self.hops):
                x = self._aggregate(x, data.adj_t)
                x_list.append(x)
                x = self._normalize(x)
                # ====== 加入聚合系数 α_k ======
                # alpha_k = 1.0 / self.hops  # 举例，可自定义
                # x = alpha_k * x
                # =================================

            #只对x1到xk加噪，但每个 hop 用自己的 sensitivity
            for hop_idx, xi in enumerate(x_list[1:]):
                sens_i = self._get_sensitivity_for_hop(hop_idx)
                xi = self.pma_mechanism(xi, sensitivity=sens_i)
                xi = self._normalize(xi)
                x_list_noisy.append(xi)

            data.x = torch.stack(x_list_noisy, dim=-1)
            # data.x = torch.stack([x_clean0,x_priv], dim=-1)  # [N,d,k+1]

        return data