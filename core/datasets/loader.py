import os
from functools import partial
from typing import Annotated
import torch
from core import console
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
from torch_geometric.transforms import Compose, ToSparseTensor, RandomNodeSplit
from core.args.utils import ArgInfo
from core.data.transforms import FilterClassByCount
from core.data.transforms import RemoveSelfLoops
from core.data.transforms import RemoveIsolatedNodes
from core.datasets import Facebook
from core.datasets import Amazon
from core.utils import dict2table

from core.data.transforms.appr_gdc import BuildAPPRByGDC
from core.data.transforms.row_norm_adj import RowNormalizeAdjT
from core.data.transforms.ensure_edge_index import EnsureEdgeIndex

class DatasetLoader:
    supported_datasets = {
        'reddit': partial(Reddit, 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=10000, remove_unlabeled=True)
            ])
        ),
        'amazon': partial(Amazon, 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=100000, remove_unlabeled=True)
            ])
        ),
        'facebook': partial(Facebook, name='UIllinois20', target='year', 
            transform=Compose([
                RandomNodeSplit(num_val=0.1, num_test=0.15), 
                FilterClassByCount(min_count=1000, remove_unlabeled=True)
            ])
        ),
    }

    def __init__(self,
                 dataset:    Annotated[str, ArgInfo(help='name of the dataset', choices=supported_datasets)] = 'facebook',
                 data_dir:   Annotated[str, ArgInfo(help='directory to store the dataset')] = './datasets',
                 # 新增这些：不影响 DP 代码
                 graph: Annotated[str, ArgInfo(help='graph used for propagation', choices=['orig', 'appr'])] = 'orig',
                 appr_alpha: Annotated[float, ArgInfo(help='APPR/PPR alpha')] = 0.15,
                 appr_k: Annotated[int, ArgInfo(help='APPR top-k neighbors')] = 64,
                 appr_weighted: Annotated[bool, ArgInfo(help='use APPR weights in adj_t')] = False,
                 appr_exact: Annotated[bool, ArgInfo(help='use exact diffusion in GDC')] = True,
                 ):

        self.name = dataset
        self.data_dir = data_dir
        # 新增这些：不影响 DP 代码
        self.graph = graph
        self.appr_alpha = appr_alpha
        self.appr_k = appr_k
        self.appr_weighted = appr_weighted
        self.appr_exact = appr_exact

    def load(self, verbose=False) -> Data:
        data = self.supported_datasets[self.name](root=os.path.join(self.data_dir, self.name))[0]
        data = Compose([RemoveSelfLoops(), RemoveIsolatedNodes(), ToSparseTensor()])(data)

        tfms = [EnsureEdgeIndex(),RemoveSelfLoops(), RemoveIsolatedNodes()]

        if self.graph == 'appr':
            # 先把 edge_index 替换成 APPR topk 图（结构版 or 加权版）
            tfms.append(BuildAPPRByGDC(alpha=self.appr_alpha, k=self.appr_k,
                                       weighted=self.appr_weighted, exact=self.appr_exact))

            # 关键：加权版必须把 edge_attr 变成 adj_t 的 value
            if self.appr_weighted:
                tfms.append(ToSparseTensor(attr='edge_attr'))
                # tfms.append(RowNormalizeAdjT())  # 推荐加上，尽量不破坏 sensitivity=1 假设
            else:
                tfms.append(ToSparseTensor())
        else:
            tfms.append(ToSparseTensor())

        data = Compose(tfms)(data)

        if verbose:
            self.print_stats(data)

        return data

    def print_stats(self, data: Data):
        nodes_degree: torch.Tensor = data.adj_t.sum(dim=1)
        baseline: float = (data.y[data.test_mask].unique(return_counts=True)[1].max().item() * 100 / data.test_mask.sum().item())
        train_ratio: float = data.train_mask.sum().item() / data.num_nodes * 100
        val_ratio: float = data.val_mask.sum().item() / data.num_nodes * 100
        test_ratio: float = data.test_mask.sum().item() / data.num_nodes * 100

        stat = {
            'nodes': f'{data.num_nodes:,}',
            'edges': f'{data.num_edges:,}',
            'features': f'{data.num_features:,}',
            'classes': f'{int(data.y.max() + 1)}',
            'mean degree': f'{nodes_degree.mean():.2f}',
            'median degree': f'{nodes_degree.median()}',
            'train/val/test (%)': f'{train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}',
            'baseline acc (%)': f'{baseline:.2f}'
        }

        table = dict2table(stat, num_cols=2, title=f'dataset: [yellow]{self.name}[/yellow]')
        console.info(table)
        console.print()
