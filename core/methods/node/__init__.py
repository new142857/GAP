from core.methods.node.base import NodeClassification
from core.methods.node.gap.gap_inf import GAP
from core.methods.node.gap.gap_edp import EdgePrivGAP
from core.methods.node.gap.gap_ndp import NodePrivGAP
from core.methods.node.sage.sage_inf import SAGE
from core.methods.node.sage.sage_edp import EdgePrivSAGE
from core.methods.node.sage.sage_ndp import NodePrivSAGE
from core.methods.node.mlp.mlp import MLP
from core.methods.node.mlp.mlp_dp import PrivMLP
from core.methods.node.gap.gap_edp_x1xk_s import EdgePrivGAPX1XKS
from core.methods.node.gap.gap_edp_x1xk import EdgePrivGAPX1XK
from core.methods.node.gap.gap_edp_epsilon import EdgePrivGAPE

supported_methods = {
    'gap-inf':  GAP,
    'gap-edp':  EdgePrivGAP,
    'gap-edpe': EdgePrivGAPE,
    'gap-edpxs': EdgePrivGAPX1XKS,
    'gap-edpx': EdgePrivGAPX1XK,
    'gap-ndp':  NodePrivGAP,
    'sage-inf': SAGE,
    'sage-edp': EdgePrivSAGE,
    'sage-ndp': NodePrivSAGE,
    'mlp':      MLP,
    'mlp-dp':   PrivMLP
}
