import torch
import torch_scatter
import torch_sparse
import torch_cluster

print("torch:", torch.__version__)
print("scatter ok")
print("sparse ok")
print("cluster ok")
print("spline ok")

# 再来一个最小 scatter 测试
x = torch.tensor([1., 2., 3., 4.])
idx = torch.tensor([0, 0, 1, 1])
out = torch_scatter.scatter_mean(x, idx, dim=0)
print("scatter result:", out)