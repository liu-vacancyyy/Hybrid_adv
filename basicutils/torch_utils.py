import torch
from torch import linalg

def torch_rand_float(lower, upper,shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower

def compute_grad(q1, q2, a1, a2):
    diff_q = q1 - q2
    diff_a = a1 - a2
    diff_a[diff_a == 0] = 1e-3
    grad = diff_q / diff_a
    return grad

def l2_spatial_norm(x):
    return linalg.norm(x, 2, dim=1)

def l2_spatial_project(x, distance):
    x_shape = x.size()
    norm = l2_spatial_norm(x)
    change_id = (norm > distance).nonzero(as_tuple=False).flatten()

    norms = norm.repeat_interleave(x_shape[1]).reshape(x_shape[0],x_shape[1])  # 不优雅啊
    x[change_id] = (x[change_id] / norms[change_id]) * distance

    return x