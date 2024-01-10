"""
Utilities related to MR simulation.
"""
import torch
from torch.func import vmap

# 1H Gyromagnetic Factor
gamma_bar = 42.577  # MHz / T
gamma = 2 * torch.pi * gamma_bar

# define helper
bdiag = vmap(torch.diag)

def matrix_exp(input: torch.Tensor):
    """
    Same as torch.matrix_exp - supports batching for vmap.
    """
    vals, vects = torch.linalg.eig(input)
    tmp = torch.einsum("...ij,...jk->...ik", vects, torch.diag(torch.exp(vals)))
    inv = torch.linalg.inv(vects)
    out = torch.einsum("...ij,...jk->...ik", tmp, inv)
    return out
