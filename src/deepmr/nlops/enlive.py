# """ENLIVE/NLINV nonlinear operators."""

# __all__ = []

# import numpy as np
# import torch
# import torch.nn as nn

# class Base(nn.Module):
    
#     def __init__(self, linop, linop_grad):
#         r"""
#         Initiate the linear operator.
#         """
#         super().__init__()
        
#         # compute terms
#         self.F = linop
#         self.DF = linop_grad

#     def forward(self, x, dx):
#         return self.F(x) + dx * self.DF(x)
    
# class NLINVAdjoint(nn.Module):
    
#     def __init__(self):
#         r"""
#         Initiate the linear operator.
#         """
#         super().__init__()

#     def forward(self):
#         pass
    
    
