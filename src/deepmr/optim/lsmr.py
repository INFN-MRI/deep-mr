# """
# LSMR iteration.

# Code adapted from https://github.com/cai4cai/torchsparsegradutils/blob/main/torchsparsegradutils/utils/lsmr.py,
# originally from https://github.com/rfeinman/pytorch-minimize/blob/master/torchmin/lstsq/lsmr.py
# Code modified from scipy.sparse.linalg.lsmr

# Copyright (C) 2010 David Fong and Michael Saunders

# """

# __all__ = ["lsmr_solve", "LSMRStep"]

# import numpy as np
# import torch

# import torch.nn as nn

# from .. import linops as _linops

# @torch.no_grad()
# def lsmr_solve(
#     input,
#     AHA,
#     niter=10,
#     device=None,
#     tol=1e-4,
#     lamda=0.0,
#     ndim=None,
# ):
#     """
#     Solve inverse problem using LSMR method.

#     Parameters
#     ----------
#     input : np.ndarray | torch.Tensor
#         Signal to be reconstructed. Assume it is the adjoint AH of measurement
#         operator A applied to the measured data y (i.e., input = AHy).
#     AHA : Callable
#         Normal operator AHA = AH * A.
#     niter : int, optional
#         Number of iterations. The default is ``10``.
#     device : str, optional
#         Computational device.
#         The default is ``None`` (infer from input).
#     tol : float, optional
#         Stopping condition. The default is ``1e-4``.
#     lamda : float, optional
#         Tikhonov regularization strength. The default is ``0.0``.
#     ndim : int, optional
#         Number of spatial dimensions of the problem.
#         It is used to infer the batch axes. If ``AHA`` is a ``deepmr.linop.Linop``
#         operator, this is inferred from ``AHA.ndim`` and ``ndim`` is ignored.


#     Returns
#     -------
#     output : np.ndarray | torch.Tensor
#         Reconstructed signal.

#     """
#     # cast to numpy if required
#     if isinstance(input, np.ndarray):
#         isnumpy = True
#         input = torch.as_tensor(input)
#     else:
#         isnumpy = False

#     # keep original device
#     idevice = input.device
#     if device is None:
#         device = idevice

#     # put on device
#     input = input.to(device)
#     if isinstance(AHA, _linops.Linop):
#         AHA = AHA.to(device)

#     # assume input is AH(y), i.e., adjoint of measurement operator
#     # applied on measured data
#     AHy = input.clone()

#     # add Tikhonov regularization
#     if lamda != 0.0:
#         if isinstance(AHA, _linops.Linop):
#             _AHA = AHA + lamda * _linops.Identity(AHA.ndim)
#         else:
#             _AHA = lambda x: AHA(x) + lamda * x
#     else:
#         _AHA = AHA

#     # initialize algorithm
#     LSMR = LSMRStep(_AHA, AHy, ndim)

#     # initialize
#     input = 0 * input

#     # run algorithm
#     for n in range(niter):
#         output = LSMR(input)
#         if LSMR.check_convergence():
#             break
#         input = output.clone()

#     # back to original device
#     output = output.to(device)

#     # cast back to numpy if requried
#     if isnumpy:
#         output = output.numpy(force=True)

#     return output


# class LSMRStep(nn.Module):
#     """
#     A PyTorch module implementing the LSMR iterative solver for least-squares problems.

#     Attributes
#     ----------
#     A : torch.Tensor
#         Matrix A in the linear system Ax = b.
#     b : torch.Tensor
#         Vector b in the linear system Ax = b.
#     Armat : callable
#         Function to compute the adjoint of A.
#     n : int
#         Dimension of the problem.
#     damp : float
#         Damping factor for regularized least-squares.
#     atol : float
#         Absolute stopping tolerance.
#     btol : float
#         Relative stopping tolerance.
#     conlim : float
#         Maximum allowed condition number of A.
#     maxiter : int
#         Maximum number of iterations.
#     x0 : torch.Tensor
#         Initial guess for the solution.
#     check_nonzero : bool
#         Flag to check if beta is nonzero.

#     """

#     def __init__(self, A, b, Armat=None, n=None, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=None, x0=None, check_nonzero=True, device=None):
#         super().__init__()

#         b = torch.as_tensor(b)

#         # get device device
#         if device is None:
#             device = b.device

#         # put on device
#         b = b.to(device)

#         if isinstance(A, _linops.Linop):
#             self.A = A.to(device)
#         elif callable(A) is False:
#             A = torch.as_tensor(A, dtype=b.dtype, device=device)
#             self.A = lambda x : A @ x
#         else:
#             self.A = A

#         # estimate self-adjoint
#         if Armat is None:
#             if isinstance(A, _linops.Linop):
#                 self.Armat = A.H
#             elif callable(A) is False:
#                 self.Armat = lambda x : A.conj().T @ x
#             else:
#                 raise RuntimeError("If A is neither a deepr.linops.Linop nor a np.ndarray / torch.Tensor, please provide Armat as function handle.")
#         else:
#             if isinstance(Armat, _linops.Linop):
#                 self.Armat = Armat.to(device)
#             elif callable(A) is False:
#                 Armat = torch.as_tensor(Armat, dtype=b.dtype, device=device)
#                 self.Armat = lambda x : Armat @ x
#             else:
#                 self.Armat = Armat

#         self.A = A
#         self.b = b
#         self.Armat = Armat
#         self.n = n
#         self.damp = damp
#         self.atol = atol
#         self.btol = btol
#         self.conlim = conlim
#         self.maxiter = maxiter
#         self.x0 = x0
#         self.check_nonzero = check_nonzero

#         # Initialize any necessary variables
#         self.u = self.b.clone()
#         self.normb = self.b.norm()
#         if self.x0 is None:
#             self.x = self.b.new_zeros(self.n)
#             self.beta = self.normb.clone()
#         else:
#             self.x = torch.atleast_1d(self.x0).clone()
#             self.u.sub_(self.A(self.x))
#             self.beta = self.u.norm()

#         if self.beta > 0:
#             self.u.div_(self.beta)
#             self.v = self.Armat(self.u)
#             self.alpha = self.v.norm()
#         else:
#             self.v = self.b.new_zeros(self.n)
#             self.alpha = self.b.new_tensor(0)

#         self.v = torch.where(self.alpha > 0, self.v / self.alpha, self.v)

#         # Initialize variables for 1st iteration.
#         self.zetabar = self.alpha * self.beta
#         self.alphabar = self.alpha.clone()
#         self.rho = self.b.new_tensor(1)
#         self.rhobar = self.b.new_tensor(1)
#         self.cbar = self.b.new_tensor(1)
#         self.sbar = self.b.new_tensor(0)

#         self.h = self.v.clone()
#         self.hbar = self.b.new_zeros(self.n)

#         self.betadd = self.beta.clone()
#         self.betad = self.b.new_tensor(0)
#         self.rhodold = self.b.new_tensor(1)
#         self.tautildeold = self.b.new_tensor(0)
#         self.thetatilde = self.b.new_tensor(0)
#         self.zeta = self.b.new_tensor(0)
#         self.d = self.b.new_tensor(0)

#         self.normA2 = self.alpha.square()
#         self.maxrbar = self.b.new_tensor(0)
#         self.minrbar = self.b.new_tensor(0.99 * torch.finfo(self.b.dtype).max)
#         self.normA = self.normA2.sqrt()
#         self.condA = self.b.new_tensor(1)
#         self.normx = self.b.new_tensor(0)
#         self.normr = self.beta.clone()
#         self.normar = self.alpha * self.beta

#     def forward(self, x):
#         # Perform the next step of the bidiagonalization to obtain the
#         # next  beta, u, alpha, v.  These satisfy the relations
#         #         beta*u  =  a*v   -  alpha*u,
#         #        alpha*v  =  A'*u  -  beta*v.

#         self.u.mul_(-self.alpha).add_(self.A(self.v))
#         torch.norm(self.u, out=self.beta)

#         if (not self.check_nonzero) or self.beta > 0:
#             self.u.div_(self.beta)
#             self.v.mul_(-self.beta).add_(self.Armat(self.u))
#             torch.norm(self.v, out=self.alpha)
#             self.v = torch.where(self.alpha > 0, self.v / self.alpha, self.v)

#         # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

#         _sym_ortho(self.alphabar, self.damp, out=(self.chat, self.shat, self.alphahat))

#         # Use a plane rotation (Q_i) to turn B_i to R_i

#         self.rhoold = self.rho.clone()
#         _sym_ortho(self.alphahat, self.beta, out=(self.c, self.s, self.rho))
#         self.thetanew = torch.mul(self.s, self.alpha)
#         torch.mul(self.c, self.alpha, out=self.alphabar)

#         # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

#         self.rhobarold = self.rhobar.clone()
#         self.zetaold = self.zeta.clone()
#         self.thetabar = self.sbar * self.rho
#         self.rhotemp = self.cbar * self.rho
#         _sym_ortho(self.cbar * self.rho, self.thetanew, out=(self.cbar, self.sbar, self.rhobar))
#         torch.mul(self.cbar, self.zetabar, out=self.zeta)
#         self.zetabar.mul_(-self.sbar)

#         # Update h, h_hat, x.

#         self.hbar.mul_(-self.thetabar * self.rho).div_(self.rhoold * self.rhobarold)
#         self.hbar.add_(self.h)
#         x.addcdiv_(self.zeta * self.hbar, self.rho * self.rhobar)
#         self.h.mul_(-self.thetanew).div_(self.rho)
#         self.h.add_(self.v)

#         # Estimate of ||r||.

#         # Apply rotation Qhat_{k,2k+1}.
#         torch.mul(self.chat, self.betadd, out=self.betaacute)
#         torch.mul(-self.shat, self.betadd, out=self.betacheck)

#         # Apply rotation Q_{k,k+1}.
#         torch.mul(self.c, self.betaacute, out=self.betahat)
#         torch.mul(-self.s, self.betaacute, out=self.betadd)

#         # Apply rotation Qtilde_{k-1}.
#         self.thetatildeold = self.thetatilde.clone()
#         _sym_ortho(self.rhodold, self.thetabar, out=(self.ctildeold, self.stildeold, self.rhotildeold))
#         torch.mul(self.stildeold, self.rhobar, out=self.thetatilde)
#         torch.mul(self.ctildeold, self.rhobar, out=self.rhodold)
#         self.betad.mul_(-self.stildeold).addcmul_(self.ctildeold, self.betahat)

#         self.tautildeold.mul_(-self.thetatildeold).add_(self.zetaold).div_(self.rhotildeold)
#         torch.div(self.zeta - self.thetatilde * self.tautildeold, self.rhodold, out=self.taud)
#         self.d.addcmul_(self.betacheck, self.betacheck)
#         torch.sqrt(self.d + (self.betad - self.taud).square() + self.betadd.square(), out=self.normr)

#         # Estimate ||A||.
#         self.normA2.addcmul_(self.beta, self.beta)
#         torch.sqrt(self.normA2, out=self.normA)
#         self.normA2.addcmul_(self.alpha, self.alpha)

#         # Estimate cond(A).
#         torch.max(self.maxrbar, self.rhobarold, out=self.maxrbar)

#         return x

#     def check_convergence(self):
#         # Compute norms for convergence testing.
#         torch.abs(self.zetabar, out=self.normar)
#         torch.norm(self.x, out=self.normx)
#         torch.div(torch.max(self.maxrbar, self.rhotemp), torch.min(self.minrbar, self.rhotemp), out=self.condA)

#         test1 = self.normr / self.normb
#         test2 = self.normar / (self.normA * self.normr + torch.finfo(self.b.dtype).eps)
#         test3 = 1 / (self.condA + torch.finfo(self.b.dtype).eps)
#         t1 = test1 / (1 + self.normA * self.normx / self.normb)
#         rtol = self.btol + self.atol * self.normA * self.normx / self.normb

#         # The first 3 tests guard against extremely small values of
#         # atol, btol or ctol.  (The user may have set any or all of
#         # the parameters atol, btol, conlim  to 0.)
#         # The effect is equivalent to the normAl tests using
#         # atol = eps,  btol = eps,  conlim = 1/eps.
#         # The second 3 tests allow for tolerances set by the user.
#         stop = (
#             (1 + test3 <= 1)
#             | (1 + test2 <= 1)
#             | (1 + t1 <= 1)
#             | (test3 <= self.conlim)
#             | (test2 <= self.atol)
#             | (test1 <= rtol)
#         )

#         return stop

# # %% local utils
# def _sym_ortho(a, b, out):
#     torch.hypot(a, b, out=out[2])
#     torch.div(a, out[2], out=out[0])
#     torch.div(b, out[2], out=out[1])
#     return out
