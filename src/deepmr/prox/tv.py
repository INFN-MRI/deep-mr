"""Total variation denoising prior."""

__all__ = ["TVPrior", "tv_denoise"]

import numpy as np
import torch

from deepinv.optim.prior import PnP


def TVPrior(ndim, device=None, verbose=False, niter=100, crit=1e-5, x2=None, u2=None):
    r"""
    Proximal operator of the isotropic Total Variation operator.

    This algorithm converges to the unique image :math:`x` that is the solution of

    .. math::

        \underset{x}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \gamma \|Dx\|_{1,2},

    where :math:`D` maps an image to its gradient field.

    The problem is solved with an over-relaxed Chambolle-Pock algorithm (see L. Condat, "A primal-dual splitting method
    for convex optimization  involving Lipschitzian, proximable and linear composite terms", J. Optimization Theory and
    Applications, vol. 158, no. 2, pp. 460-479, 2013.

    Code (and description) adapted from ``deepinv``, in turn adapted from
    Laurent Condat's matlab version (https://lcondat.github.io/software.html) and
    Daniil Smolyakov's `code <https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb>`_.

    This algorithm is implemented with warm restart, i.e. the primary and dual variables are kept in memory
    between calls to the forward method. This speeds up the computation when using this class in an iterative algorithm.

    Attributes
    ---------
    ndim : int
        Number of spatial dimensions, can be either ``2`` or ``3``.
    device : str, optional 
        Device on which the wavelet transform is computed. Default is ``None``.
    verbose : bool, optional
        Whether to print computation details or not. Default: ``False``.
    niter : int, optional,
        Maximum number of iterations. Default: ``1000``.
    crit : float, optional 
        Convergence criterion. Default: 1e-5.
    x2 : torch.Tensor, optional
        Primary variable for warm restart. Default: ``None``.
    u2 : torch.Tensor, optional
        Dual variable for warm restart. Default: ``None``.

    Notes
    -----
    The regularization term :math:`\|Dx\|_{1,2}` is implicitly normalized by its Lipschitz constant, i.e.
    :math:`\sqrt{8}`, see e.g. A. Beck and M. Teboulle, "Fast gradient-based algorithms for constrained total
    variation image denoising and deblurring problems", IEEE T. on Image Processing. 18(11), 2419-2434, 2009.

    """  
    return PnP(denoiser=ComplexTVDenoiser(ndim, device, verbose, niter, crit, x2, u2))


def tv_denoise(input, ndim, ths=0.1, device=None, verbose=False, niter=100, crit=1e-5, x2=None, u2=None):
    r"""
    Apply isotropic Total Variation denoising.

    This algorithm converges to the unique image :math:`x` that is the solution of

    .. math::

        \underset{x}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \gamma \|Dx\|_{1,2},

    where :math:`D` maps an image to its gradient field.

    The problem is solved with an over-relaxed Chambolle-Pock algorithm (see L. Condat, "A primal-dual splitting method
    for convex optimization  involving Lipschitzian, proximable and linear composite terms", J. Optimization Theory and
    Applications, vol. 158, no. 2, pp. 460-479, 2013.

    Code (and description) adapted from ``deepinv``, in turn adapted from
    Laurent Condat's matlab version (https://lcondat.github.io/software.html) and
    Daniil Smolyakov's `code <https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb>`_.

    This algorithm is implemented with warm restart, i.e. the primary and dual variables are kept in memory
    between calls to the forward method. This speeds up the computation when using this class in an iterative algorithm.

    Arguments
    ---------
    input : np.ndarray | torch.Tensor
        Input image of shape (..., n_ndim, ..., n_0).
    ndim : int
        Number of spatial dimensions, can be either ``2`` or ``3``. 
    ths : float, optional
        Denoise threshold. Default is ``0.1``.
    device : str, optional 
        Device on which the wavelet transform is computed. Default is ``None``.
    verbose : bool, optional
        Whether to print computation details or not. Default: ``False``.
    niter : int, optional,
        Maximum number of iterations. Default: ``1000``.
    crit : float, optional 
        Convergence criterion. Default: 1e-5.
    x2 : torch.Tensor, optional
        Primary variable for warm restart. Default: ``None``.
    u2 : torch.Tensor, optional
        Dual variable for warm restart. Default: ``None``.

    Notes
    -----
    The regularization term :math:`\|Dx\|_{1,2}` is implicitly normalized by its Lipschitz constant, i.e.
    :math:`\sqrt{8}`, see e.g. A. Beck and M. Teboulle, "Fast gradient-based algorithms for constrained total
    variation image denoising and deblurring problems", IEEE T. on Image Processing. 18(11), 2419-2434, 2009.
        
    Returns
    -------
    output : np.ndarray | torch.Tensor
        Denoised image of shape (..., n_ndim, ..., n_0).
    
    """
    TV = ComplexTVDenoiser(ndim, device, verbose, niter, crit, x2, u2)
    return TV(input, ths)


# %% local utils
class ComplexTVDenoiser(torch.nn.Module):
    def __init__(self, ndim, device=None, verbose=False, n_it_max=1000, crit=1e-5, x2=None, u2=None):
        super().__init__()
        self.denoiser = _TVDenoiser(ndim, device, verbose, n_it_max, crit, x2, u2)
    
    def forward(self, input, ths):
    
        # cast to numpy if required
        if isinstance(input, np.ndarray):
            isnumpy = True
            input = torch.as_tensor(input)
        else:
            isnumpy = False
            
        # get complex
        if torch.is_complex(input):
            iscomplex = True
        else:
            iscomplex = False
            
        # default device
        idevice = input.device
        if self.denoiser.device is None:
            device = idevice
        else:
            device = self.denoiser.device
            
        # get input shape
        ndim = self.denoiser.ndim
        ishape = input.shape
        
        # reshape for computation
        input = input.reshape(-1, *ishape[-ndim:])
        if iscomplex:
            input = torch.stack((input.real, input.imag), axis=1)
            input = input.reshape(-1, *ishape[-ndim:])
        
        # apply denoising
        output = self.denoiser(input[:, None, ...].to(device), ths).to(idevice)  # perform the denoising on the real-valued tensor
        
        # reshape back
        if iscomplex:
            output = output[::2, ...] + 1j * output[1::2, ...]  # build the denoised complex data
        output = output.reshape(ishape)
        
        # cast back to numpy if requried
        if isnumpy:
            output = output.numpy(force=True)
        
        return output
    
class _TVDenoiser(torch.nn.Module):
    def __init__(self, ndim, device=None, verbose=False, n_it_max=1000, crit=1e-5, x2=None, u2=None):
        super().__init__()
        self.device = device
        self.ndim = ndim
        
        if ndim == 2:
            self.nabla = self.nabla2
            self.nabla_adjoint = self.nabla2_adjoint
        elif ndim == 3:
            self.nabla = self.nabla3
            self.nabla_adjoint = self.nabla3_adjoint

        self.verbose = verbose
        self.n_it_max = n_it_max
        self.crit = crit
        self.restart = True

        self.tau = 0.01  # > 0

        self.rho = 1.99  # in 1,2
        self.sigma = 1 / self.tau / 8

        self.x2 = x2
        self.u2 = u2

        self.has_converged = False

    def prox_tau_fx(self, x, y):
        return (x + self.tau * y) / (1 + self.tau)

    def prox_sigma_g_conj(self, u, lambda2):
        return u / (
            torch.maximum(
                torch.sqrt(torch.sum(u**2, axis=-1)) / lambda2,
                torch.tensor([1], device=u.device).type(u.dtype),
            ).unsqueeze(-1)
        )

    def forward(self, y, ths=None):
        r"""
        Computes the proximity operator of the TV norm.

        :param torch.Tensor y: Noisy image.
        :param float, torch.Tensor ths: Regularization parameter :math:`\gamma`.
        :return: Denoised image.
        """
        restart = (
            True
            if (self.restart or self.x2 is None or self.x2.shape != y.shape)
            else False
        )

        if restart:
            self.x2 = y.clone()
            self.u2 = torch.zeros((*self.x2.shape, 2), device=self.x2.device).type(
                self.x2.dtype
            )
            self.restart = False

        if ths is not None:
            lambd = ths

        for _ in range(self.n_it_max):
            x_prev = self.x2.clone()

            x = self.prox_tau_fx(self.x2 - self.tau * self.nabla_adjoint(self.u2), y)
            u = self.prox_sigma_g_conj(
                self.u2 + self.sigma * self.nabla(2 * x - self.x2), lambd
            )
            self.x2 = self.x2 + self.rho * (x - self.x2)
            self.u2 = self.u2 + self.rho * (u - self.u2)

            rel_err = torch.linalg.norm(
                x_prev.flatten() - self.x2.flatten()
            ) / torch.linalg.norm(self.x2.flatten() + 1e-12)

            if _ > 1 and rel_err < self.crit:
                if self.verbose:
                    print("TV prox reached convergence")
                break

        return self.x2

    @staticmethod
    def nabla2(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        b, c, h, w = x.shape
        u = torch.zeros((b, c, h, w, 2), device=x.device).type(x.dtype)
        u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] - x[:, :, :-1]
        u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] + x[:, :, 1:]
        u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] - x[..., :-1]
        u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] + x[..., 1:]
        return u

    @staticmethod
    def nabla2_adjoint(x):
        r"""
        Applies the adjoint of the finite difference operator.
        """
        b, c, h, w = x.shape[:-1]
        u = torch.zeros((b, c, h, w), device=x.device).type(
            x.dtype
        )  # note that we just reversed left and right sides of each line to obtain the transposed operator
        u[:, :, :-1] = u[:, :, :-1] - x[:, :, :-1, :, 0]
        u[:, :, 1:] = u[:, :, 1:] + x[:, :, :-1, :, 0]
        u[..., :-1] = u[..., :-1] - x[..., :-1, 1]
        u[..., 1:] = u[..., 1:] + x[..., :-1, 1]
        return u
    
    @staticmethod
    def nabla3(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        b, c, d, h, w = x.shape
        u = torch.zeros((b, c, d, h, w, 3), device=x.device).type(x.dtype)
        u[:, :, :-1, :, :, 0] = u[:, :, :-1, :, :, 0] - x[:, :, :-1]
        u[:, :, :-1, :, :, 0] = u[:, :, :-1, :, :, 0] + x[:, :, 1:]        
        u[:, :, :, :-1, :, 1] = u[:, :, :, :-1, :, 1] - x[:, :, :, :-1]
        u[:, :, :, :-1, :, 1] = u[:, :, :, :-1, :, 1] + x[:, :, :, 1:]        
        u[:, :, :, :, :-1, 2] = u[:, :, :, :, :-1, 2] - x[:, :, :, :, :-1]
        u[:, :, :, :, :-1, 2] = u[:, :, :, :, :-1, 2] + x[:, :, :, :, 1:]
        
        return u

    @staticmethod
    def nabla3_adjoint(x):
        r"""
        Applies the adjoint of the finite difference operator.
        """
        b, c, d, h, w = x.shape
        u = torch.zeros((b, c, d, h, w), device=x.device).type(x.dtype)        
        u[:, :, :-1, :, 0] = u[:, :, :-1, :, 0] - x[:, :, :-1]
        u[:, :, 1:, :, 0] = u[:, :, 1:, :, 0] + x[:, :, :-1]        
        u[:, :, :, :-1, 1] = u[:, :, :, :-1, 1] - x[:, :, :, :-1]
        u[:, :, :, 1:, 1] = u[:, :, :, 1:, 1] + x[:, :, :, :-1]
        u[:, :, :, :, :-1, 2] = u[:, :, :, :, :-1, 2] - x[:, :, :, :, :-1]
        u[:, :, :, :, 1:, 2] = u[:, :, :, :, 1:, 2] + x[:, :, :, :, :-1]
        
        return u