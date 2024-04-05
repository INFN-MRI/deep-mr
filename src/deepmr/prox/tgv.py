"""Total generalized variation denoising prior."""

__all__ = ["TGVDenoiser", "tgv_denoise"]

import numpy as np
import torch
import torch.nn as nn


class TGVDenoiser(nn.Module):
    r"""
    Proximal operator of (2nd order) Total Generalised Variation operator.

    (see K. Bredies, K. Kunisch, and T. Pock, "Total generalized variation," SIAM J. Imaging Sci., 3(3), 492-526, 2010.)

    This algorithm converges to the unique image :math:`x` (and the auxiliary vector field :math:`r`) minimizing

    .. math::

        \underset{x, r}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \lambda_1 \|r\|_{1,2} + \lambda_2 \|J(Dx-r)\|_{1,F}

    where :math:`D` maps an image to its gradient field and :math:`J` maps a vector field to its Jacobian.
    For a large value of :math:`\lambda_2`, the TGV behaves like the TV.
    For a small value, it behaves like the :math:`\ell_1`-Frobenius norm of the Hessian.

    The problem is solved with an over-relaxed Chambolle-Pock algorithm (see L. Condat, "A primal-dual splitting method
    for convex optimization  involving Lipschitzian, proximable and linear composite terms", J. Optimization Theory and
    Applications, vol. 158, no. 2, pp. 460-479, 2013.

    Code (and description) adapted from Laurent Condat's matlab version (https://lcondat.github.io/software.html) and
    Daniil Smolyakov's `code <https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb>`_.

    Attributes
    ----------
    ndim : int
        Number of spatial dimensions, can be ``1``, ``2`` or ``3``.
    ths : float, optional
        Denoise threshold. The default is ``0.1``.
    axis : int, optional
        Axis over which to perform finite difference. Used only if ``ndim == 1``,
        ignored otherwise. The default is ``0``.
    trainable : bool, optional
        If ``True``, threshold value is trainable, otherwise it is not.
        The default is ``False``.
    device : str, optional
        Device on which the wavelet transform is computed.
        The default is ``None`` (infer from input).
    verbose : bool, optional
        Whether to print computation details or not. The default is ``False``.
    niter : int, optional,
        Maximum number of iterations. The default is ``1000``.
    crit : float, optional
        Convergence criterion. The default is 1e-5.
    x2 : torch.Tensor, optional
        Primary variable for warm restart. The default is ``None``.
    u2 : torch.Tensor, optional
        Dual variable for warm restart. The default is ``None``.
    r2 : torch.Tensor, optional
        Auxiliary variable for warm restart. The default is ``None``.
    offset : torch.Tensor, optional
        Offset applied to regularization input, i.e. ``output = W(input + offset)``
        Must be either a scalar or its shape must support broadcast with ``input``.

    Notes
    -----
    The regularization term :math:`\|r\|_{1,2} + \|J(Dx-r)\|_{1,F}` is implicitly normalized by its Lipschitz
    constant, i.e. :math:`\sqrt{72}`, see e.g. K. Bredies et al., "Total generalized variation," SIAM J. Imaging
    Sci., 3(3), 492-526, 2010.

    """

    def __init__(
        self,
        ndim,
        ths=0.1,
        axis=0,
        trainable=False,
        device=None,
        verbose=False,
        niter=100,
        crit=1e-5,
        x2=None,
        u2=None,
        r2=None,
        offset=None,
    ):
        super().__init__()

        if trainable:
            self.ths = nn.Parameter(ths)
        else:
            self.ths = ths

        self.denoiser = _TGVDenoiser(
            ndim=ndim,
            axis=axis,
            device=device,
            verbose=verbose,
            n_it_max=niter,
            crit=crit,
            x2=x2,
            u2=u2,
            r2=r2,
        )
        self.denoiser.device = device

        if offset is not None:
            self.offset = torch.as_tensor(offset)
        else:
            self.offset = None

    def forward(self, input):
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

        # apply offset
        if self.offset is not None:
            input = input.to(device) + self.offset.to(device)

        # reshape for computation
        input = input.reshape(-1, *ishape[-ndim:])
        if iscomplex:
            input = torch.stack((input.real, input.imag), axis=1)
            input = input.reshape(-1, *ishape[-ndim:])

        # apply denoising
        output = self.denoiser(input.to(device), self.ths).to(
            idevice
        )  # perform the denoising on the real-valued tensor

        # reshape back
        if iscomplex:
            output = (
                output[::2, ...] + 1j * output[1::2, ...]
            )  # build the denoised complex data
        output = output.reshape(ishape)

        return output.to(idevice)


def tgv_denoise(
    input,
    ndim,
    ths=0.1,
    axis=0,
    device=None,
    verbose=False,
    niter=100,
    crit=1e-5,
    x2=None,
    u2=None,
):
    r"""
    Apply Total Generalized Variation denoising.

    (see K. Bredies, K. Kunisch, and T. Pock, "Total generalized variation," SIAM J. Imaging Sci., 3(3), 492-526, 2010.)

    This algorithm converges to the unique image :math:`x` (and the auxiliary vector field :math:`r`) minimizing

    .. math::

        \underset{x, r}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \lambda_1 \|r\|_{1,2} + \lambda_2 \|J(Dx-r)\|_{1,F}

    where :math:`D` maps an image to its gradient field and :math:`J` maps a vector field to its Jacobian.
    For a large value of :math:`\lambda_2`, the TGV behaves like the TV.
    For a small value, it behaves like the :math:`\ell_1`-Frobenius norm of the Hessian.

    The problem is solved with an over-relaxed Chambolle-Pock algorithm (see L. Condat, "A primal-dual splitting method
    for convex optimization  involving Lipschitzian, proximable and linear composite terms", J. Optimization Theory and
    Applications, vol. 158, no. 2, pp. 460-479, 2013.

    Code (and description) adapted from Laurent Condat's matlab version (https://lcondat.github.io/software.html) and
    Daniil Smolyakov's `code <https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb>`_.

    Arguments
    ---------
    input : np.ndarray | torch.Tensor
        Input image of shape (..., n_ndim, ..., n_0).
    ndim : int
        Number of spatial dimensions, can be ``1``, ``2`` or ``3``.
    ths : float, optional
        Denoise threshold. Default is ``0.1``.
    axis : int, optional
        Axis over which to perform finite difference. Used only if ``ndim == 1``,
        ignored otherwise. The default is ``0``.
    trainable : bool, optional
        If ``True``, threshold value is trainable, otherwise it is not.
        The default is ``False``.
    device : str, optional
        Device on which the wavelet transform is computed.
        The default is ``None`` (infer from input).
    verbose : bool, optional
        Whether to print computation details or not. The default is ``False``.
    niter : int, optional,
        Maximum number of iterations. The default is ``1000``.
    crit : float, optional
        Convergence criterion. The default is 1e-5.
    x2 : torch.Tensor, optional
        Primary variable for warm restart. The default is ``None``.
    u2 : torch.Tensor, optional
        Dual variable for warm restart. The default is ``None``.
    r2 : torch.Tensor, optional
        Auxiliary variable for warm restart. The default is ``None``.

    Returns
    -------
    output : np.ndarray | torch.Tensor
        Denoised image of shape (..., n_ndim, ..., n_0).

    """
    # cast to numpy if required
    if isinstance(input, np.ndarray):
        isnumpy = True
        input = torch.as_tensor(input)
    else:
        isnumpy = False

    # initialize denoiser
    TGV = TGVDenoiser(ndim, ths, axis, False, device, verbose, niter, crit, x2, u2)
    output = TGV(input)

    # cast back to numpy if requried
    if isnumpy:
        output = output.numpy(force=True)

    return output


# %% local utils
class _TGVDenoiser(nn.Module):
    def __init__(
        self,
        ndim,
        axis=0,
        device=None,
        verbose=False,
        n_it_max=1000,
        crit=1e-5,
        x2=None,
        u2=None,
        r2=None,
    ):
        super().__init__()
        self.device = device
        self.ndim = ndim
        self.axis = axis

        if ndim == 1:
            self.nabla = self.nabla1
            self.nabla_adjoint = self.nabla1_adjoint
            self.epsilon = self.epsilon1
            self.epsilon_adjoint = self.epsilon1_adjoint
        elif ndim == 2:
            self.nabla = self.nabla2
            self.nabla_adjoint = self.nabla2_adjoint
            self.epsilon = self.epsilon2
            self.epsilon_adjoint = self.epsilon2_adjoint
        elif ndim == 3:
            self.nabla = self.nabla3
            self.nabla_adjoint = self.nabla3_adjoint
            self.epsilon = self.epsilon3
            self.epsilon_adjoint = self.epsilon3_adjoint

        self.verbose = verbose
        self.n_it_max = n_it_max
        self.crit = crit
        self.restart = True

        self.tau = 0.01  # > 0

        self.rho = 1.99  # in 1,2
        self.sigma = 1 / self.tau / 72

        self.x2 = x2
        self.r2 = r2
        self.u2 = u2

        self.has_converged = False

    def prox_tau_fx(self, x, y):
        return (x + self.tau * y) / (1 + self.tau)

    def prox_tau_fr(self, r, lambda1):
        left = torch.sqrt(torch.sum(r**2, axis=-1)) / (self.tau * lambda1)
        tmp = r - r / (
            torch.maximum(
                left, torch.tensor([1], device=left.device).type(left.dtype)
            ).unsqueeze(-1)
        )
        return tmp

    def prox_sigma_g_conj(self, u, lambda2):
        return u / (
            torch.maximum(
                torch.sqrt(torch.sum(u**2, axis=-1)) / lambda2,
                torch.tensor([1], device=u.device).type(u.dtype),
            ).unsqueeze(-1)
        )

    def forward(self, y, ths=None):
        restart = (
            True
            if (self.restart or self.x2 is None or self.x2.shape != y.shape)
            else False
        )

        if restart:
            self.x2 = y.clone()
            self.r2 = torch.zeros((*self.x2.shape, 2), device=self.x2.device).type(
                self.x2.dtype
            )
            self.u2 = torch.zeros((*self.x2.shape, 4), device=self.x2.device).type(
                self.x2.dtype
            )
            self.restart = False

        if ths is not None:
            lambda1 = ths * 0.1
            lambda2 = ths * 0.15

        cy = (y**2).sum() / 2
        primalcostlowerbound = 0

        for _ in range(self.n_it_max):
            x_prev = self.x2.clone()
            tmp = self.tau * self.epsilon_adjoint(self.u2)
            x = self.prox_tau_fx(self.x2 - self.nabla_adjoint(tmp), y)
            r = self.prox_tau_fr(self.r2 + tmp, lambda1)
            u = self.prox_sigma_g_conj(
                self.u2
                + self.sigma
                * self.epsilon(self.nabla(2 * x - self.x2) - (2 * r - self.r2)),
                lambda2,
            )
            self.x2 = self.x2 + self.rho * (x - self.x2)
            self.r2 = self.r2 + self.rho * (r - self.r2)
            self.u2 = self.u2 + self.rho * (u - self.u2)

            rel_err = torch.linalg.norm(
                x_prev.flatten() - self.x2.flatten()
            ) / torch.linalg.norm(self.x2.flatten() + 1e-12)

            if _ > 1 and rel_err < self.crit:
                self.has_converged = True
                if self.verbose:
                    print("TGV prox reached convergence")
                break

            if self.verbose and _ % 100 == 0:
                primalcost = (
                    torch.linalg.norm(x.flatten() - y.flatten()) ** 2
                    + lambda1 * torch.sum(torch.sqrt(torch.sum(r**2, axis=-1)))
                    + lambda2
                    * torch.sum(
                        torch.sqrt(
                            torch.sum(self.epsilon(self.nabla(x) - r) ** 2, axis=-1)
                        )
                    )
                )
                # dualcost = cy - ((y - nablaT(epsilonT(u))) ** 2).sum() / 2.0
                tmp = torch.max(
                    torch.sqrt(torch.sum(self.epsilon_adjoint(u) ** 2, axis=-1))
                )  # to check feasibility: the value will be  <= lambda1 only at convergence. Since u is not feasible, the dual cost is not reliable: the gap=primalcost-dualcost can be <0 and cannot be used as stopping criterion.
                u3 = u / torch.maximum(
                    tmp / lambda1, torch.tensor([1], device=tmp.device).type(tmp.dtype)
                )  # u3 is a scaled version of u, which is feasible. so, its dual cost is a valid, but very rough lower bound of the primal cost.
                dualcost2 = (
                    cy
                    - torch.sum((y - self.nabla_adjoint(self.epsilon_adjoint(u3))) ** 2)
                    / 2.0
                )  # we display the best value of dualcost2 computed so far.
                primalcostlowerbound = max(primalcostlowerbound, dualcost2.item())
                if self.verbose:
                    print(
                        "Iter: ",
                        _,
                        " Primal cost: ",
                        primalcost.item(),
                        " Rel err:",
                        rel_err,
                    )

            if _ == self.n_it_max - 1:
                if self.verbose:
                    print(
                        "The algorithm did not converge, stopped after "
                        + str(_ + 1)
                        + " iterations."
                    )

        return self.x2

    def nabla1(self, x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        # move selected axis upfront
        x = x.swapaxes(self.axis, -1)

        # perform finite difference
        u = torch.zeros(list(x.shape) + [1], device=x.device, dtype=x.dtype)
        u[..., :-1, 0] = u[..., :-1, 0] - x[..., :-1]
        u[..., :-1, 0] = u[..., :-1, 0] + x[..., 1:]

        # place axis back into original position
        x = x.swapaxes(self.axis, -1)
        u = u[..., 0].swapaxes(self.axis, -1)[..., None]

        return u

    def nabla1_adjoint(self, x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        # move selected axis upfront
        x = x[..., 0].swapaxes(self.axis, -1)[..., None]

        # perform finite difference
        u = torch.zeros(
            x.shape[:-1], device=x.device, dtype=x.dtype
        )  # note that we just reversed left and right sides of each line to obtain the transposed operator        u[..., :-1, 0] = u[..., :-1, 0] - x[..., :-1]
        u[..., :-1] = u[..., :-1] - x[..., :-1, 0]
        u[..., 1:] = u[..., 1:] + x[..., :-1, 0]

        # place axis back into original position
        x = x[..., 0].swapaxes(self.axis, -1)[..., None]
        u = u.swapaxes(self.axis, -1)

        return u

    @staticmethod
    def nabla2(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        u = torch.zeros(list(x.shape) + [2], device=x.device, dtype=x.dtype)
        u[..., :-1, :, 0] = u[..., :-1, :, 0] - x[..., :, :-1]
        u[..., :-1, :, 0] = u[..., :-1, :, 0] + x[..., :, 1:]
        u[..., :, :-1, 1] = u[..., :, :-1, 1] - x[..., :-1, :]
        u[..., :, :-1, 1] = u[..., :, :-1, 1] + x[..., 1:, :]
        return u

    @staticmethod
    def nabla2_adjoint(x):
        r"""
        Applies the adjoint of the finite difference operator.
        """
        u = torch.zeros(
            x.shape[:-1], device=x.device, dtype=x.dtype
        )  # note that we just reversed left and right sides of each line to obtain the transposed operator
        u[..., :, :-1] = u[..., :, :-1] - x[..., :-1, :, 0]
        u[..., :, 1:] = u[..., :, 1:] + x[..., :-1, :, 0]
        u[..., :-1, :] = u[..., :-1, :] - x[..., :, :-1, 1]
        u[..., 1:, :] = u[..., 1:, :] + x[..., :, :-1, 1]
        return u

    @staticmethod
    def nabla3(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        u = torch.zeros(list(x.shape) + [3], device=x.device, dtype=x.dtype)
        u[..., :-1, :, :, 0] = u[..., :-1, :, :, 0] - x[..., :, :, :-1]
        u[..., :-1, :, :, 0] = u[..., :-1, :, :, 0] + x[..., :, :, 1:]
        u[..., :, :-1, :, 1] = u[..., :, :-1, :, 1] - x[..., :, :-1, :]
        u[..., :, :-1, :, 1] = u[..., :, :-1, :, 1] + x[..., :, 1:, :]
        u[..., :, :, :-1, 2] = u[..., :, :, :-1, 2] - x[..., :-1, :, :]
        u[..., :, :, :-1, 2] = u[..., :, :, :-1, 2] + x[..., 1:, :, :]

        return u

    @staticmethod
    def nabla3_adjoint(x):
        r"""
        Applies the adjoint of the finite difference operator.
        """
        u = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        u[..., :, :, :-1] = u[..., :, :, :-1] - x[..., :-1, :, :, 0]
        u[..., :, :, 1:] = u[..., :, :, 1:] + x[..., :-1, :, :, 0]
        u[..., :, :-1, :] = u[..., :, :-1, :] - x[..., :, :-1, :, 1]
        u[..., :, 1:, :] = u[..., :, 1:, :] + x[..., :, :-1, :, 1]
        u[..., :-1, :, :] = u[..., :-1, :, :] - x[..., :, :, :-1, 2]
        u[..., 1:, :, :] = u[..., 1:, :, :] + x[..., :, :, :-1, 2]

        return u

    def epsilon1(self, I):
        r"""
        Applies the jacobian of a vector field.
        """
        # move selected axis upfront
        I = I[..., 0].swapaxes(self.axis, -1)[..., None]

        # perform finite difference
        G = torch.zeros(list(I.shape[:-1]) + [1], device=I.device, dtype=I.dtype)
        G[..., 1:, :, 0] = G[..., 1:, :, 0] - I[..., :-1, :, 0]  # xdx
        G[..., 0] = G[..., 0] + I[..., 0]

        # place axis back into original position
        I = I[..., 0].swapaxes(self.axis, -1)[..., None]
        G = G[..., 0].swapaxes(self.axis, -1)[..., None]

        return G

    def epsilon1_adjoint(self, G):
        r"""
        Applies the adjoint of the jacobian of a vector field.
        """
        # move selected axis upfront
        G = G[..., 0].swapaxes(self.axis, -1)[..., None]

        # perform finite difference
        I = torch.zeros(list(G.shape[:-1]) + [1], device=G.device, dtype=G.dtype)
        I[..., :-1, :, 0] = I[..., :-1, :, 0] - G[..., 1:, :, 0]  # xdx
        I[..., 0] = I[..., 0] + G[..., 0]
        I[..., :-1, 0] = I[..., :-1, 0] - G[..., 1:, 1]  # xdy

        # place axis back into original position
        I = I[..., 0].swapaxes(self.axis, -1)[..., None]
        G = G[..., 0].swapaxes(self.axis, -1)[..., None]

        return I

    @staticmethod
    def epsilon2(I):
        r"""
        Applies the jacobian of a vector field.
        """
        G = torch.zeros(list(I.shape[:-1]) + [4], device=I.device, dtype=I.dtype)
        G[..., 1:, :, 0] = G[..., 1:, :, 0] - I[..., :-1, :, 0]  # xdx
        G[..., 0] = G[..., 0] + I[..., 0]
        G[..., 1:, 1] = G[..., 1:, 1] - I[..., :-1, 0]  # xdy
        G[..., 1:, 1] = G[..., 1:, 1] + I[..., 1:, 0]
        G[..., 1:, 2] = G[..., 1:, 2] - I[..., :-1, 1]  # ydx
        G[..., 2] = G[..., 2] + I[..., 1]
        G[..., :-1, :, 3] = G[..., :-1, :, 3] - I[..., :-1, :, 1]  # ydy
        G[..., :-1, :, 3] = G[..., :-1, :, 3] + I[..., 1:, :, 1]

        return G

    @staticmethod
    def epsilon2_adjoint(G):
        r"""
        Applies the adjoint of the jacobian of a vector field.
        """
        I = torch.zeros(list(G.shape[:-1]) + [2], device=G.device, dtype=G.dtype)
        I[..., :-1, :, 0] = I[..., :-1, :, 0] - G[..., 1:, :, 0]  # xdx
        I[..., 0] = I[..., 0] + G[..., 0]
        I[..., :-1, 0] = I[..., :-1, 0] - G[..., 1:, 1]  # xdy
        I[..., 1:, 0] = I[..., 1:, 0] + G[..., 1:, 1]
        I[..., :-1, 1] = I[..., :-1, 1] - G[..., 1:, 2]  # ydx
        I[..., 1] = I[..., 1] + G[..., 2]
        I[..., :-1, :, 1] = I[..., :-1, :, 1] - G[..., :-1, :, 3]  # ydy
        I[..., 1:, :, 1] = I[..., 1:, :, 1] + G[..., :-1, :, 3]

        return I

    @staticmethod
    def epsilon3(I):
        r"""
        Applies the jacobian of a vector field.
        """
        G = torch.zeros(list(I.shape[:-1]) + [9], device=I.device, dtype=I.dtype)
        G[..., 1:, :, :, 0] = G[..., 1:, :, :, 0] - I[..., :-1, :, :, 0]  # xdx
        G[..., 0] = G[..., 0] + I[..., 0]
        G[..., 1:, :, 1, 1] = G[..., 1:, :, 1, 1] - I[..., :-1, :, :, 1]  # xdy
        G[..., 1:, :, 1, 1] = G[..., 1:, :, 1, 1] + I[..., 1:, :, :, 1]
        G[..., 1:, 1, :, 2] = G[..., 1:, 1, :, 2] - I[..., :-1, :, :, 2]  # xdz
        G[..., 2] = G[..., 2] + I[..., 1, :, :, 2]
        G[..., :-1, :, :, 3] = G[..., :-1, :, :, 3] - I[..., :-1, :, :, 3]  # ydx
        G[..., :-1, :, :, 3] = G[..., :-1, :, :, 3] + I[..., 1:, :, :, 3]
        G[..., 1:, :, 1, 4] = G[..., 1:, :, 1, 4] - I[..., :-1, :, :, 4]  # ydy
        G[..., 1:, :, 1, 4] = G[..., 1:, :, 1, 4] + I[..., 1:, :, :, 4]
        G[..., 1:, 1, :, 5] = G[..., 1:, 1, :, 5] - I[..., :-1, :, :, 5]  # ydz
        G[..., 3] = G[..., 3] + I[..., 1, :, :, 5]
        G[..., :-1, 1, :, 6] = G[..., :-1, 1, :, 6] - I[..., :-1, :, :, 6]  # zdx
        G[..., 4] = G[..., 4] + I[..., 1, :, :, 6]
        G[..., 1:, :, :, 7] = G[..., 1:, :, :, 7] - I[..., :-1, :, :, 7]  # zdy
        G[..., 5] = G[..., 5] + I[..., 1:, :, :, 7]
        G[..., :-1, :, 1, 8] = G[..., :-1, :, 1, 8] - I[..., :-1, :, 1, 8]  # zdz
        G[..., 6] = G[..., 6] + I[..., 1:, :, 1, 8]

        return G

    @staticmethod
    def epsilon3_adjoint(G):
        r"""
        Applies the adjoint of the jacobian of a vector field.
        """
        I = torch.zeros(list(G.shape[:-1]) + [3], device=G.device, dtype=G.dtype)
        I[..., :-1, :, :, 0] = I[..., :-1, :, :, 0] - G[..., 1:, :, :, 0]  # xdx
        I[..., 0] = I[..., 0] + G[..., 0]
        I[..., :-1, :, 0] = I[..., :-1, :, 0] - G[..., 1:, :, 1]  # xdy
        I[..., 1:, :, 0] = I[..., 1:, :, 0] + G[..., 1:, :, 1]
        I[..., :-1, 0] = I[..., :-1, 0] - G[..., 1:, :, 2]  # xdz
        I[..., 2] = I[..., 2] + G[..., 2]
        I[..., :-1, :, :, 1] = I[..., :-1, :, :, 1] - G[..., 1:, :, :, 3]  # ydx
        I[..., 1:, :, :, 1] = I[..., 1:, :, :, 1] + G[..., :-1, :, :, 3]
        I[..., :-1, :, 1] = I[..., :-1, :, 1] - G[..., 1:, :, 4]  # ydy
        I[..., 1:, :, 1] = I[..., 1:, :, 1] + G[..., :-1, :, 4]
        I[..., :-1, 1] = I[..., :-1, 1] - G[..., 1:, :, 5]  # ydz
        I[..., 2] = I[..., 2] + G[..., :-1, :, 5]
        I[..., :, :-1, :, 2] = I[..., :, :-1, :, 2] - G[..., :, :-1, :, 6]  # zdx
        I[..., :, 1:, :, 2] = I[..., :, 1:, :, 2] + G[..., :, :-1, :, 6]
        I[..., :, :-1, 2] = I[..., :, :-1, 2] - G[..., :, 1:, 7]  # zdy
        I[..., :, 1:, 2] = I[..., :, 1:, 2] + G[..., :, 1:, 7]
        I[..., :-1, :-1, 2] = I[..., :-1, :-1, 2] - G[..., 1:, 1:, 8]  # zdz
        I[..., 1:, 1:, 2] = I[..., 1:, 1:, 2] + G[..., 1:, 1:, 8]

        return I
