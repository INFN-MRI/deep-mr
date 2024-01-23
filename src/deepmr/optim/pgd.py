"""Proximal Gradient Descent optimizer."""

__all__ = ["pgd"]

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from deepinv.optim.data_fidelity import L2
    from deepinv.optim.optimizers import optim_builder
    from deepinv.optim.prior import PnP


def pgd(
    input,
    encoding,
    denoiser,
    lamda=0.01,
    stepsize=1.0,
    accelerate=True,
    max_iter=20,
    verbose=False,
):
    r"""
    Proximal Gradient Descent.
    
    Proximal Gradient Descent (PGD) algorithm for minimising :math:`\lambda f(x) + g(x)`.

    The iteration is given by

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k} &= x_k - \lambda \gamma \nabla f(x_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma g}(u_k),
        \end{aligned}
        \end{equation*}


    where :math:`\gamma` is a stepsize that should satisfy :math:`\lambda \gamma \leq 2/\operatorname{Lip}(\|\nabla f\|)`.


    Parameters
    ----------
    input : torch.Tensor
        Input data of shape ().
    encoding : deepinv.Physics
        Encoding operator.
    denoiser : deepinv.Model
        Denoiser to be used as an image prior.
    lamda : float, optional
        Regularization strength. The default is 0.01.
    stepsize : float, optional
        Gradient step size. The default is 1.0.
    accelerate : bool, optional
        Toggle Anderson acceleration. The default is True.
    max_iter : int, optional
        Maximum number of iterations. The default is 20.
    verbose : bool, optional
        Verbosity flag. The default is False.

    Returns
    -------
    output: torch.Tensor
        Output data of shape ().
        
    References
    ----------
        Beck, A., & Teboulle, M. (2009).
        A fast iterative shrinkage-thresholding algorithm
        for linear inverse problems.
        SIAM journal on imaging sciences, 2(1), 183-202.
        
        Vien V. Mai and Mikael Johansson. (2020). 
        Anderson acceleration of proximal gradient methods. 
        In Proceedings of the 37th International Conference on Machine Learning (ICML'20), 
        Vol. 119. JMLR.org, Article 614, 66206629.

    """
    # Select the data fidelity term
    data_fidelity = L2()

    # Instantiate the algorithm class to solve the problem.
    optimalgo = optim_builder(
        iteration="PGD",
        prior=PnP(denoiser=denoiser),
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        verbose=verbose,
        anderson_acceleration=accelerate,
        params_algo={"stepsize": stepsize, "lambda": lamda},
    ).to(input.device)

    # Run the algorithm
    return optimalgo(input, encoding)
