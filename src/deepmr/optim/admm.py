"""Alternating direction method of multipliers optimizer."""

__all__ = ["admm"]

from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import PnP


def admm(
    input,
    encoding,
    denoiser,
    lamda=0.01,
    stepsize=1.0,
    beta=1,
    max_iter=20,
    verbose=False,
):
    r"""
    Alternating direction method of multipliers.

    Alternating Direction Method of Multipliers (ADMM) algorithm for
    minimising :math:`\lambda f(x) + g(x)`.

    The iteration is given by (`see this paper <https://www.nowpublishers.com/article/Details/MAL-016>`_):

    .. math::
        \begin{equation*}
        \begin{aligned}
        u_{k+1} &= \operatorname{prox}_{\gamma \lambda f}(x_k - z_k) \\
        x_{k+1} &= \operatorname{prox}_{\gamma g}(u_{k+1} + z_k) \\
        z_{k+1} &= z_k + \beta (u_{k+1} - x_{k+1})
        \end{aligned}
        \end{equation*}

    where :math:`\gamma>0` is a stepsize and :math:`\beta>0` is a relaxation parameter.

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
    beta : float, optional
        Relaxation parameter. The default is 1.
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
    Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato and Jonathan Eckstein (2011), 
    Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers, 
    Foundations and Trends® in Machine Learning: Vol. 3: No. 1, pp 1-122. 
    http://dx.doi.org/10.1561/2200000016

    """
    # Select the data fidelity term
    data_fidelity = L2()

    # Instantiate the algorithm class to solve the problem.
    optimalgo = optim_builder(
        iteration="ADMM",
        prior=PnP(denoiser=denoiser),
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        verbose=verbose,
        params_algo={"stepsize": stepsize, "lambda": lamda, "beta": beta},
    ).to(input.device)

    # Run the algorithm
    return optimalgo(input, encoding)
