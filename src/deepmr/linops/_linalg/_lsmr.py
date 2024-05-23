"""LSMR algorithm."""

__all__ = ["lsmr"]

import math


import torch


def lsmr(A, y, x0=None, niter=5, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, device=None):
    """
    Solve inverse problem using LSMR method.

    Parameters
    ----------
    AHA : deepmr.linop.Linop
        Forward model.
    y : torch.Tensor
        Measured data y.
    x0 : torch.Tensor, optional
        Initial guess for solution. The default is ``None``, (i.e., 0.0).
    niter : int, optional
        Number of iterations. The default is ``10``.
    damp : float, optional
        Damping factor for regularized least-squares. `lsmr` solves
        the regularized least-squares problem::

         min ||(b) - (  A   )x||
             ||(0)   (damp*I) ||_2

        where damp is a scalar.  If damp is ``None`` or ``0``, the system
        is solved without regularization. Default is ``0``.
    atol, btol : float, optional
        Stopping tolerances. `lsmr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, `lsmr` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, `lsmr` terminates when ``norm(A^H r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (default),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final ``x`` will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of ``A`` and ``b`` respectively.  For example, if the entries
        of ``A`` have 7 correct digits, set ``atol = 1e-7``. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        `lsmr` terminates if an estimate of ``cond(A)`` exceeds
        `conlim`.  For compatible systems ``Ax = b``, conlim could be
        as large as 1.0e+12 (say).  For least-squares problems,
        `conlim` should be less than 1.0e+8. If `conlim` is None, the
        default value is 1e+8.  Maximum precision can be obtained by
        setting ``atol = btol = conlim = 0``, but the number of
        iterations may then be excessive. Default is 1e8.
    device : str, optional
        Computational device.
        The default is ``None`` (infer from input).

    Returns
    -------
    output : torch.Tensor
        Reconstructed signal.

    """
    return LSMR.apply(y, A, x0, damp, niter, atol, btol, conlim, device)

# %% local utils
eps = torch.finfo(torch.float32).eps

class LSMR(torch.autograd.Function):
    @staticmethod
    def forward(y, A, x0=None, niter=10, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, device=None):
        return _lsmr_solve(A, y, x0, niter, damp, atol, btol, conlim)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        y, A, x0, niter, damp, atol, btol, conlim, device = inputs
        ctx.save_for_backward(y)
        ctx.A = A
        ctx.niter = niter
        ctx.damp = damp
        ctx.atol = atol
        ctx.atol = atol
        ctx.btol = btol
        ctx.conlim = conlim
        ctx.device = device

    @staticmethod
    def backward(ctx, dx):
        y = ctx.saved_tensors[0]
        A = ctx.AHA
        niter = ctx.niter
        damp = ctx.damp
        atol = ctx.atol
        btol = ctx.btol 
        conlim = ctx.conlim 
        device = ctx.device
        return (
            _lsmr_solve(dx, A.H, y, niter, damp, atol, btol, conlim, device),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

def _lsmr_solve(A, y, x0, maxiter, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, device=None):
    # keep original device
    idevice = _get_device(y)
    if device is None:
        device = idevice

    # put on device
    A = A.to(device)
    y = _clone(y)
    y = _to_device(y, device)
    x0 = x0.to(device)
    
    # initialize
    u = y
    normy = _norm(y)

    x = x0.clone()
    u = _diff(u, A(x))
    beta = _norm(u)

    if beta > 0:
        u = (1 / beta) * u
        v = A.H(u)
        alpha = _norm(v)
    else:
        v = torch.zeros_like(x0)
        alpha = 0

    if alpha > 0:
        v = (1 / alpha) * v

    # Initialize variables for 1st iteration.

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.clone()
    hbar = torch.zeros_like(x0)

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = (normA2)**0.5
    condA = 1
    normx = 0

    # Items for use in stopping rules, normy set earlier
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        # back to original device
        x = x.to(device)
        return x

    if normy == 0:
        x = 0 * x
        # back to original device
        x = x.to(device)
        return x

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  A@v   -  alpha*u,
        #        alpha*v  =  A'@u  -  beta*v.

        u *= -alpha
        u += A(v)
        beta = _norm(u)

        if beta > 0:
            u = _prod(u, 1 / beta)
            v *= -beta
            v += A.H(u)
            alpha = _norm(v)
            if alpha > 0:
                v *= (1 / alpha)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.

        hbar *= - (thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        h *= - (thetanew / rho)
        h += v

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = (d + (betad - taud)**2 + betadd * betadd)**0.5

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = (normA2)**0.5
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = _norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normy
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = torch.inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normy)
        rtol = btol + atol * normA * normx / normy

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if istop > 0:
            break
        
        # back to original device
        x = x.to(device)
        
    return x
    

def _sym_ortho(a, y):
    if y == 0:
        return math.sign(a), 0, 0
    elif a == 0:
        return 0, math.sign(y), abs(y)
    elif abs(y) > abs(a):
        tau = a / y
        s = math.sign(y) / (1 + tau * tau)**0.5
        c = s * tau
        r = y / s
    else:
        tau = y / a
        c = math.sign(a) / (1+tau*tau)**0.5
        s = c * tau
        r = a / c
    return c, s, r


def _clone(input):
    if isinstance(input, torch.Tensor):
        return input.clone()
    else:
        return [el.clone for el in input]


def _zeros_like(input):
    if isinstance(input, torch.Tensor):
        return torch.zeros_like(input)
    else:
        return [torch.zeros_like(el) for el in input]


def _get_device(input):
    if isinstance(input, torch.Tensor):
        return input.device
    else:
        return input[0].device()
    
    
def _to_device(input, device):
    if isinstance(input, torch.Tensor):
        return input.to(device)
    else:
        return [el.to(device) for el in input]


def _norm(input):
    if isinstance(input, torch.Tensor):
        return torch.linalg.norm(input)
    else:
        return sum([torch.linalg.norm(el)**2 for el in input])**0.5


def _diff(x, y):
    if isinstance(x, torch.Tensor):
        x - y
    else:
        return [x[n] - y[n] for n in range(len(x))]


def _prod(x, y):
    if isinstance(x, torch.Tensor):
        x * y
    else:
        return [el * y for el in x]

    
