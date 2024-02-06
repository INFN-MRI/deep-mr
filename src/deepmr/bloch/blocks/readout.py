"""Common readout blocks."""

__all__ = ["ExcPulse", "FSEStep", "bSSFPStep", "SSFPFidStep", "SSFPEchoStep"]

from .. import ops


def ExcPulse(states, B1, rf_props):
    """
    RF operator.

    Parameters
    ----------
    states : dict
        EPG states matrix.
    B1 : float | complex
        Flip angle scaling of shape ``(...,nmodes)``.
    rf_props : dict
        Extra RF parameters.

    Returns
    -------
    RFPulse : deepmr.bloch.Operator 
        RF pulse rotation operator.

    """
    # parse rf properties
    if rf_props is None:
        rf_props = {}

    # parse
    device = states["F"].device
    nlocs = states["F"].shape[-3]

    return ops.RFPulse(device, nlocs=nlocs, B1=B1, name="Excitation Pulse", **rf_props)


def FSEStep(
    states,
    ESP,
    T1,
    T2,
    weight=None,
    k=None,
    chemshift=None,
    D=None,
    v=None,
    grad_props=None,
):
    """
    (Unbalanced) SSFP propagator.

    Consists of rfpulse (omitted) - free precession - spoiling gradient.

    Parameters
    ----------
    states : dict
        EPG states matrix.
    ESP : float
        Echo Spacing in ``[ms]``.
    T1 : torch.Tensor 
        T1 relaxation time of shape ``(..., npools) in ``[ms]``.
    T2 : torch.Tensor 
        T2 relaxation time of shape ``(..., npools) in ``[ms]``.
    weight : torch.Tensor  
        Pool relative fraction.
    k : torch.Tensor 
        Chemical exchange matrix ``(...., npools, npools)`` in ``[Hz]``.
    chemshift : torch.Tensor  
        Chemical shift of each pool of shape ``(npools,).
    D : torch.Tensor  
        Apparent diffusion coefficient of shape ``(...,)`` in ``[um**2/ms]``. Assume same coefficient for each pool.
    v : torch.Tensor  
        Spin velocity of shape ``(...,)`` in ``[cm/s]``. Assume same coefficient for each pool.
    grad_props : dict
        Extra parameters.

    Returns
    -------
    TEop : deepmr.bloch.Operator 
        Propagator until TE. If TE=0, this is the Identity.
    TETRop : deepmr.bloch.Operator  
        Propagator until next TR.

    """
    X, S = _free_precess(
        states, 0.5 * ESP, T1, T2, weight, k, chemshift, D, v, grad_props
    )

    # build Xpre and Xpost
    Xpre = ops.CompositeOperator(S, X, name="FSEpre")
    Xpost = ops.CompositeOperator(X, S, name="FSEpost")

    return Xpre, Xpost


def bSSFPStep(states, TE, TR, T1, T2, weight=None, k=None, chemshift=None):
    """
    (Balanced) SSFP propagator.

    Consists of rfpulse (omitted) - free precession.

    Parameters
    ----------
    states : dict
        EPG states matrix.
    TE : float
        Echo Time in ``[ms]``.
    TR : float
        Repetition Time in ``[ms]``.
    T1 : torch.Tensor 
        T1 relaxation time of shape ``(..., npools) in ``[ms]``.
    T2 : torch.Tensor 
        T2 relaxation time of shape ``(..., npools) in ``[ms]``.
    weight : torch.Tensor  
        Pool relative fraction.
    k : torch.Tensor 
        Chemical exchange matrix ``(...., npools, npools)`` in ``[Hz]``.
    chemshift : torch.Tensor  
        Chemical shift of each pool of shape ``(npools,).
    D : torch.Tensor  
        Apparent diffusion coefficient of shape ``(...,)`` in ``[um**2/ms]``. Assume same coefficient for each pool.
    v : torch.Tensor  
        Spin velocity of shape ``(...,)`` in ``[cm/s]``. Assume same coefficient for each pool.
    grad_props : dict
        Extra parameters.

    Returns
    -------
    TEop : deepmr.bloch.Operator 
        Propagator until TE. If TE=0, this is the Identity.
    TETRop : deepmr.bloch.Operator  
        Propagator until next TR.

    """
    XTE, _ = _free_precess(states, TE, T1, T2, weight, k, chemshift, None, None, None)
    XTETR, _ = _free_precess(
        states, TR - TE, T1, T2, weight, k, chemshift, None, None, None
    )
    return XTE, XTETR


def SSFPFidStep(
    states,
    TE,
    TR,
    T1,
    T2,
    weight=None,
    k=None,
    chemshift=None,
    D=None,
    v=None,
    grad_props=None,
):
    """
    (Unbalanced) SSFP propagator.

    Consists of rfpulse (omitted) - free precession - spoiling gradient.

    Parameters
    ----------
    states : dict
        EPG states matrix.
    TE : float
        Echo Time in ``[ms]``.
    TR : float
        Repetition Time in ``[ms]``.
    T1 : torch.Tensor 
        T1 relaxation time of shape ``(..., npools) in ``[ms]``.
    T2 : torch.Tensor 
        T2 relaxation time of shape ``(..., npools) in ``[ms]``.
    weight : torch.Tensor  
        Pool relative fraction.
    k : torch.Tensor 
        Chemical exchange matrix ``(...., npools, npools)`` in ``[Hz]``.
    chemshift : torch.Tensor  
        Chemical shift of each pool of shape ``(npools,).
    D : torch.Tensor  
        Apparent diffusion coefficient of shape ``(...,)`` in ``[um**2/ms]``. Assume same coefficient for each pool.
    v : torch.Tensor  
        Spin velocity of shape ``(...,)`` in ``[cm/s]``. Assume same coefficient for each pool.
    grad_props : dict
        Extra parameters.

    Returns
    -------
    TEop : deepmr.bloch.Operator 
        Propagator until TE. If TE=0, this is the Identity.
    TETRop : deepmr.bloch.Operator  
        Propagator until next TR.

    """
    XTE, _ = _free_precess(states, TE, T1, T2, weight, k, chemshift, D, v, grad_props)
    XTETR, S = _free_precess(
        states, TR - TE, T1, T2, weight, k, chemshift, D, v, grad_props
    )
    return XTE, ops.CompositeOperator(S, XTETR, name="SSFPFid TE-TR Propagator")


def SSFPEchoStep(
    states,
    TE,
    TR,
    T1,
    T2,
    weight=None,
    k=None,
    chemshift=None,
    D=None,
    v=None,
    grad_props=None,
):
    """
    (Reverse) SSFP propagator.

    Consists of rfpulse (omitted) - spoiling gradient - free precession.

    Parameters
    ----------
    states : dict
        EPG states matrix.
    TE : float
        Echo Time in ``[ms]``.
    TR : float
        Repetition Time in ``[ms]``.
    T1 : torch.Tensor 
        T1 relaxation time of shape ``(..., npools) in ``[ms]``.
    T2 : torch.Tensor 
        T2 relaxation time of shape ``(..., npools) in ``[ms]``.
    weight : torch.Tensor  
        Pool relative fraction.
    k : torch.Tensor 
        Chemical exchange matrix ``(...., npools, npools)`` in ``[Hz]``.
    chemshift : torch.Tensor  
        Chemical shift of each pool of shape ``(npools,).
    D : torch.Tensor  
        Apparent diffusion coefficient of shape ``(...,)`` in ``[um**2/ms]``. Assume same coefficient for each pool.
    v : torch.Tensor  
        Spin velocity of shape ``(...,)`` in ``[cm/s]``. Assume same coefficient for each pool.
    grad_props : dict
        Extra parameters.

    Returns
    -------
    TEop : deepmr.bloch.Operator 
        Propagator until TE. If TE=0, this is the Identity.
    TETRop : deepmr.bloch.Operator  
        Propagator until next TR.

    """
    XTE, _ = _free_precess(states, TE, T1, T2, weight, k, chemshift, D, v, grad_props)
    XTETR, S = _free_precess(
        states, TR - TE, T1, T2, weight, k, chemshift, D, v, grad_props
    )
    return ops.CompositeOperator(S, XTE, name="SSFPFid TE-TR Propagator"), XTETR


# %% local subroutine
def _free_precess(states, t, T1, T2, weight, k, chemshift, D, v, grad_props):
    # parse gradient properties
    if grad_props is None:
        grad_props = {}
        tau = None
    elif grad_props:
        tau = grad_props["duration"]
        grad_props.pop("duration")
    else:
        tau = 0.0

    # parse
    device = states["F"].device
    nstates = states["F"].shape[-4]

    # check if has exchange, diffusion and flow
    hasK = k is not None
    hasD = D is not None
    hasV = v is not None

    # prep until TR
    if t != 0:
        if hasK:
            E = ops.Relaxation(
                device,
                t,
                T1,
                T2,
                weight,
                k,
                chemshift,
                name="Relaxation-MT",
            )
        else:
            E = ops.Relaxation(device, t, T1, T2, name="Relaxation")

        # setup washout
        if hasV and "moving" in states:
            W = ops.FlowWash(
                device,
                t,
                v,
                name="Magnetization inflow / washout until next TR",
                **grad_props,
            )
        else:
            W = ops.Identity(name="Inflow+Washout")

        # set up diffusion
        if hasD:
            D = ops.DiffusionDamping(
                device, tau, D, nstates, name="Diffusion Damping", **grad_props
            )
        else:
            D = ops.Identity(name="Diffusion until TE")

        # set up flow
        if hasV:
            J = ops.FlowDephasing(
                device, tau, v, nstates, name="Flow Dephasing", **grad_props
            )
        else:
            J = ops.Identity(name="Flow until next TR")
    else:
        E = ops.Identity(name="Relaxation")
        D = ops.Identity(name="Diffusion Damping")
        J = ops.Identity(name="Flow Dephasing")
        W = ops.Identity(name="Inflow-Washout")

    # set up gradient spoiling
    S = ops.Shift(name="Gradient Spoiling")

    # put everything together
    X = ops.CompositeOperator(W, J, D, E, name="Free Precession")

    return X, S
