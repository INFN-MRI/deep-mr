"""Common preparation blocks."""

__all__ = ["InversionPrep", "T2Prep"]

from .. import ops


def InversionPrep(TI, T1, T2, weight, k, inv_props):
    """
    Adiabatic inversion operator.

    Consists of a 180° pulse followed by a crusher gradient.

    Parameters
    ----------
    TI : torch.Tensor
        Inversion time in ``[ms]``.
    T1 : torch.Tensor
        T1 relaxation time of shape ``(..., npools) in ``[ms]``.
    T2 : torch.Tensor
        T2 relaxation time of shape ``(..., npools) in ``[ms]``.
    weight : torch.Tensor
        Pool relative fraction.
    k : torch.Tensor
        Chemical exchange matrix ``(...., npools, npools)`` in ``[Hz]``.
    prep_props : dict
        Extra pulse parameters.

    Returns
    -------
    PrepPulse : deepmr.bloch.Operator
        Adiabatic Inversion pulse operator, including crusher.

    """
    if TI is not None and TI != 0.0:
        # parse inversion properties
        if inv_props is None:
            inv_props = {}

        # prep operator
        Tinv = ops.AdiabaticPulse(
            T1.device, alpha=180.0, name="Inversion Pulse", **inv_props
        )
        Einv = ops.Relaxation(
            T1.device, TI, T1, T2, weight, k, name="Preparation Interval"
        )
        Sinv = ops.Spoil(name="Inversion Crusher")

        return ops.CompositeOperator(Sinv, Einv, Tinv, name="Inversion Propagator")
    else:
        return ops.Identity(name="Inversion Propagator")


def T2Prep(Tprep, T1, T2, weight, k, prep_props):
    """
    T2 prep operator.

    Consists of a 90°-180°--90° composite pulse followed by a crusher gradient.

    Parameters
    ----------
    Tprep : torch.Tensor
        T2 preparation time in ``[ms]``.
    T1 : torch.Tensor
        T1 relaxation time of shape ``(..., npools) in ``[ms]``.
    T2 : torch.Tensor
        T2 relaxation time of shape ``(..., npools) in ``[ms]``.
    weight : torch.Tensor
        Pool relative fraction.
    k : torch.Tensor
        Chemical exchange matrix ``(...., npools, npools)`` in ``[Hz]``.
    prep_props : dict
        Extra pulse parameters.

    Returns
    -------
    PrepPulse : deepmr.bloch.Operator
        Adiabatic T2prep pulse operator, including crusher.

    """
    if Tprep is not None and Tprep != 0.0:
        # parse inversion properties
        if prep_props is None:
            prep_props = {}

        # prep operator
        T90p = ops.AdiabaticPulse(
            T1.device, alpha=90.0, phi=0.0, name="Flip Pulse", **prep_props
        )
        Eprep = ops.Relaxation(
            T1.device, 0.5 * Tprep, T1, T2, weight, k, name="Preparation Interval"
        )
        T180 = ops.AdiabaticPulse(
            T1.device, alpha=180.0, name="Inversion Pulse", **prep_props
        )
        T90m = ops.AdiabaticPulse(
            T1.device, alpha=90.0, phi=-180.0, name="Flip-back Pulse", **prep_props
        )
        Sprep = ops.Spoil(name="Prep Crusher")

        return ops.CompositeOperator(
            Sprep, T90m, Eprep, T180, Eprep, T90p, name="T2prep Propagator"
        )
    else:
        return ops.Identity(name="T2prep Propagator")
