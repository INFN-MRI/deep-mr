__all__ = ["EPGstates"]

import numpy as np
import torch


def EPGstates(
    device,
    batch_size,
    nstates,
    nlocs,
    npulses,
    npools=1,
    weight=None,
    model="single",
    moving=False,
):
    """
    EPG states matrix.

    Stores dephasing states for transverse and longitudinal magnetization.

    Parameters
    ----------
    device : str
        Computational device (e.g., ``cpu`` or ``cuda:n``, with ``n=0,1,2...``).
    batch_size : int
        Number of different atoms (e.g., voxels) to be simultaneously simulated.
    nstates : int
        Maximum number of dephasing states.
    nlocs : int
        Number of spatial locations contributing to each atom states (e.g., slice points).
    npulses : int
        Number of RF pulses applied during the sequence.
    npools : int
        Number of pools contributing to signal (e.g., Free Water / Myelin Water).
    weight : torch.Tensor[float], optional
        Relative fraction for each pool. For single pool, this is the initial magnetization.
        The default is ``None`` (i.e., ``M0 = [0, 0, 1]``).
    model : str, optional
        Type of signal model. If "mt" is in the model name,
        include pure longitudinal states (bound pool).
        The default is ``single`` (single pool model).
    moving : bool, optional
        Flag for moving spins. If ``True`` include a fresh magnetization pool
        to replace states with ``v != 0``. The default is ``False``.

    Returns
    -------
    out : dict
        Dictionary with states (e.g., ``F``, ``Z``) and signal buffer.

    """
    # prepare output
    out = {}

    if weight is not None:
        weight = weight.clone()
        if len(weight.shape) == 1:
            weight = weight[None, :]

    if np.isscalar(npulses):
        npulses = [npulses]

    # initialize free pool
    # transverse
    F = torch.zeros(
        (batch_size, nstates, nlocs, npools, 2), dtype=torch.complex64, device=device
    )
    F = {"real": F.real, "imag": F.imag}

    # initialize free pool
    # longitudinal
    Z = torch.zeros(
        (batch_size, nstates, nlocs, npools), dtype=torch.complex64, device=device
    )
    Z[:, 0, ...] = 1.0

    if weight is not None:
        Z = Z * weight[:, :npools][:, None, None]
    Z = {"real": Z.real, "imag": Z.imag}

    # append
    out["states"] = {"F": F, "Z": Z}

    # initialize moving pool
    if moving:
        # transverse
        Fmoving = torch.zeros(
            (batch_size, nstates, nlocs, npools, 2),
            dtype=torch.complex64,
            device=device,
        )
        Fmoving = {"real": Fmoving.real, "imag": Fmoving.imag}

        # initialize free pool
        # longitudinal
        Zmoving = torch.zeros(
            (batch_size, nstates, nlocs, npools), dtype=torch.complex64, device=device
        )
        Zmoving[:, 0, ...] = 1.0
        if weight is not None:
            Zmoving = Zmoving * weight[:, :npools][:, None, None]
        Zmoving = {"real": Zmoving.real, "imag": Zmoving.imag}

        # append
        out["states"]["moving"] = {}
        out["states"]["moving"]["F"] = Fmoving
        out["states"]["moving"]["Z"] = Zmoving

    # initialize bound pool
    if model is not None and "mt" in model:
        Zbound = torch.zeros(
            (batch_size, nstates, nlocs, 1), dtype=torch.complex64, device=device
        )
        Zbound[:, 0, :, :] = 1.0
        Zbound = Zbound * weight[:, -1][:, None, None, None]
        Zbound = {"real": Zbound.real, "imag": Zbound.imag}
        out["states"]["Zbound"] = Zbound

        if moving:
            out["states"]["moving"]["Zbound"] = Zbound.clone()

    # initialize output signal
    sig = torch.zeros([batch_size] + npulses, dtype=torch.complex64, device=device)
    sig = {"real": sig.real, "imag": sig.imag}

    # append
    out["signal"] = sig

    return out
