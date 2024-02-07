"""
EPG Relaxation operators.

Can be used to simulate longitudinal and transverse relaxation either
in absence or presence of exchange (Chemical Exchange or MT), as well as
accounting for chemical shift.
"""

__all__ = ["Relaxation"]

import math

import torch

from ._abstract_op import Operator
from ._utils import matrix_exp

class Relaxation(Operator):
    """
    The "decay operator" applying relaxation and "regrowth" of the magnetization components.

    Parameters
    ----------
    device (str): str
        Computational device (e.g., ``cpu`` or ``cuda:n``, with ``n=0,1,2...``).
    time : torch.Tensor)
        Time step in ``[ms]``.
    T1 : torch.Tensor
        Longitudinal relaxation time in ``[ms]`` of shape ``(npools,)``.
    T2 : torch.Tensor 
        Transverse relaxation time in ``[ms]`` of shape ``(npools,)``.
    weight : torch.Tensor, optional
        Relative pool fractions of shape ``(npools,)``.
    k : torch.Tensor, optional
        Exchange matrix of shape ``(npools, npools)`` in ``[s**-1]``.
    df : torch.Tensor, optional
        Chemical shift in ``[Hz]`` of shape ``(npools,)``.

    Other Parameters
    ----------------
    name : str
        Name of the operator.

    """

    def __init__(
        self, device, time, T1, T2, weight=None, k=None, df=None, **kwargs
    ):  # noqa
        super().__init__(**kwargs)

        # offload (not sure if this is needed?)
        time = torch.as_tensor(time, dtype=torch.float32, device=device)
        time = torch.atleast_1d(time)
        T1 = torch.as_tensor(T1, dtype=torch.float32, device=device)
        T1 = torch.atleast_1d(T1)
        T2 = torch.as_tensor(T2, dtype=torch.float32, device=device)
        T2 = torch.atleast_1d(T2)

        # cast to tensors
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32, device=device)
        if k is not None:
            k = torch.as_tensor(k, dtype=torch.float32, device=device)
            k = _prepare_exchange(weight, k)

        if df is not None:
            df = torch.as_tensor(df, dtype=torch.float32, device=device)
            df = torch.atleast_1d(df)

        # prepare operators
        if weight is None or k is None:
            E2 = _transverse_relax_prep(time, T2)
            E1, rE1 = _longitudinal_relax_prep(time, T1)

            # assign functions
            self._transverse_relax_apply = _transverse_relax_apply
            self._longitudinal_relax_apply = _longitudinal_relax_apply

        else:
            E2, self._transverse_relax_apply = _transverse_relax_exchange_prep(
                time, T2, k, df
            )
            E1, rE1 = _longitudinal_relax_exchange_prep(time, T1, weight, k)

            # assign functions
            self._longitudinal_relax_apply = _longitudinal_relax_exchange_apply

        # assign matrices
        self.E1 = E1
        self.rE1 = rE1
        self.E2 = E2

    def apply(self, states):
        """
        Apply free precession (relaxation + precession + exchange + recovery).

        Parameters
        ----------
        states : dict
            Input states matrix for free pools 
            and, optionally, for bound pools.

        Returns
        -------
        states : dict 
            Output states matrix for free pools
            and, optionally, for bound pools.

        """
        states = self._transverse_relax_apply(states, self.E2)
        states = self._longitudinal_relax_apply(states, self.E1, self.rE1)

        # relaxation for moving spins
        if "moving" in states:
            states["moving"] = self._transverse_relax_apply(states["moving"], self.E2)
            states["moving"] = self._longitudinal_relax_apply(
                states["moving"], self.E1, self.rE1
            )

        return states


# %% local utils
def _prepare_exchange(weight, k):
    # prepare
    if k.shape[-1] == 1:  # BM or MT
        k0 = 0 * k
        k1 = torch.cat((k0, k * weight[..., [0]]), axis=-1)
        k2 = torch.cat((k * weight[..., [1]], k0), axis=-1)
        k = torch.stack((k1, k2), axis=-2)
    else:  # BM-MT
        k0 = 0 * k[..., [0]]
        k1 = torch.cat((k0, k[..., [0]] * weight[..., [0]], k0), axis=-1)
        k2 = torch.cat(
            (k[..., [0]] * weight[..., [1]], k0, k[..., [1]] * weight[..., [1]]),
            axis=-1,
        )
        k3 = torch.cat((k0, k[..., [1]] * weight[..., [2]], k0), axis=-1)
        k = torch.stack((k1, k2, k3), axis=-2)

    # finalize exchange
    return _particle_conservation(k)


def _particle_conservation(k):
    """Adjust diagonal of exchange matrix by imposing particle conservation."""
    # get shape
    npools = k.shape[-1]

    for n in range(npools):
        k[..., n, n] = 0.0  # ignore existing diagonal
        k[..., n, n] = -k[..., n].sum(dim=-1)

    return k


def _transverse_relax_apply(states, E2):
    # parse
    F = states["F"]

    # apply
    F[..., 0] = F[..., 0].clone() * E2  # F+
    F[..., 1] = F[..., 1].clone() * E2.conj()  # F-

    # prepare for output
    states["F"] = F
    return states


def _longitudinal_relax_apply(states, E1, rE1):
    # parse
    Z = states["Z"]

    # apply
    Z = Z.clone() * E1  # decay
    Z[0] = Z[0].clone() + rE1  # regrowth

    # prepare for output
    states["Z"] = Z
    return states


def _transverse_relax_prep(time, T2):
    # compute R2
    R2 = 1 / T2

    # calculate operators
    E2 = torch.exp(-R2 * time)

    return E2


def _longitudinal_relax_prep(time, T1):
    # compute R2
    R1 = 1 / T1

    # calculate operators
    E1 = torch.exp(-R1 * time)
    rE1 = 1 - E1

    return E1, rE1


def _transverse_relax_exchange_apply(states, E2):
    # parse
    F = states["F"]

    # apply
    F[..., 0] = torch.einsum("...ij,...j->...i", E2, F[..., 0].clone())
    F[..., 1] = torch.einsum("...ij,...j->...i", E2.conj(), F[..., 1].clone())

    # prepare for output
    states["F"] = F
    return states


def _longitudinal_relax_exchange_apply(states, E1, rE1):
    # parse
    Z = states["Z"]

    # get ztot
    if "Zbound" in states:
        Zbound = states["Zbound"]
        Ztot = torch.cat((Z, Zbound), axis=-1)
    else:
        Ztot = Z

    # apply
    Ztot = torch.einsum("...ij,...j->...i", E1, Ztot.clone())
    Ztot[0] = Ztot[0].clone() + rE1

    # prepare for output
    if "Zbound" in states:
        states["Z"] = Ztot[..., :-1]
        states["Zbound"] = Ztot[..., [-1]]
    else:
        states["Z"] = Ztot
    return states


def _transverse_relax_exchange_prep(time, T2, k, df=None):
    # compute R2
    R2 = 1 / T2

    # add chemical shift
    if df is not None:
        R2tot = R2 + 1j * 2 * math.pi * df * 1e-3  # (account for time in [ms])
    else:
        R2tot = R2

    # get npools
    npools = R2tot.shape[-1]

    # case 1: MT
    if npools == 1:
        return torch.exp(-R2tot * time), _transverse_relax_apply

    # case 2: BM or BM-MT
    else:
        # cast to complex
        R2tot = R2tot.to(torch.complex64)

        # recovery
        Id = torch.eye(npools, dtype=R2tot.dtype, device=R2tot.device)

        # coefficients
        lambda2 = (
            k[..., :npools, :npools] * 1e-3 - R2tot[:, None] * Id
        )  # assume MT pool is the last

        # actual operators
        E2 = matrix_exp(lambda2 * time)

        return E2, _transverse_relax_exchange_apply


def _longitudinal_relax_exchange_prep(time, T1, weight, k):
    # compute R2
    R1 = 1 / T1

    # get npools
    npools = R1.shape[-1]

    if weight.shape[-1] == npools + 1:  # MT case
        R1 = torch.cat((R1, R1[..., [0]]), axis=-1)
        npools += 1

    # cast to complex
    R1 = R1.to(torch.complex64)

    # recovery
    Id = torch.eye(npools, dtype=R1.dtype, device=R1.device)
    C = weight * R1

    # coefficients
    lambda1 = k * 1e-3 - R1 * Id

    # actual operators
    E1 = matrix_exp(lambda1 * time)
    rE1 = torch.einsum("...ij,...j->...i", (E1 - Id), torch.linalg.solve(lambda1, C))

    return E1, rE1
