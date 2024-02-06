"""
EPG Motion operators.

Can be used to simulate bulk motion and (isotropic) diffusion damping.
"""
__all__ = ["DiffusionDamping", "FlowDephasing", "FlowWash"]

import torch

from ._abstract_op import Operator
from ._utils import gamma_bar

gamma_bar *= 1e6  # [MHz / T] -> [Hz / T]


class DiffusionDamping(Operator):
    """
    Simulate diffusion effects by state dependent damping of the coefficients.

    Parameters
    ----------
    device (str): str
        Computational device (e.g., ``cpu`` or ``cuda:n``, with ``n=0,1,2...``).
    time : torch.Tensor)
        Time step in ``[ms]``.
    D : torch.Tensor 
        Apparent diffusion coefficient ``[um**2 ms**-1]``.
    nstates : int 
        Number of EPG dephasing orders.
    total_dephasing : float, optional
        Total dephasing due to unbalanced gradients in ``[rad]``.
    voxelsize : float, optional 
        Voxel thickness along unbalanced direction in ``[mm]``.
    grad_amplitude : float, optional 
        Gradient amplitude along unbalanced direction in ``[mT / m]``.
    grad_direction : str | torch.Tensor
        Gradient orientation (``"x"``, ``"y"``, ``"z"`` or ``versor``).

    Notes
    -----
    User must provide either total dephasing and voxel size or gradient amplitude and duration.

    Other Parameters
    ----------------
    name : str
        Name of the operator.
        
    """

    def __init__(
        self,
        device,
        time,
        D,
        nstates,
        total_dephasing=None,
        voxelsize=None,
        grad_amplitude=None,
        grad_direction=None,
        **kwargs
    ):  # noqa
        super().__init__(**kwargs)

        # offload (not sure if this is needed?)
        time = torch.as_tensor(time, dtype=torch.float32, device=device)
        time = torch.atleast_1d(time)
        D = torch.as_tensor(D, dtype=torch.float32, device=device)
        D = torch.atleast_1d(D)

        # initialize direction
        if grad_direction is None:
            grad_direction = "z"

        if grad_direction == "x":
            grad_direction = (1.0, 0.0, 0.0)
        if grad_direction == "y":
            grad_direction = (0.0, 1.0, 0.0)
        if grad_direction == "z":
            grad_direction = (0.0, 0.0, 1.0)

        # prepare operators
        D1, D2 = _diffusion_damp_prep(
            time,
            D,
            nstates,
            total_dephasing,
            voxelsize,
            grad_amplitude,
            grad_direction,
        )

        # assign matrices
        self.D1 = D1
        self.D2 = D2

    def apply(self, states):
        """
        Apply diffusion damping.

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
        states = diffusion_damp_apply(states, self.D1, self.D2)

        # diffusion for moving spins
        if "moving" in states:
            states["moving"] = diffusion_damp_apply(states["moving"], self.D1, self.D2)

        return states


class FlowDephasing(Operator):
    """
    Simulate state dependent phase accrual of the EPG coefficients due to flow.

    Parameters
    ----------
    device (str): str
        Computational device (e.g., ``cpu`` or ``cuda:n``, with ``n=0,1,2...``).
    time : torch.Tensor)
        Time step in ``[ms]``.
    v : torch.Tensor 
        Spin velocity of shape ``(3,)`` in ``[cm / s]``. 
        If scalar, assume same direction as unbalanced gradient.
    nstates : int 
        Number of EPG dephasing orders.
    total_dephasing : float, optional
        Total dephasing due to unbalanced gradients in ``[rad]``.
    voxelsize : float, optional 
        Voxel thickness along unbalanced direction in ``[mm]``.
    grad_amplitude : float, optional 
        Gradient amplitude along unbalanced direction in ``[mT / m]``.
    grad_direction : str | torch.Tensor
        Gradient orientation (``"x"``, ``"y"``, ``"z"`` or ``versor``).
    
    Notes
    -----
    User must provide either total dephasing and voxel size or gradient amplitude and duration.

    """

    def __init__(
        self,
        device,
        time,
        v,
        nstates,
        total_dephasing=None,
        voxelsize=None,
        grad_amplitude=None,
        grad_direction=None,
        **kwargs
    ):  # noqa
        super().__init__(**kwargs)

        # offload (not sure if this is needed?)
        time = torch.as_tensor(time, dtype=torch.float32, device=device)
        time = torch.atleast_1d(time)
        v = torch.as_tensor(v, dtype=torch.float32, device=device)
        v = torch.atleast_1d(v)

        # initialize direction
        if grad_direction is None:
            grad_direction = "z"

        if grad_direction == "x":
            grad_direction = (1.0, 0.0, 0.0)
        if grad_direction == "y":
            grad_direction = (0.0, 1.0, 0.0)
        if grad_direction == "z":
            grad_direction = (0.0, 0.0, 1.0)

        # prepare operators
        J1, J2 = _flow_dephase_prep(
            time,
            v,
            nstates,
            total_dephasing,
            voxelsize,
            grad_amplitude,
            grad_direction,
        )

        # assign matrices
        self.J1 = J1
        self.J2 = J2

    def apply(self, states):
        """
        Apply flow dephasing.

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
        states = flow_dephase_apply(states, self.J1, self.J2)

        # dephasing for moving spins
        if "moving" in states:
            states["moving"] = flow_dephase_apply(states["moving"], self.J1, self.J2)

        return states


class FlowWash(Operator):
    """
    Simulate EPG states replacement due to flow.
    
    device (str): str
        Computational device (e.g., ``cpu`` or ``cuda:n``, with ``n=0,1,2...``).
    time : torch.Tensor)
        Time step in ``[ms]``.
    v : torch.Tensor 
        Spin velocity of shape ``(3,)`` in ``[cm / s]``. 
        If scalar, assume same direction as unbalanced gradient.
    voxelsize : float, optional 
        Voxel thickness along unbalanced direction in ``[mm]``.
    slice_direction : str | torch.Tensor
        Slice orientation (``"x"``, ``"y"``, ``"z"`` or ``versor``).

    """

    def __init__(
        self, device, time, v, voxelsize, slice_direction=None, **kwargs
    ):  # noqa
        super().__init__(**kwargs)

        # offload (not sure if this is needed?)
        time = torch.as_tensor(time, dtype=torch.float32, device=device)
        time = torch.atleast_1d(time)
        v = torch.as_tensor(v, dtype=torch.float32, device=device)
        v = torch.atleast_1d(v)

        # initialize direction
        if slice_direction is None:
            slice_direction = "z"

        if slice_direction == "x":
            slice_direction = (1.0, 0.0, 0.0)
        if slice_direction == "y":
            slice_direction = (0.0, 1.0, 0.0)
        if slice_direction == "z":
            slice_direction = (0.0, 0.0, 1.0)

        # prepare operators
        Win, Wout = _flow_washout_prep(time, v, voxelsize, slice_direction)

        # assign matrices
        self.Win = Win
        self.Wout = Wout

    def apply(self, states):
        """
        Apply spin replacement.

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
        states = flow_washout_apply(states, self.Win, self.Wout)
        return states


# %% local utils
def _diffusion_damp_prep(
    time,
    D,
    nstates,
    total_dephasing,
    voxelsize,
    grad_amplitude,
    grad_direction,
):
    # check inputs
    if total_dephasing is None or voxelsize is None:
        assert (
            grad_amplitude is not None
        ), "Please provide either total_dephasing/voxelsize or grad_amplitude."
    if grad_amplitude is None:
        assert (
            total_dephasing is not None and voxelsize is not None
        ), "Please provide either total_dephasing/voxelsize or grad_amplitude."

    # if total dephasing is not provided, calculate it:
    if total_dephasing is None or voxelsize is None:
        gamma = 2 * torch.pi * gamma_bar
        k0_2 = (gamma * grad_amplitude * time * 1e-6) ** 2
    else:
        voxelsize = _get_projection(voxelsize, grad_direction)
        k0_2 = (total_dephasing / voxelsize / 1e-3) ** 2

    # cast to tensor
    k0_2 = torch.as_tensor(k0_2, dtype=torch.float32, device=time.device)
    k0_2 = torch.atleast_1d(k0_2)

    # actual operator calculation
    b_prime = k0_2 * time * 1e-3

    # calculate dephasing order
    l = torch.arange(nstates, dtype=torch.float32, device=D.device)[:, None, None]
    lsq = l**2

    # calculate b-factor
    b1 = b_prime * lsq
    b2 = b_prime * (lsq + l + 1.0 / 3.0)

    # actual operator calculation
    D1 = torch.exp(-b1 * D * 1e-9)
    D2 = torch.exp(-b2 * D * 1e-9)

    return D1, D2


def diffusion_damp_apply(states, D1, D2):
    # parse
    F, Z = states["F"], states["Z"]

    # apply
    F[..., 0] = F[..., 0].clone() * D2  # Transverse damping
    F[..., 1] = F[..., 1].clone() * D2  # Transverse damping
    Z = Z.clone() * D1  # Longitudinal damping

    # prepare for output
    states["F"], states["Z"] = F, Z
    return states


def _flow_dephase_prep(
    time,
    v,
    nstates,
    total_dephasing,
    voxelsize,
    grad_amplitude,
    grad_direction,
):
    # check inputs
    if total_dephasing is None or voxelsize is None:
        assert (
            grad_amplitude is not None
        ), "Please provide either total_dephasing/voxelsize or grad_amplitude."
    if grad_amplitude is None:
        assert (
            total_dephasing is not None and voxelsize is not None
        ), "Please provide either total_dephasing/voxelsize or grad_amplitude."

    # if total dephasing is not provided, calculate it:
    if total_dephasing is None or voxelsize is None:
        dk = 2 * torch.pi * gamma_bar * grad_amplitude * time * 1e-6
    else:
        voxelsize = _get_projection(voxelsize, grad_direction)
        dk = total_dephasing / voxelsize / 1e-3

    # calculate dephasing order
    l = torch.arange(nstates, dtype=torch.float32, device=v.device)[:, None, None]
    k0 = dk * l

    # get velocity
    v = _get_projection(v, grad_direction)

    # actual operator calculation
    J1 = torch.exp(-1j * k0 * v * 1e-5 * time)  # cm / s -> m / ms
    J2 = torch.exp(-1j * (k0 + 0.5 * dk) * v * 1e-5 * time)  # cm / s -> m / ms

    return J1, J2


def flow_dephase_apply(states, J1, J2):
    # parse
    F, Z = states["F"], states["Z"]

    # apply
    F[..., 0] = F[..., 0].clone() * J2  # Transverse dephasing
    F[..., 1] = F[..., 1].clone() * J2.conj()  # Transverse dephasing
    Z = Z.clone() * J1  # Longitudinal dephasing

    # prepare for output
    states["F"], states["Z"] = F, Z
    return states


def _flow_washout_prep(time, v, voxelsize, slice_direction=None):
    # get effective velocity and voxel size
    v = _get_projection(v, slice_direction) * 1e-2  # [cm / s] -> [mm / ms]
    voxelsize = _get_projection(voxelsize, slice_direction)  # [mm]

    # calculate washout rate
    R = torch.abs(v / voxelsize)  # [1 / ms]

    # flow wash-in/out
    Win = R * time
    Wout = 1 - Win

    # erase unphysical entries
    Win = (
        1.0
        - torch.heaviside(
            Win - 1.0, torch.as_tensor(1.0, dtype=R.dtype, device=R.device)
        )
    ) * Win + torch.heaviside(
        Win - 1.0, torch.as_tensor(1.0, dtype=R.dtype, device=R.device)
    )
    Wout = (
        torch.heaviside(Wout, torch.as_tensor(1.0, dtype=R.dtype, device=R.device))
        * Wout
    )

    return Win, Wout


def flow_washout_apply(states, Win, Wout):
    # parse
    F, Z = states["F"], states["Z"]
    Fmoving, Zmoving = states["moving"]["F"], states["moving"]["Z"]

    # apply
    F[..., 0] = Wout * F[..., 0].clone() + Win * Fmoving[..., 0]
    F[..., 1] = Wout * F[..., 1].clone() + Win * Fmoving[..., 1]
    Z = Wout * Z.clone() + Win * Zmoving

    # prepare for output
    states["F"], states["Z"] = F, Z
    return states


def _get_projection(arr, direction):
    # prepare direction
    arr = torch.as_tensor(arr)
    direction = torch.as_tensor(direction)

    # get device
    device = arr.device
    arr = arr.to(device)
    direction = direction.to(device)

    # expand if required
    arr = torch.atleast_1d(arr)
    if arr.shape[-1] == 1:
        arr = torch.cat(
            (direction[0] * arr, direction[1] * arr, direction[2] * arr), dim=-1
        )

    return arr @ direction
