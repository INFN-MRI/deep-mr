"""
EPG RF Pulses operators.

Can be used to simulate different types of RF pulses (soft and hard)
and multiple transmit coil modes.
"""
__all__ = ["AdiabaticPulse", "RFPulse"]


import numpy as np
import scipy.interpolate
import torch

from ._abstract_op import Operator
from ._stats import pulse_analysis
from ._utils import gamma

class BasePulse(Operator):
    """
    Operator implementing the transformation performed by application of an RF pulse.

    Args:
        device (str): computational device (e.g., 'cpu' or 'cuda:n', with n=0,1,2...).
        nlocs (int): number of spatial locations for slice profile simulation.
        alpha (torch.Tensor, optional): pulse flip angle in [deg] of shape (nmodes,).
        phi (torch.Tensor, optional): pulse phase in [deg] of shape (nmodes,).

    Props:
        name (str): Name of the operator.
        rf_envelope (torch.Tensor): pulse time envelope.
        duration (float): pulse duration in [ms].
        b1rms (float): Pulse root-mean-squared B1 in [uT / deg] (when pulse is scaled such as flip angle = 1 [deg]).
        freq_offset (float): pulse frequency offset in [Hz].

    """

    def __init__(self, device, alpha=0.0, phi=0.0, **props):
        super().__init__(**props)

        # save device
        self.device = device

        # get pulse stats
        self.b1rms = None  # pulse root-mean-squared b1 (for alpha=1 deg)
        self.freq_offset = None  # pulse frequency offset [Hz] (nbands,)
        self.G = None

        # slice selection
        if "slice_selective" in props:
            self.slice_selective = props["slice_selective"]

        # duration
        if "duration" in props:
            self.tau = torch.as_tensor(
                props["duration"], dtype=torch.float32, device=device
            )

        # frequency offset
        if "freq_offset" in props:
            self.freq_offset = torch.as_tensor(
                props["freq_offset"], dtype=torch.float32, device=device
            )

        # b1rms
        # calculate from envelope...
        if "rf_envelope" in props and "duration" in props:
            info, _, _, _, _ = pulse_analysis(
                props["rf_envelope"], props["duration"],
            )
            
            # b1rms
            self.b1rms = torch.as_tensor(
                info["b1rms"], dtype=torch.float32, device=device
            )

        # ...or directly get from input
        # b1rms
        if "b1rms" in props:
            self.b1rms = torch.as_tensor(
                props["b1rms"], dtype=torch.float32, device=device
            )

        # calculate absorption linewidth and local field fluctuation
        if self.freq_offset is not None:
            G = super_lorentzian_lineshape(self.freq_offset) * 1e3  # s -> ms
            G = torch.as_tensor(G, dtype=torch.float32, device=device)
            self.G = torch.atleast_1d(G)
        else:
            G = super_lorentzian_lineshape(0.0) * 1e3  # s -> ms
            G = torch.as_tensor(G, dtype=torch.float32, device=device)
            self.G = torch.atleast_1d(G)

        # initialize saturation
        self.initialize_saturation()
        
        # default slice profile
        slice_profile = torch.as_tensor(1.0, dtype=torch.float32, device=device)
        self.slice_profile = torch.atleast_1d(slice_profile)
        
        # default B1 value
        B1 = torch.ones(1, dtype=torch.float32, device=device)

        # set B1
        B1abs = B1.abs()
        self.B1abs = torch.atleast_1d(B1abs)
        B1angle = B1.angle()
        self.B1angle = torch.atleast_1d(B1angle)

    def prepare_rotation(self, alpha, phi):
        """
        Prepare the matrix describing rotation due to RF pulse.

        Args:
            alpha (torch.Tensor): pulse flip angle in [deg] of shape (nmodes,).
            phi (torch.Tensor): pulse phase in [deg] of shape (nmodes,).

        """
        # get device
        device = self.device
    
        # get B1
        B1abs = self.B1abs
        B1angle = self.B1angle

        # cast to tensor if needed
        alpha = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        alpha = torch.atleast_1d(alpha)
        phi0 = torch.as_tensor(phi, dtype=torch.float32, device=device)
        phi0 = torch.atleast_1d(phi0)

        # convert from degrees to radians
        alpha = torch.deg2rad(alpha)
        phi0 = torch.deg2rad(phi0)

        # apply B1 effect
        fa = (B1abs * alpha).sum(axis=-1)
        phi = (phi0 + B1angle).sum(axis=-1)

        # apply slice profile
        if self.slice_profile is not None:
            fa = self.slice_profile * fa

        # calculate operator
        T00 = torch.cos(fa / 2) ** 2
        T01 = torch.exp(2 * 1j * phi) * (torch.sin(fa / 2)) ** 2
        T02 = -1j * torch.exp(1j * phi) * torch.sin(fa)
        T10 = T01.conj()
        T11 = T00
        T12 = 1j * torch.exp(-1j * phi) * torch.sin(fa)
        T20 = -0.5 * 1j * torch.exp(-1j * phi) * torch.sin(fa)
        T21 = 0.5 * 1j * torch.exp(1j * phi) * torch.sin(fa)
        T22 = torch.cos(fa)

        # build rows
        T0 = [T00[..., None], T01[..., None], T02[..., None]]
        T1 = [T10[..., None], T11[..., None], T12[..., None]]
        T2 = [T20[..., None], T21[..., None], T22[..., None]]

        # build matrix
        T = [T0, T1, T2]

        # keep matrix
        self.T = T

        # return phase for demodulation
        self.phi = phi0.sum(axis=-1)

    def prepare_saturation(self, alpha):
        """
        Prepare the matrix describing saturation due to RF pulse.

        Args:
            alpha (torch.Tensor): pulse flip angle in [deg] of shape (nmodes,).

        """
        if self.WT is not None:
            # get device
            device = self.device

            # get B1
            B1abs = self.B1abs

            # cast to tensor if needed
            alpha = torch.as_tensor(alpha, dtype=torch.float32, device=device)
            alpha = torch.atleast_1d(alpha)

            # convert from degrees to radians
            alpha = torch.deg2rad(alpha)

            # apply B1 effect
            fa = (B1abs * alpha).sum(axis=-1)

            # apply slice profile
            if self.slice_profile is not None:
                fa = self.slice_profile * fa

            # get scale
            scale = fa**2

            # actual calculation
            self.S = torch.exp(scale * self.WT)

    def initialize_saturation(self):
        # build operator
        try:
            # get parameters
            tau = self.tau  # [ms]
            b1rms = self.b1rms  # [uT]
            G = self.G  # [ms]

            # calculate W and D
            W = torch.pi * (gamma * 1e-3) ** 2 * b1rms**2 * G
            self.WT = -W * tau

        except:
            self.WT = None

    def check_saturation_operator(self):
        if self.WT is None:
            missing = []
            msg = " - please provide tau and either pulse envelope or its b1rms and frequency offset."
            if self.tau is None:
                missing.append("Tau")
            if self.b1rms is None:
                missing.append("B1rms")
            if self.freq_offset is None:
                missing.append("Frequency Offset")
            missing = ", ".join(missing)
            raise RuntimeError(f"{missing} not provided" + msg)

    def apply(self, states, alpha=None, phi=0.0):
        """
        Apply RF pulse (rotation + saturation).

        Args:
            states (dict): input states matrix for free pools
                and, optionally, for bound pools.
            alpha(torch.Tensor, optional): flip angle in [deg].
            phi (torch.Tensor, optional): rf phase in [deg].

        Returns:
            (dict): output states matrix for free pools
                and, optionally, for bound pools.

        """
        # rotate free pools
        if alpha is not None:
            self.prepare_rotation(alpha, phi)
        states = _apply_rotation(states, self.T)

        # rotate moving pools
        if "moving" in states and self.slice_selective is False:
            states["moving"] = _apply_rotation(states["moving"], self.T)

        # saturate bound pool
        if "Zbound" in states:
            if alpha is not None:
                self.prepare_saturation(alpha)
            states = _apply_saturation(states, self.S)

            # saturate moving pools
            if "moving" in states and self.slice_selective is False:
                states["moving"] = _apply_saturation(states["moving"], self.S)

        return states


class RFPulse(BasePulse):  # noqa
    def __init__(self, device, nlocs=None, alpha=0.0, phi=0.0, B1=1.0, **props):  # noqa
        
        # base initialization    
        super().__init__(device, alpha, phi, **props)
        
        # slice selectivity
        if "slice_selective" in props:
            self.slice_selective = props["slice_selective"]
        elif "slice_profile" in props:
            self.slice_selective = True
        else:
            self.slice_selective = False

        # calculate from envelope...
        if "rf_envelope" in props and "duration" in props:
            # slice profile
            if self.slice_selective:
                _, slice_profile, _, _, _ = pulse_analysis(
                    props["rf_envelope"], props["duration"], npts=2*nlocs
                )
                self.slice_profile = torch.as_tensor(
                    abs(slice_profile), dtype=torch.float32, device=device
                )
                self.slice_profile = torch.atleast_1d(slice_profile.squeeze())[:nlocs]
                self.slice_profile = self.slice_profile / self.slice_profile[-1]

        # ...or directly get from input
        # slice profile
        if self.slice_selective and "slice_profile" in props:
            slice_profile = torch.as_tensor(
                props["slice_profile"], dtype=torch.float32, device=device
            )
            self.slice_profile = torch.atleast_1d(slice_profile)
        
        # number of locations
        if nlocs is not None:
            self.nlocs = nlocs
        else:
            self.nlocs = len(self.slice_profile)

        # interpolate slice profile
        if len(self.slice_profile) != self.nlocs:
            x = np.linspace(0, 1, len(self.slice_profile))
            xq = np.linspace(0, 1, self.nlocs)
            y = self.slice_profile.detach().cpu().numpy()
            yq = _spline(x, y, xq)
            slice_profile = torch.as_tensor(yq, dtype=torch.float32, device=device)
            self.slice_profile = torch.atleast_1d(slice_profile)
            self.slice_profile = self.slice_profile / self.slice_profile[-1]
                                       
        # default B1 value
        if B1 is not None:
            B1 = torch.as_tensor(B1, device=device)     
            B1abs = B1.abs()
            self.B1abs = torch.atleast_1d(B1abs)
            B1angle = B1.angle()
            self.B1angle = torch.atleast_1d(B1angle)
        
        # actual preparation (if alpha is provided)
        self.prepare_rotation(alpha, phi)
        self.prepare_saturation(alpha)


class AdiabaticPulse(BasePulse):  # noqa
    def __init__(
        self, device, alpha=0.0, phi=0.0, efficiency=1.0, **props
    ):  # noqa
        super().__init__(device, alpha, phi, **props)
        
        # actual preparation (if alpha is provided)
        self.prepare_rotation(alpha, phi)
        self.prepare_saturation(alpha)

        # compute efficiency
        self.efficiency = efficiency

    def apply(self, states, alpha=None, phi=0.0): # noqa
        states = super().apply(states, alpha, phi)
        # states = states * self.efficiency
        return states


# %% local utils
def _apply_rotation(states, rf_mat):
    """
    Propagate EPG states through an RF rotation.

    Args:
        states (dict): input states matrix for free pool.
        rf_mat (torch.Tensor): rf matrix of shape (nloc, 3, 3).

    Returns:
        (dict): output states matrix for free pools.

    """
    # parse
    Fin, Zin = states["F"], states["Z"]

    # prepare out
    Fout = Fin.clone()
    Zout = Zin.clone()
    
    # apply
    Fout[..., 0] = (
        rf_mat[0][0] * Fin[..., 0] + rf_mat[0][1] * Fin[..., 1] + rf_mat[0][2] * Zin
    )
    Fout[..., 1] = (
        rf_mat[1][0] * Fin[..., 0] + rf_mat[1][1] * Fin[..., 1] + rf_mat[1][2] * Zin
    )
    Zout = rf_mat[2][0] * Fin[..., 0] + rf_mat[2][1] * Fin[..., 1] + rf_mat[2][2] * Zin

    # prepare for output
    states["F"], states["Z"] = Fout, Zout
    return states


def _apply_saturation(states, sat_mat):
    """
    Propagate EPG states through an RF saturation.

    Args:
        states (dict): input states matrix for bound pool.
        sat_mat (torch.Tensor): rf saturation factor of shape (nloc,).

    Returns:
        (dict): output states matrix for bound pools.
    """    
    # parse
    Zbound = states["Zbound"]
    
    # prepare
    Zbound = sat_mat * Zbound.clone()

    # prepare for output
    states["Zbound"] = Zbound
    return states


def super_lorentzian_lineshape(f, T2star=12e-6, fsample=[-30e3, 30e3]):
    """
    Super Lorentzian lineshape.

    Usage:
    >>> G = SuperLorentzianLineshape(12e-3, torch.arange(-500, 500))

    Args:
        f (float): frequency offset of the pulse in [Hz].
        T2star (float, optional): T2 of semisolid compartment in [ms]. Defaults to 12e-3 (12 us).
        fsample (list, tuple, optional): frequency range at which function is to be evaluated (in [Hz]). Defaults to [-2e3, 2e3].

    Returns:
        G(omega): interpolation function to estimate lineshape at arbitrary frequency offset omega in [fsample[0], fsample[1]].

    Shaihan Malik (c), King's College London, April 2019
    Matteo Cencini: Python porting (December 2022)
    """
    # clone
    if isinstance(f, torch.Tensor):
        f = f.clone()
        f = f.cpu().numpy()
    else:
        f = np.asarray(f, dtype=np.float32)
    f = np.atleast_1d(f)

    # as suggested by Gloor, we can interpolate the lineshape across from
    # ± 1kHz
    nu = 100  # <-- number of points for theta integral

    # compute over a wider range of frequencies
    n = 128
    if fsample[0] > -30e3:
        fmin = -33e3
    else:
        fmin = 1.1 * fsample[0]

    if fsample[1] < 30e3:
        fmax = 33e3
    else:
        fmax = 1.1 * fsample[1]

    ff = np.linspace(fmin, fmax, n, dtype=np.float32)

    # np for Super Lorentzian, predefine
    u = np.linspace(0.0, 1.0, nu)
    du = np.diff(u)[0]

    # get integration grid
    ug, ffg = np.meshgrid(u, ff, indexing="ij")

    # prepare integrand
    g = np.sqrt(2 / np.pi) * T2star / np.abs(3 * ug**2 - 1)
    g = g * np.exp(-2 * (2 * np.pi * ffg * T2star / (3 * ug**2 - 1)) ** 2)

    # integrate over theta
    G = du * g.sum(axis=0)

    # interpolate zero frequency
    po = np.abs(ff) < 1e3  # points to interpolate
    pu = np.logical_not(po) * (
        np.abs(ff) < 2e3
    )  # points to use to estimate interpolator

    Gi = _spline(ff[pu], G[pu], ff[po])
    G[po] = Gi  # replace interpolated

    # calculate
    if np.isscalar(f):
        idx = np.argmin(abs(ff - f))
    else:
        idx = [np.argmin(abs(ff - f0)) for f0 in f]
        idx = np.asarray(idx)

    # get actual absorption
    G = G[idx]

    return G


def _spline(x, y, xq):
    """
    Same as MATLAB cubic spline interpolation.
    """
    # interpolate
    cs = scipy.interpolate.InterpolatedUnivariateSpline(x, y)
    return cs(xq)
