"""
Code for reference (non-optimized) Bloch / Extended Phase Graphs simulation
to validate hybrid state model. Follows ISMRM Abstract by Malik et al.

In this package, we restrict to IR-prepared bSSFP and SSFP, including
slice profile, inversion efficiency and chemical exchange effects.
For the latter, we will restrict to two pools only, as for more component
estimation is ill-posed and computation gets intractable.

We neglect flow and diffusion phenomena, as:
    - flow: there are 2 main effects: in-flow effect for slice-selective
            excitation and phase effects for unbalanced gradient.

            in-flow: negligible for 3D imaging, only affect vessels in 2D
                     (we do not care for accurate quantification in vessels here).

            phase: negligible for bSSFP imaging, untractable or over-simplified
                   for SSFP (must know exact trajectory of spins to be meaningful).

    - diffusion: negligible for bSSFP; acceptable for common spoiler strengths
                 in WM and GM (mostly affect CSF [1], for which we do not care about
                               accurate quantification).

References:
    [1] Kobayashi Y, Terada Y. Diffusion-weighting Caused by Spoiler Gradients
        in the Fast Imaging with Steady-state Precession Sequence
        May Lead to Inaccurate T2 Measurements in MR Fingerprinting.
        Magn Reson Med Sci.
        2019 Jan 10;18(1):96-104.
        doi: 10.2463/mrms.tn.2018-0027.
        Epub 2018 May 24.
        PMID: 29794408;
        PMCID: PMC6326765.
"""
import numpy as np
import scipy.linalg


def bSSFPsim(sequence, spin):
    """
    run a bSSFP simulation
    """
    return bSSFPsimulator(sequence, spin, kmax=1).run()


def SSFPsim(sequence, spin, kmax=None):
    """
    run a SSFP simulation
    """
    return SSFPsimulator(sequence, spin, kmax).run()


def SPGRsim(sequence, spin, kmax=None):
    """
    run a SPGR simulation
    """
    return SPGRsimulator(sequence, spin, kmax).run()


def sequence_params(flip, TR, TI=18, TE=None, slice_profile=None):
    """
    prepare simulation sequence parameters.
    """
    return {
        "flip": flip * np.pi / 180,
        "slice_profile": slice_profile,
        "TI": TI,
        "TE": TE,
        "TR": TR,
    }


def spin_params(
    T1, T2, B1=1, inv_efficiency=1, R2p=0, df0=0, chemshift=0, weight=None, xrate=None
):
    """
    prepare simulation spun parameters.
    """
    return {
        "T1": T1,
        "T2": T2,
        "B1": B1,
        "inv_efficiency": inv_efficiency,
        "R2p": R2p,
        "df0": df0,
        "chemshit": chemshift,
        "weight": weight,
        "xrate": xrate,
    }


# %% common flip angle patterns
def constant_flip(flip, npulses):
    """
    constant flip angle train.
    """
    return flip * np.ones(npulses, dtype=np.float32)


def ramp_flip(flip, npulses):
    """
    ascending linear ramp flip angle train.
    """
    return np.linspace(0, flip, dtype=np.float32)


def piecewise_flip(flip, npulses):
    """
    ascending ramp flip angle train followed by
    descending ramp and constant train.
    """
    # divide in ramp up, ramp down and recovery
    nramp = int(np.ceil(npulses * 3 // 4))

    ramp_up = np.linspace(0, flip, int(nramp // 2), dtype=np.float32)
    ramp_down = np.linspace(flip, 5, int(nramp // 2), dtype=np.float32)
    const = 5 * np.ones(npulses - (len(ramp_up) + len(ramp_down)), dtype=np.float32)

    return np.concatenate((ramp_up, ramp_down, const))


# %% Utils
class abstractSimulator:
    def __init__(self, sequence, spin, kmax=None):
        # unpack sequence
        flip, slice_profile = sequence["flip"], sequence["slice_profile"]
        TI, TE, TR = sequence["TI"], sequence["TE"], sequence["TR"]

        # unpack spin
        T1, T2 = spin["T1"], spin["T2"]
        B1, inv_efficiency = spin["B1"], spin["inv_efficiency"]
        R2p, df0, chemshift = spin["R2p"], spin["df0"], spin["chemshift"]
        weight, xrate = spin["weight"], spin["xrate"]

        # get values
        npulses = len(flip)

        if kmax is None:
            kmax = npulses + 1

        # get correct formalism
        if xrate is None:
            epg = EPG

            # initialize configuration matrix
            omega = epg.set_configuration_matrix(kmax, weight)

            # prepare operators
            W = epg.set_configuration_weight(TE, 2 * np.pi * (df0 + chemshift), R2p)
            E1, E2, rE1 = epg.set_freeprecession(TR, T1, T2)
            E1inv, E2inv, rE1inv = epg.set_freeprecession(TI, T1, T2)

        else:
            epg = bmEPG
            xrate = epg.set_exchange_rate(xrate, weight)

            # initialize configuration matrix
            omega = epg.set_configuration_matrix(kmax, weight)

            # prepare operators
            W = epg.set_configuration_weight(TE, 2 * np.pi * df0, R2p)
            E1, E2, rE1 = epg.set_freeprecession(
                TR, T1, T2, 2 * np.pi * chemshift, xrate, weight
            )
            E1inv, E2inv, rE1inv = epg.set_freeprecession(
                TI, T1, T2, 2 * np.pi * chemshift, xrate, weight
            )

        # initialize signal
        sig = np.zeros(npulses, dtype=np.complex64)

        # assign
        self.epg = epg

        self.flip = B1 * flip
        self.slice_profile = slice_profile
        self.inv_efficiency = inv_efficiency

        self.W = W
        self.E1 = E1
        self.E2 = E2
        self.rE1 = rE1
        self.E1inv = E1inv
        self.E2inv = E2inv
        self.rE1inv = rE1inv

        self.omega = omega
        self.sig = sig

    def run(self):
        # unpack sequence
        flip, slice_profile = self.flip, self.slice_profile

        # unpack spin
        W = self.W
        E1, E2, rE1 = self.E1, self.E2, self.rE1
        E1inv, E2inv, rE1inv = self.E1inv, self.E2inv, self.rE1inv

        # get configuration matrix
        omega = self.omega

        # get signal
        sig = self.sig

        # sequence loop
        if slice_profile is not None:
            self.slice_selective_sequence(
                sig, omega, flip, slice_profile, W, E1, E2, rE1, E1inv, E2inv, rE1inv
            )

        else:
            self.nonselective_sequence(
                sig, omega, flip, slice_profile, W, E1, E2, rE1, E1inv, E2inv, rE1inv
            )

        return sig


class bSSFPsimulator(abstractSimulator):
    def slice_selective_sequence(
        self, sig, omega, flip, slice_profile, W, E1, E2, rE1, E1inv, E2inv, rE1inv
    ):
        # engine
        epg = self.epg

        # phase inc
        phi = np.zeros(len(flip))
        phi[::2] = np.pi

        # run inversion pulse
        omega[..., -1] *= -self.inv_efficiency
        omega = epg.apply_freeprecession(omega, E1inv, E2inv, rE1inv)
        omega[..., :2] = 0  # spoil

        # run sequence
        for B1 in slice_profile:
            for n in range(len(flip)):
                # prepare pulse
                RF = epg.set_rf_pulse(B1 * flip[n], phi[n])

                # apply pulse
                omega = epg.apply_rf_pulse(omega, RF)

                # record signal
                sig[n] += epg.sample_signal(omega, phi[n], W)

                # apply free precession
                omega = epg.apply_freeprecession(omega, E1, E2, rE1)

    def nonselective_sequence(
        self, sig, omega, flip, slice_profile, W, E1, E2, rE1, E1inv, E2inv, rE1inv
    ):
        # engine
        epg = self.epg

        # phase inc
        phi = np.zeros(len(flip))
        phi[::2] = np.pi

        # run sequence
        for rep in range(2):
            # run inversion pulse
            omega[..., -1] *= -self.inv_efficiency
            omega = epg.apply_freeprecession(omega, E1inv, E2inv, rE1inv)
            omega[..., :2] = 0  # spoil

            for n in range(len(flip)):
                # prepare pulse
                RF = epg.set_rf_pulse(flip[n], phi[n])

                # apply pulse
                omega = epg.apply_rf_pulse(omega, RF)

                # record signal
                sig[n] = epg.sample_signal(omega, phi[n], W)

                # apply free precession
                omega = epg.apply_freeprecession(omega, E1, E2, rE1)


class SSFPsimulator(abstractSimulator):
    def slice_selective_sequence(
        self, sig, omega, flip, slice_profile, W, E1, E2, rE1, E1inv, E2inv, rE1inv
    ):
        # engine
        epg = self.epg

        # phase inc
        phi = 0

        # run inversion pulse
        omega[..., -1] *= -self.inv_efficiency
        omega = epg.apply_freeprecession(omega, E1inv, E2inv, rE1inv)
        omega[..., :2] = 0  # spoil

        # run sequence
        for B1 in slice_profile:
            for n in range(len(flip)):
                # prepare pulse
                RF = epg.set_rf_pulse(B1 * flip[n], phi)

                # apply pulse
                omega = epg.apply_rf_pulse(omega, RF)

                # record signal
                sig[n] += epg.sample_signal(omega, phi, W)

                # apply free precession
                omega = epg.apply_freeprecession(omega, E1, E2, rE1)

                # apply gradient dephasing
                omega = epg.dephase_states(omega)

    def nonselective_sequence(
        self, sig, omega, flip, slice_profile, W, E1, E2, rE1, E1inv, E2inv, rE1inv
    ):
        # engine
        epg = self.epg

        # phase inc
        phi = 0

        # run sequence
        for rep in range(2):
            # run inversion pulse
            omega[..., -1] *= -self.inv_efficiency
            omega = epg.apply_freeprecession(omega, E1inv, E2inv, rE1inv)
            omega[..., :2] = 0  # spoil

            for n in range(len(flip)):
                # prepare pulse
                RF = epg.set_rf_pulse(flip[n], phi)

                # apply pulse
                omega = epg.apply_rf_pulse(omega, RF)

                # record signal
                sig[n] = epg.sample_signal(omega, phi, W)

                # apply free precession
                omega = epg.apply_freeprecession(omega, E1, E2, rE1)

                # apply gradient dephasing
                omega = epg.dephase_states(omega)


class SPGRsimulator(abstractSimulator):
    def slice_selective_sequence(
        self, sig, omega, flip, slice_profile, W, E1, E2, rE1, E1inv, E2inv, rE1inv
    ):
        # engine
        epg = self.epg

        # phase inc
        phase_inc = 117
        phi = np.zeros(len(flip), dtype=np.float32)
        for ph in range(len(flip)):
            phi[ph] = ph * (ph + 1) / 2 * phase_inc
        phi *= np.pi / 180

        # run inversion pulse
        omega[..., -1] *= -self.inv_efficiency
        omega = epg.apply_freeprecession(omega, E1inv, E2inv, rE1inv)
        omega[..., :2] = 0  # spoil

        # run sequence
        for B1 in slice_profile:
            for n in range(len(flip)):
                # prepare pulse
                RF = epg.set_rf_pulse(B1 * flip[n], phi[n])

                # apply pulse
                omega = epg.apply_rf_pulse(omega, RF)

                # record signal
                sig[n] += epg.sample_signal(omega, phi[n], W)

                # apply free precession
                omega = epg.apply_freeprecession(omega, E1, E2, rE1)

                # apply gradient dephasing
                omega = epg.dephase_states(omega)

    def nonselective_sequence(
        self, sig, omega, flip, slice_profile, W, E1, E2, rE1, E1inv, E2inv, rE1inv
    ):
        # engine
        epg = self.epg

        # phase inc
        phase_inc = 117
        phi = np.zeros(len(flip), dtype=np.float32)
        for ph in range(len(flip)):
            phi[ph] = ph * (ph + 1) / 2 * phase_inc

        # run sequence
        for rep in range(2):
            # run inversion pulse
            omega[..., -1] *= -self.inv_efficiency
            omega = epg.apply_freeprecession(omega, E1inv, E2inv, rE1inv)
            omega[..., :2] = 0  # spoil

            for n in range(len(flip)):
                # prepare pulse
                RF = epg.set_rf_pulse(flip[n], phi[n])

                # apply pulse
                omega = epg.apply_rf_pulse(omega, RF)

                # record signal
                sig[n] = epg.sample_signal(omega, phi[n], W)

                # apply free precession
                omega = epg.apply_freeprecession(omega, E1, E2, rE1)

                # apply gradient dephasing
                omega = epg.dephase_states(omega)


class EPG:
    """Regular EPG simulator."""

    @staticmethod
    def set_configuration_matrix(kmax):
        """
        Initialize empty configuration matrix.
        """
        omega = np.zeros((kmax, 3), dtype=np.complex64)
        omega[..., -1] = 1

        return omega

    @staticmethod
    def set_rf_pulse(alpha, phi):
        """
        Helper function to define EPG transition matrix
        As per Weigel et al JMR 2010 276-285
        """
        T = np.zeros((3, 3), dtype=np.complex64)
        T[0, 0] = np.cos(alpha / 2) ** 2
        T[0, 1] = np.exp(2 * 1j * phi) * (np.sin(alpha / 2)) ** 2
        T[0, 2] = -1j * np.exp(1j * phi) * np.sin(alpha)
        T[1, 0] = T[1, 0].conj()
        T[1, 1] = T[0, 0]
        T[1, 2] = 1j * np.exp(-1j * phi) * np.sin(alpha)
        T[2, 0] = -0.5 * 1j * np.exp(-1j * phi) * np.sin(alpha)
        T[2, 1] = 0.5 * 1j * np.exp(1j * phi) * np.sin(alpha)
        T[2, 2] = np.cos(alpha)

        return T

    @staticmethod
    def apply_rf_pulse(omega_in, T):
        """
        Apply rf pulse to configuration matrix.
        """
        # initialize omega matrix
        omega_out = np.zeros(omega_in.shape, omega_in.dtype)

        # apply pulse
        omega_out[..., 0] = (
            T[0, 0] * omega_in[..., 0]
            + T[0, 1] * omega_in[..., 1]
            + T[0, 2] * omega_in[..., 2]
        )
        omega_out[..., 1] = (
            T[1, 0] * omega_in[..., 0]
            + T[1, 1] * omega_in[..., 1]
            + T[1, 2] * omega_in[..., 2]
        )
        omega_out[..., 2] = (
            T[2, 0] * omega_in[..., 0]
            + T[2, 1] * omega_in[..., 1]
            + T[2, 2] * omega_in[..., 2]
        )

        return omega_out

    @staticmethod
    def set_freeprecession(t, T1, T2):
        """
        Longitudinal and Transverse relaxation operators.
        """
        E1 = np.exp(-t / T1)
        E2 = np.exp(-t / T2)
        rE1 = 1 - E1

        return E1, E2, rE1

    @staticmethod
    def apply_freeprecession(omega, E1, E2, rE1):
        """
        Apply Longitudinal and Transverse relaxations.
        """
        # relaxation
        omega[..., 0] *= E2
        omega[..., 1] *= E2.conj()
        omega[..., 2] *= E1

        # recovery
        omega[0, 2] += rE1

        return omega

    @staticmethod
    def dephase_states(omega):
        """
        Apply constant dephasing within TR (due to gradient moment + off resonance).
        """
        omega[0, :] = np.roll(omega[0, :], 1)  # shift Fp states
        omega[1, :] = np.roll(omega[1, :], -1)  # shift Fm states
        omega[1, -1] = 0  # Zero highest Fm state
        omega[0, 0] = omega[1, 0].conj()  # Fill in lowest Fp state

    @staticmethod
    def set_configuration_weight(t, df, R2p):
        """
        Apply complex z field (b0 + T2*) until TE.
        Assume sufficient spoil to account for 0th order only.
        """
        return np.exp((1j * df - R2p) * t)

    @staticmethod
    def sample_signal(omega, phi, W):
        """
        Demodulate rf pulse phase and record signal from F0 state.
        """
        return W * omega[0, 0] * np.exp(-1j * phi)


class bmEPG(EPG):
    """Bloch-McConnell EPG simulator."""

    @staticmethod
    def set_exchange_rate(k, weight):
        return np.array([k, k * (1 - weight) / weight])

    @staticmethod
    def set_configuration_matrix(kmax, weight):
        # initialize configuration
        omega = EPG.set_configuration_matrix(kmax)
        omega[0] *= 1 - weight
        omega[1] *= weight

        return omega

    @staticmethod
    def set_freeprecession(t, T1, T2, df0, k, weight):
        """
        Longitudinal and Transverse relaxation operators in presence of exchange.
        """
        # coefficients
        lambda0 = np.eye(2) * (-1 / T1 - k) + (1 - np.eye(2)) * k
        lambda1 = np.eye(2) * (-1 / T2 - k - 1j * df0) + (1 - np.eye(2)) * k
        C = np.array(((1 - weight), weight)) / T1

        # calculate operators
        E1 = scipy.linalg.expm(lambda0 * t)
        E2 = scipy.linalg.expm(lambda1 * t)
        rE1 = (E1 - np.eye(2)) @ np.linalg.solve(lambda0, C)

        return E1, E2, rE1
