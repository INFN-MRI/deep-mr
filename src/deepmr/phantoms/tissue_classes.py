"""Utils to generate MR parameters distributions and handle physical dimensions."""

__all__ = [
    "Air",
    "Fat",
    "WhiteMatter",
    "GrayMatter",
    "CSF",
    "Muscle",
    "Skin",
    "Bone",
    "Blood",
]

from dataclasses import dataclass, fields
from typing import Union

import numpy as np
import numpy.typing as npt

# reproducibility
np.random.seed(42)

# Speed of light [m/s].
c0 = 299792458.0

# Vacuum permeability [H/m].
mu0 = 4.0e-7 * np.pi

# Vacuum permeability [F/m].
eps0 = 1.0 / mu0 / c0**2

# gyromagnetic factors
gamma_bar = 42.575 * 1e6 # MHz / T -> Hz / T
gamma = 2 * np.pi * gamma_bar # rad / T / s

def _default_bm(n_atoms, model):
    if "bm" in model:
        z = np.zeros((n_atoms, 1), dtype=np.float32)
        return {
            "T1": z.copy(),
            "T2": z.copy(),
            "chemshift": z.copy(),
            "k": z.copy(),
            "weight": z.copy(),
        }
    else:
        return {}

def _default_mt(n_atoms, model):
    if "mt" in model:
        z = np.zeros((n_atoms, 1), dtype=np.float32)
        return {"k": z.copy(), "weight": z.copy()}
    else:
        return {}

@dataclass
class AbstractTissue:
    # electro-magnetic properties
    M0: float = 1.0
    chi: float = None
    sigma: float = None  # S / m
    epsilon: float = None

    # relaxation properties
    T1: Union[float, npt.NDArray] = None  # ms
    T2: Union[float, npt.NDArray] = None  # ms
    T2star: Union[float, npt.NDArray] = None  # ms

    # chemical properties
    chemshift: Union[float, npt.NDArray] = None  # Hz / T

    # motion properties
    D: Union[float, npt.NDArray] = None  # um**2 / ms
    v: Union[float, npt.NDArray] = None  # cm / s

    # smaller pools
    bm: dict = None
    mt: dict = None

    def __init__(self, n_atoms, model):
        for field in fields(self):
            value = getattr(self, field.name)
            # default values
            if (
                field.name != "bm"
                and field.name != "mt"
                # and field.name != "k"
                and value is None
            ):
                setattr(self, field.name, np.zeros(n_atoms, dtype=np.float32))
            elif (
                field.name != "bm"
                and field.name != "mt"
                # and field.name != "k"
            ):
                setattr(self, field.name, np.atleast_1d(value))
            elif field.name == "bm" and "bm" in model and value is None:
                setattr(self, field.name, _default_bm(n_atoms, model))
            elif field.name == "mt" and "mt" in model and value is None:
                setattr(self, field.name, _default_mt(n_atoms, model))


    # utils
    def _calculate_t1(self, B0, A):
        return A * B0 ** (1 / 3)

    def _calculate_t2star(self, B0, T2, susceptibility):
        if susceptibility:
            R2 = 1 / (T2 * 1e-3)  # 1 / ms -> 1 / s
            R2prime = gamma * np.abs(gamma * B0 * susceptibility)  # rad / s
            R2star = R2 + R2prime
            return 1e3 / R2star  # s -> ms

        return None

class Air(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate Air parameters.

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
        """
        # set electro-magnetic properties
        self.M0 = 0.0
        self.chi = 0.4e-6
        self.sigma = 10e-12
        self.epsilon = 8.85e-12

        # set T1
        self.T1 = np.zeros(n_atoms, dtype=np.float32)

        # set T2
        self.T2 = np.zeros(n_atoms, dtype=np.float32)

        super().__init__(n_atoms, model)

class Fat(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate Fat parameters

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
        """
        # set electro-magnetic properties
        self.chi = -7.79e-6
        self.sigma = 0.54
        self.epsilon = 62

        # set T1
        A0 = 194  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310288/table/T2/?report=objectonly
        self.T1 = rand(A0, 0.1 * A0, n_atoms)

        # set T2
        T20 = 54  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3310288/table/T2/?report=objectonly
        self.T2 = rand(T20, 0.1 * T20, n_atoms)

        # set T2*
        self.T2star = super()._calculate_t2star(
            B0, T20, rand(self.chi, 0.1 * self.chi, n_atoms)
        )

        # set chemical_shift
        chemshift = 3.28  # Hz / T
        self.chemshift = gamma_bar * B0 * chemshift  # Hz

        super().__init__(n_atoms, model)

class WhiteMatter(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate White Matter parameters.

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
            model (str, optional): MR signal model (default is "single"). Options are:
                - "single": Single pool T1 / T2.
                - "bm": 2-pool model with exchange (iew pool + mw pool).
                - "mt": 2-pool model with exchange (iew/mw pool + Zeeman semisolid pool).
                - "bm-mt": 3-pool model (iew pool, mw pool, Zeeman semisolid pool; mw exchange with iew and ss).

        """
        # set model
        if model == "single":
            simulate_multipool = False
        else:
            simulate_multipool = model

        # set electro-magnetic properties
        self.M0 = 0.77
        self.chi = -8.8e-6
        self.sigma = _get_complex_dielectric_properties(B0)["conductivity"]["wm"]
        self.epsilon = _get_complex_dielectric_properties(B0)["permittivity"]["wm"]
        self.chemshift = 0.0

        # set water T1 and T2
        if simulate_multipool:
            # intra-/extra-cellular water # from https://onlinelibrary.wiley.com/doi/10.1002/mrm.22131
            A0 = 725
            A = rand(A0, 0.3 * A0, n_atoms)
            self.T1 = super()._calculate_t1(B0, A)
            T20 = 80
            self.T2 = rand(T20, 0.3 * T20, n_atoms)
        else:
            A0 = 611
            A = rand(A0, 0.3 * A0, n_atoms)
            self.T1 = super()._calculate_t1(B0, A)
            T20 = 70
            self.T2 = rand(T20, 0.3 * T20, n_atoms)

        # set water T2* (then assume same T2' for all pools)
        self.T2star = super()._calculate_t2star(
            B0, self.T2, rand(self.chi, 0.1 * self.chi, n_atoms)
        )

        # set diffusion (assume the same for all pools)
        d0 = 0.69  # https://www.sciencedirect.com/science/article/pii/S1090780718301228
        self.D = rand(d0, 0.1 * d0, n_atoms)

        # set Bloch-McConnell pool T1 and T2
        if "bm" in str(simulate_multipool).lower():
            self.bm = {}
            # myelin water # from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7478173/table/T1/?report=objectonly
            A0 = 275
            A = rand(A0, 0.3 * A0, n_atoms)
            self.bm["T1"] = np.atleast_1d(super()._calculate_t1(B0, A))[..., None]
            # myelin water # from https://onlinelibrary.wiley.com/doi/10.1002/mrm.22131
            T20 = 10
            self.bm["T2"] = np.atleast_1d(rand(T20, 0.3 * T20, n_atoms))[..., None]
            self.bm["chemshift"] = (
                15.0 / 3.0 * B0 * np.ones((n_atoms, 1), dtype=np.float32)
            )

        # set Bloch-McConnell pool weight
        if "bm" in str(simulate_multipool).lower():
            w0 = 0.15
            wmw = rand(w0, 0.9 * w0, n_atoms)
            self.bm["weight"] = wmw * np.ones(
                (n_atoms, 1), dtype=np.float32
            )  # myelin water
            
        # set semisolid pool weight
        if "mt" in str(simulate_multipool).lower():
            self.mt = {}
            w0 = 0.1
            wss = rand(w0, 0.9 * w0, n_atoms)
            self.mt["weight"] = wss * np.ones((n_atoms, 1), dtype=np.float32)

        # set Bloch-McConnell pool exchange
        if "bm" in str(simulate_multipool).lower():
            kmw_iew = 13.3
            kmw_iew = np.atleast_1d(rand(kmw_iew, 0.5 * kmw_iew, n_atoms))
            self.bm["k"] = np.atleast_1d(rand(kmw_iew, 0.5 * kmw_iew, n_atoms))[:, None]
        # set semisolid pool exchange
        if "mt" in str(simulate_multipool).lower():
            kmw_ss = 200
            self.mt["k"] = np.atleast_1d(rand(kmw_ss, 0.5 * kmw_ss, n_atoms))[:, None]

        super().__init__(n_atoms, model)

class GrayMatter(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate Gray Matter parameters.

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
            model:  signal model to be used:
                    - "single": single pool T1 / T2.
                    - "bm": 2-pool model with exchange (iew pool + mw pool).
                    - "mt": 2-pool model with exchange( iew/mw pool + zeeman semisolid pool)
                    - "bm-mt": 3-pool model (iew pool, mw pool, zeeman semisolid pool; mw exchange with iew and ss).
        """
        # set model
        if model == "single":
            simulate_multipool = False
        else:
            simulate_multipool = model

        # set electro-magnetic properties
        self.M0 = 0.87
        self.chi = -8.8e-6
        self.sigma = _get_complex_dielectric_properties(B0)["conductivity"]["gm"]
        self.epsilon = _get_complex_dielectric_properties(B0)["permittivity"]["gm"]
        self.chemshift = 0.0

        # set water T1 and T2
        if simulate_multipool:
            # intra-/extra-cellular water # from https://onlinelibrary.wiley.com/doi/10.1002/mrm.22131
            A0 = 800
            A = rand(A0, 0.3 * A0, n_atoms)
            self.T1 = super()._calculate_t1(B0, A)
            T20 = 90
            self.T2 = rand(T20, 0.3 * T20, n_atoms)
        else:
            A0 = 1025
            A = rand(A0, 0.3 * A0, n_atoms)
            self.T1 = super()._calculate_t1(B0, A)
            T20 = 83
            self.T2 = rand(T20, 0.3 * T20, n_atoms)

        # set water T2* (then assume same T2' for all pools)
        self.T2star = super()._calculate_t2star(
            B0, self.T2, rand(self.chi, 0.1 * self.chi, n_atoms)
        )

        # set diffusion (assume the same for all pools)
        d0 = 0.83  # https://www.sciencedirect.com/science/article/pii/S1090780718301228
        self.D = rand(d0, 0.1 * d0, n_atoms)

        # set Bloch-McConnell pool T1 and T2
        if "bm" in str(simulate_multipool).lower():
            self.bm = {}
            # myelin water # from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7478173/table/T1/?report=objectonly
            A0 = 300
            A = rand(A0, 0.3 * A0, n_atoms)
            self.bm["T1"] = np.atleast_1d(super()._calculate_t1(B0, A))[..., None]
            # myelin water # from https://onlinelibrary.wiley.com/doi/10.1002/mrm.22131
            T20 = 20
            self.bm["T2"] = np.atleast_1d(rand(T20, 0.3 * T20, n_atoms))[..., None]
            self.bm["chemshift"] = (
                5.0 / 3.0 * B0 * np.ones((n_atoms, 1), dtype=np.float32)
            )
       
        # set Bloch-McConnell pool weight
        if "bm" in str(simulate_multipool).lower():
            w0 = 0.03
            wmw = rand(w0, 0.9 * w0, n_atoms)
            self.bm["weight"] = wmw * np.ones(
                (n_atoms, 1), dtype=np.float32
            )  # myelin water
            
        # set semisolid pool weight
        if "mt" in str(simulate_multipool).lower():
            self.mt = {}
            w0 = 0.03
            wss = rand(w0, 0.9 * w0, n_atoms)
            self.mt["weight"] = wss * np.ones((n_atoms, 1), dtype=np.float32)

        # set Bloch-McConnell pool exchange
        if "bm" in str(simulate_multipool).lower():
            kmw_iew = 53.2
            kmw_iew = np.atleast_1d(rand(kmw_iew, 0.5 * kmw_iew, n_atoms))
            self.bm["k"] = np.atleast_1d(rand(kmw_iew, 0.5 * kmw_iew, n_atoms))[:, None]
        # set semisolid pool exchange
        if "mt" in str(simulate_multipool).lower():
            kmw_ss = 3333.3
            self.mt["k"] = np.atleast_1d(rand(kmw_ss, 0.5 * kmw_ss, n_atoms))[:, None]

        super().__init__(n_atoms, model)


class CSF(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate CSF parameters.

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
        """
        # set electro-magnetic properties
        self.chi = -9.04e-6
        self.sigma = _get_complex_dielectric_properties(B0)["conductivity"]["csf"]
        self.epsilon = _get_complex_dielectric_properties(B0)["permittivity"]["csf"]

        # set T1
        A0 = 2244
        A = rand(A0, 0.1 * A0, n_atoms)
        self.T1 = super()._calculate_t1(B0, A)

        # set T2
        T20 = 329
        self.T2 = rand(T20, 0.1 * T20, n_atoms)

        # set T2*
        self.T2star = super()._calculate_t2star(
            B0, self.T2, rand(self.chi, 0.1 * self.chi, n_atoms)
        )

        # set diffusion
        d0 = 3.19
        self.D = rand(d0, 0.1 * d0, n_atoms)

        # set velocity
        v0 = 5.0
        self.v = rand(v0, v0, n_atoms)

        super().__init__(n_atoms, model)

class Muscle(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate Muscle parameters.

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
        """
        # simulate electro-magnetic properties
        self.chi = -9.04e-6

        # https://www.researchgate.net/publication/51820086_Local_tissue_temperature_increase_of_a_generic_implant_compared_to_the_basic_restrictions_defined_in_safety_guidelines/figures?lo=1
        self.sigma = 0.71
        self.epsilon = 66

        # set T1
        A0 = 786
        A = rand(A0, 0.1 * A0, n_atoms)
        self.T1 = super()._calculate_t1(B0, A)

        # set T2
        T20 = 50
        self.T2 = rand(T20, 0.1 * T20, n_atoms)

        # set T2*
        self.T2star = super()._calculate_t2star(
            B0, self.T2, rand(self.chi, 0.1 * self.chi, n_atoms)
        )

        super().__init__(n_atoms, model)

class Skin(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate Skin parameters.

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
        """
        # simulate electro-magnetic properties
        self.chi = -9.04e-6
        self.sigma = 0.54
        self.epsilon = 62

        # set T1
        A0 = 786
        A = rand(A0, 0.1 * A0, n_atoms)
        self.T1 = super()._calculate_t1(B0, A)

        # set T2
        T20 = 50
        self.T2 = rand(T20, 0.1 * T20, n_atoms)

        # set T2*
        self.T2star = super()._calculate_t2star(
            B0, self.T2, rand(self.chi, 0.1 * self.chi, n_atoms)
        )

        super().__init__(n_atoms, model)

class Bone(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate Bone parameters.

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
        """
        self.chi = -8.44e-6
        self.sigma = 0.12
        self.epsilon = 21

        # set T1
        A0 = 434
        A = rand(A0, 0.1 * A0, n_atoms)
        self.T1 = super()._calculate_t1(B0, A)

        # set T2
        T20 = 1
        self.T2 = rand(T20, 0.1 * T20, n_atoms)

        # set T2*
        self.T2star = super()._calculate_t2star(
            B0, self.T2, rand(self.chi, 0.1 * self.chi, n_atoms)
        )

        super().__init__(n_atoms, model)

class Blood(AbstractTissue):
    def __init__(self, n_atoms, B0, model="single"):
        """
        Generate blood parameters.

        Args:
            n_atoms: number of entries to be simulated (assume uniform distribution).
            B0: static field strength in units of [T].
        """
        # set electro-magnetic properties
        self.chi = -9.12e-6  # Oxygenated blood
        self.sigma = 2.14
        self.epsilon = 84

        # set T1
        A0 = 1205
        A = rand(A0, 0.1 * A0, n_atoms)
        self.T1 = super()._calculate_t1(B0, A)

        # set T2
        T20 = 275
        self.T2 = rand(T20, 0.1 * T20, n_atoms)

        # set T2*
        self.T2star = super()._calculate_t2star(
            B0, self.T2, rand(self.chi, 0.1 * self.chi, n_atoms)
        )

        # set velocity
        v0 = 50.0
        self.v = rand(v0, v0, n_atoms)

        super().__init__(n_atoms, model)

# %% local utils
def rand(mean, hwidth, n_atoms, seed=42):
    """
    Generate uniformly distributed random numbers.

    Args:
        mean: average value of the distribution.
        hwidth: half-width of the distribution.
        n_atoms: number of samples to be generated. If equals to 1, returns mean of the distribution.
    """
    # calculate lower and upper bound
    lower_bound = mean - hwidth
    upper_bound = mean + hwidth

    # distribution case
    if n_atoms > 1:
        np.random_seed(seed)
        return (upper_bound - lower_bound) * np.rand(n_atoms) + lower_bound

    # delta peak case
    return mean

# 4th order Cole-Cole model parameters (N De Geeter et al 2012 Phys. Med. Biol. 57 2169)
brain_params = {
    "wm": {
        "epsInf": 4.0,
        "deps": np.array([32.0, 100.0, 4e4, 3.5e7]),
        "tau": np.array([7.96e-12, 7.96e-9, 53.05e-6, 7.958e-3]),
        "alpha": np.array([0.10, 0.10, 0.30, 0.02]),
        "sigma": 0.02,
    },
    "gm": {
        "epsInf": 4.0,
        "deps": np.array([45.0, 400.0, 2e5, 4.5e7]),
        "tau": np.array([7.96e-12, 15.92e-9, 106.10e-6, 5.305e-3]),
        "alpha": np.array([0.10, 0.15, 0.22, 0.00]),
        "sigma": 0.02,
    },
    "csf": {
        "epsInf": 4.0,
        "deps": np.array([65.0, 40.0, 00.0, 0.0]),
        "tau": np.array([7.958e-12, 1.592e-9, 0.0, 0.0]),
        "alpha": np.array([0.10, 0.0, 0.0, 0.0]),
        "sigma": 2.0,
    },
}

cole_cole_model_params = {"brain": brain_params}

def _get_complex_dielectric_properties(field_strength, anatomic_region="brain"):
    """Calculate theoretical complex dielectric properties.

    Assume 4 Cole-Cole model for dielectric properties.

    Args:
        field_strength (float): B0 field strength (units: [T])..
        anatomic_region (str): anatomy of interest (default: brain).

    Returns:
        dielectric_properties (dict): dictionary with the following fields:

                                        - conductivity: representative conductivity
                                                        values for the selected
                                                        anatomy and field strength
                                                        (units: [S/m]).

                                        - permittivity: representative permittivity
                                                        values for the selected
                                                        anatomy and field strength.
    """
    # get Larmor frequency [rad/s]
    omega = 2 * np.pi * field_strength * gamma_bar * 1e6

    # initialize output dict
    dielectric_properties = {"permittivity": {}, "conductivity": {}}

    # get params
    try:
        params = cole_cole_model_params[anatomic_region]
    except:
        print("Not implemented!")

    # loop over representative tissues
    for tissue in params.keys():
        tmp = params[tissue]
        epsInf = tmp["epsInf"]
        eps = tmp["deps"] / (1 + (1j * omega * tmp["tau"]) ** (1 - tmp["alpha"]))
        sigma = tmp["sigma"] / (1j * omega * eps0)

        # get complex electical properties
        complex_ep = epsInf + eps.sum() + sigma

        # assign
        dielectric_properties["permittivity"][tissue] = complex_ep.real
        dielectric_properties["conductivity"][tissue] = -complex_ep.imag * omega * eps0

    return dielectric_properties