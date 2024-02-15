"""Global configuration."""

__all__ = ["ss_globals"]

from types import SimpleNamespace


def ss_globals():
    out = SimpleNamespace()

    # Gamma definitions -- all in Hz/G
    out.SS_GAMMA_HYDROGEN = 4257.6
    out.SS_GAMMA_LITHIUM = 1654.6
    out.SS_GAMMA_CARBON = 1070.5
    out.SS_GAMMA_SODIUM = 1126.2
    out.SS_GAMMA_PHOSPHOROUS = 1723.5

    # Nucleus info
    out.SS_NUCLEUS = "Hydrogen"
    out.SS_GAMMA = out.SS_GAMMA_HYDROGEN

    # Gradient/timing parameters
    out.SS_MXG = 5.0  # G/cm
    out.SS_MXS = 20  # G/cm/ms
    out.SS_TS = 4e-6  # Sampling time (s)

    # RF parameters
    out.SS_MAX_B1 = 0.2  # Gauss
    out.SS_MAX_DURATION = 20e-3  # Max allowed duration

    # Design tolerances, parameters
    out.SS_NUM_LOBE_ITERS = 10
    out.SS_EQUAL_LOBES = 0
    out.SS_VERSE_FRAC = 0.8
    out.SS_NUM_FS_TEST = 100
    out.SS_SPECT_CORRECT_FLAG = 0
    out.SS_SPECT_CORRECT_REGULARIZATION = 0
    out.SS_SLR_FLAG = 0
    out.SS_MIN_ORDER = 1
    out.SS_VERSE_B1 = 0
    out.SS_SLEW_PENALTY = 0

    return out
