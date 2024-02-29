"""Rigid motion parameter generation routines."""

__all__ = ["rigid_motion"]

import numpy as np
import numba as nb


def rigid_motion(ndims, nframes, degree="moderate", seed=42):
    """
    Generate rigid motion pattern as a Markov Chain process.

    Parameters
    ----------
    ndims : int
        Generate 2D (in-plane only) or 3D motion pattern.
    nframes : int
        Number of motion frames.
    degree : str | Iterable[float], optional
        Severity of motion. The default is ``"moderate"``.
    seed : int, optional
        Random number generator seed.
        The default is ``42``.

    Notes
    -----
    Severity of motion can be specified via the ``degree`` argument.
    This can be a string - accepted values are ``"subtle"``, ``"moderate"``
    and ``"severe"``. These corresponds to the following motion ranges:

    * ``"subtle"``: maximum rotation ``5.0 [deg]``; maximum translation ``2.0 [mm]``
    * ``"moderate"``: maximum rotation ``10.0 [deg]``; maximum translation ``8.0 [mm]``
    * ``"severe"``: maximum rotation ``16.0 [deg]``; maximum translation ``16.0 [mm]``

    As an alternative, user can specify a tuple of floats, where ``degree[0]``
    is the maximum rotation in ``[deg]`` and ``degree[1]`` is the maximum translation
    in ``[mm]``.

    Returns
    -------
    angleX : torch.Tensor
        Rotation about ``x`` axis in ``[deg]`` of shape ``(nframes,)``.
    angleY : torch.Tensor
        Rotation about ``y`` axis in ``[deg]`` of shape ``(nframes,)``.
    angleZ : torch.Tensor
        Rotation about ``z`` axis in ``[deg]`` of shape ``(nframes,)``.
    dx : torch.Tensor
        Translation towards ``x`` axis in ``[mm]`` of shape ``(nframes,)``.
    dy : torch.Tensor
        Translation towards ``y`` axis in ``[mm]`` of shape ``(nframes,)``.
    dz : torch.Tensor
        Translation towards ``z`` axis in ``[mm]`` of shape ``(nframes,)``.

    """
    # Markov rate (I don't remember what this is :()
    rate = [[0.9, 0.05, 0.05], [0.4, 0.3, 0.3], [0.4, 0.3, 0.3]]
    transition_mtx = np.array(rate, np.float32)

    # generate probability array
    np.random.seed(seed)
    change = np.random.rand(6, nframes)

    # generate six random series
    x = _generate_series(6, nframes, transition_mtx, change)

    # rescale series
    x_max = np.abs(x).max(axis=1)[:, None]
    x_max[x_max == 0] = 1
    x = x / x_max

    # get motion range
    if isinstance(degree, str):
        if degree == "subtle":
            degree = [5.0, 2.0]
        elif degree == "moderate":
            degree = [10.0, 8.0]
        elif degree == "severe":
            degree = [16.0, 16.0]
        else:
            raise ValueError(
                f"Severity of motion not recognized - must be either 'subtle', 'moderate', 'severe' or a (rotation, translation) tuple in (deg, mm). Found {degree}."
            )

    # set
    roll = degree[0] * x[0]  # deg, rotation around x
    pitch = degree[0] * x[1]  # deg, rotation around y
    yaw = degree[0] * x[2]  # deg, rotation around z
    dx = degree[1] * x[3]  # mm, translation around x
    dy = degree[1] * x[4]  # mm, translation around x
    dz = degree[1] * x[5]  # mm, translation around x

    if ndims == 2:
        return yaw, dy, dx
    elif ndims == 3:
        return roll, pitch, yaw, dx, dy, dz
    else:
        raise ValueError(f"Invalid number of dims! must be 2 or 3 - found {ndims}")


# %% local utils
# adapted from
# https://ipython-books.github.io/131-simulating-a-discrete-time-markov-chain/

# The statespace
states = np.array([0, -1, 1], np.int64)


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _generate_series(n_parameters, n_frames, transition_matrix, change):
    # pre-allocate state history
    state_history = np.zeros((n_parameters, n_frames), dtype=np.int64)

    # initialize states
    current_state = np.zeros(n_parameters, dtype=np.int64)

    for p in nb.prange(n_parameters):
        for t in range(1, n_frames):
            current_state[p] = _generate_state(
                transition_matrix[current_state[p]], change[p, t]
            )
            state_history[p, t] = state_history[p, t - 1] + states[current_state[p]]

    return state_history


@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _generate_state(probability, change):
    if change <= probability[0]:
        out_state = 0
    elif change <= probability[0] + probability[1]:
        out_state = 1
    else:
        out_state = 2

    return out_state
