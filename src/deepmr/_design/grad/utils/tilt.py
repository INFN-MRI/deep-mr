"""Utils for waveform rotation."""

__all__ = ["make_tilt", "broadcast_tilt", "projection", "angleaxis2rotmat", "tilt_increment"]

import numpy as np

def make_tilt(tilt_type, nshots, accel=1, nframes=1, nechoes=1, tilt_echoes=False, dummy=False):
    """
    Generate list of tilt angles.

    Args:
        tilt (str, float): Name of the tilt or tilt increment in [rad].
        nshots (int): Number of shots to fully sample a plane.
        accel (int): In-plane acceleration factor (max to nshots).
        nframes (int): Number of frames for k-t acquisitions.
        nechoes (int): number of echoes.
        tilt_echoes (bool): If True, rotate across echoes, otherwise keep same 
            trajectory for each echo (default to False).

    Returns:
        (array): in-plane tilt angle.

    Raises:
        NotImplementedError: If the tilt name is unknown.

    Notes:
        The following values are accepted for the tilt name, with :math:`N` the number of
        partitions:

        - "none": no tilt
        - "uniform": uniform tilt: 2:math:`\pi / N`
        - "intergaps": :math:`\pi/2N`
        - "inverted": inverted tilt :math:`\pi/N + \pi`
        - "golden": tilt of the golden angle :math:`\pi(3-\sqrt{5})`
        - "tiny-golden": tilt of the tiny golden angle 2:math:`\pi(15-\sqrt{5})`
        - "mri-golden": tilt of the golden angle used in MRI :math:`\pi(\sqrt{5}-1)/2`

    """   
    # initialize readout tilt
    if "golden" in tilt_type or tilt_type == "tgas":
        # initialize increments
        increment = tilt_increment(tilt_type, int(nshots / accel) * nframes)
    else:
        # initialize increments
        increment = tilt_increment(tilt_type, min(int(nshots / accel) * nframes, nshots))
    
    # get angles
    if tilt_echoes:
        angles = tilt_angles(increment, int(nshots / accel) * nframes * nechoes, dummy)
    else:
        angles = tilt_angles((increment, 0.0), (int(nshots / accel) * nframes, nechoes), dummy)
    
    return angles.sum(axis=0) % (2 * np.pi)


def broadcast_tilt(new_arr, *input, expand_input=False, loop_order="new-first"):
    """
    Broadcast list of tilt angles.

    Args:
        new_arr (array): new array to be broadcasted
        input (array): input array(s) to be broadcasted.
        expand_input (bool): if True, assume "input" already account for len(new_arr) repetitions.
        loop_order (str): new array in outer or inner loop.

    Returns:
        (array): (broadcasted) version of 'input' array (along outer axis).
        (array): broadcasted version of 'new_array' (along outer axis).

    """
    assert loop_order in ["new-first", "old-first"], f"Error! Valid loop_order are 'new-first' and 'old-first' (found {loop_order})."
    
    # parse sizes of a and b           
    asize = input[0].shape[0]
    bsize = new_arr.shape[0]
    
    # perform expansion
    if expand_input is False:   
        # get output dim
        osize = asize
        
        # expand dims              
        if loop_order == "new-first":
            new_arr = np.apply_along_axis(np.tile, 0, new_arr, int(osize // bsize))
        else:
            new_arr = np.repeat(new_arr, int(osize // bsize), axis=0)
    else:
        # get output dim
        osize = asize * bsize
        
        # expand dims              
        if loop_order == "new-first":
            new_arr = np.apply_along_axis(np.tile, 0, new_arr, asize)
            input = [np.repeat(el, bsize, axis=1) for el in input]
        else:
            new_arr = np.repeat(new_arr, asize, axis=0)
            input = [np.apply_along_axis(np.tile, 0, el, bsize) for el in input]
            
    # return
    return [new_arr] + list(input)


def tilt_increment(tilt, nb_partitions=1, halfplane=False):
    r"""
    Initialize the tilt angle.

    Args:
        tilt (str, float): Name of the tilt.
        nb_partitions (int, optional): Number of partitions. The default is 1.

    Returns:
        (float): Tilt angle increment in rad.

    Raises:
        NotImplementedError: If the tilt name is unknown.

    Notes:
        The following values are accepted for the tilt name, with :math:`N` the number of
        partitions:

        - "none": no tilt
        - "uniform": uniform tilt: 2:math:`\pi / N`
        - "intergaps": :math:`\pi/2N`
        - "inverted": inverted tilt :math:`\pi/N + \pi`
        - "golden": tilt of the golden angle :math:`\pi(3-\sqrt{5})`
        - "tiny-golden": tilt of the tiny golden angle 2:math:`\pi(15-\sqrt{5})`
        - "mri-golden": tilt of the golden angle used in MRI :math:`\pi(\sqrt{5}-1)/2`

    """
    if halfplane:
        fufa = 0.5
    else:
        fufa = 1.0
    if tilt is None:
        tilt = "none"
    if not isinstance(tilt, str):
        return tilt
    elif tilt == "none":
        return 0
    elif tilt == "uniform":
        return 2 * np.pi / nb_partitions * fufa
    elif tilt == "intergaps":
        return np.pi / nb_partitions / 2 * fufa
    elif tilt == "inverted":
        return np.pi / nb_partitions * fufa + np.pi
    elif tilt == "golden":
        return np.pi * (3 - np.sqrt(5))
    elif tilt == "tiny-golden" or tilt == "tgas":
        return np.deg2rad(23.63)
    elif tilt == "mri-golden":
        return np.pi * (np.sqrt(5) - 1) / 2
    else:
        raise NotImplementedError(f"Unknown tilt name: {tilt}")


def tilt_angles(increments, nincrements, dummy=False):
    """
    Create list of tilt angles.

    Args:
        increments (list, tuple): tilt increment for each axis.
        nincrements (list, tuple): number of increments for each axis.

    Returns:
        np.ndarray: increment for each excitation.

    """
    # adjust scalars
    if np.isscalar(increments):
        increments = [increments]
    if np.isscalar(nincrements):
        nincrements = [nincrements]

    # count axis
    naxis = len(increments)

    # create increment array for each axis
    increments_ary = [np.arange(nincrements[n]) * increments[n] for n in range(naxis)]

    # append null increment along readouts if required (for steady-state ?)
    if dummy:
        increments_ary[0] = np.concatenate((np.asarray([0]), increments_ary[0]))

    # build grid
    output_increments = np.meshgrid(*increments_ary, indexing="ij")
    output_increments = [ary.flatten() for ary in output_increments]

    # return
    return np.stack(output_increments, axis=0)

def projection(k, R):
    """
    Create a 2/3D ____ projection trajectory.

    Use a single interleaf of a 2D trajectory as basis.

    Args:
        k (array): ND array in [2 x Nt]. Will be first projection.
        phi (array): in-plane rotation (around z axis), units: [rad]
        theta (array): plane rotation (around y axis), units: [rad]
    """
    ndim = k.shape[0]
    if ndim == 2: # expand to 3D assuming we are in the xy plane
        k = np.stack((k[0], k[1], 0 * k[1]), axis=0)
    kout = np.einsum("bij,j...->bi...", R, k)
    kout = kout.swapaxes(0, 1)
    return kout[:ndim]
    # if theta is None:
    #     kout = _2d_rotation(k, phi)
    # else: # rotate around y
    #     kout = _3d_rotation(k, phi, theta)

    # return kout

#%% local utils
# def _2d_rotation(input, phi):
#     phi = phi[:, None]
#     output = np.zeros(
#         (input.shape[0], phi.shape[0], input.shape[-1]), dtype=input.dtype
#     )
#     output[0] = input[0] * np.cos(phi) - input[1] * np.sin(phi)
#     output[1] = input[0] * np.sin(phi) + input[1] * np.cos(phi)

#     return output


# def _3d_rotation(input, phi, theta):
#     phi = phi[:, None]
#     theta = theta[:, None]
    
#     output = np.zeros(
#         (input.shape[0], phi.shape[0], input.shape[-1]), dtype=input.dtype
#     )
#     output[0] = (
#         input[0] * (np.cos(theta) * np.cos(phi))
#         + input[1] * np.sin(phi)
#         - input[2] * (np.sin(theta) * np.cos(phi))
#     )
#     output[1] = (
#         -input[0] * (np.cos(theta) * np.sin(phi))
#         + input[1] * np.cos(phi)
#         + input[2] * (np.sin(theta) * np.sin(phi))
#     )
#     output[2] = input[0] * np.sin(theta) + input[2] * np.cos(theta)

#     return output

def angleaxis2rotmat(alpha, u):
    """
    % function R = angleaxis2rotmat(theta, n)
    %
    % Inputs
    %  theta   [1 1]            Rotation angle (radians)
    %  n       [3 1] or [1 3]   Vector lying along the axis of rotation
    %
    % Output
    %  R       [3 3] rotation matrix
    %
    % Useful for, e.g., applying "in-plane" rotations for multi-shot spiral imaging.
    %
    % From https://www.mathworks.com/matlabcentral/fileexchange/66446-rotation-matrix?s_tid=mwa_osa_a
    % Accessed 22-Sep-2019
    
    % RotMatrix - N-dimensional Rotation matrix
    % R = RotMatrix(alpha, u, v)
    % INPUT:
    %   alpha: Angle of rotation in radians, counter-clockwise direction.
    %   u, v:  Ignored for the 2D case.
    %          For the 3D case, u is the vector to rotate around.
    %          For the N-D case, there is no unique axis of rotation anymore, so 2
    %          orthonormal vectors u and v are used to define the (N-1) dimensional
    %          hyperplane to rotate in.
    %          u and v are normalized automatically and in the N-D case it is cared
    %          for u and v being orthogonal.
    % OUTPUT:
    %   R:     Rotation matrix.
    %          If the u (and/or v) is zero, or u and v are collinear, The rotation
    %          matrix contains NaNs.
    %
    % REFERENCES:
    % analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
    % en.wikipedia.org/wiki/Rotation_matrix
    % application.wiley-vch.de/books/sample/3527406204_c01.pdf
    %
    % Tested: Matlab 7.7, 7.8, 7.13, 9.1, WinXP/32, Win7/64
    % Author: Jan Simon, Heidelberg, (C) 2018 matlab.2010(a)n(MINUS)simon.de
    
    % $JRev: R-b V:001 Sum:GwB8LUFcZ+7i Date:10-Mar-2018 19:26:01 $
    % $License: BSD (use/copy/change/redistribute on own risk, mention the author) $
    % $File: Tools\GL3D\RotMatrix.m $
    % History:
    % 001: 10-Mar-2018 14:31, First version.
    """
    # Do the work: =================================================================
    s = np.sin(alpha)
    c = np.cos(alpha)
    
    # Normalized vector:
    u = np.asarray(u, dtype=np.float32)
    u = u / np.sqrt(u.T @ u)
          
    # 3D rotation matrix:
    x  = u[0]
    y  = u[1]
    z  = u[2]
    mc = 1 - c
    
    # build rows
    R0 = np.stack((c + x * x * mc, x * y * mc - z * s, x * z * mc + y * s), axis=-1) # (nalpha, 3)
    R1 = np.stack((x * y * mc + z * s,  c + y * y * mc, y * z * mc - x * s), axis=-1) # (nalpha, 3)
    R2 = np.stack((x * z * mc - y * s,  y * z * mc + x * s,  c + z * z * mc), axis=-1) # (nalpha, 3)

    # stack rows
    R = np.stack((R0, R1, R2), axis=1)
             
    return R