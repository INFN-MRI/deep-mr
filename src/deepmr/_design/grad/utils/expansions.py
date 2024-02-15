"""Expansions routines."""

__all__ = ["make_stack", "make_multiecho", "make_projection"]

import numpy as np

from .dcf import angular_compensation
from .misc import scale_traj
from .tilt import angleaxis2rotmat
from .tilt import make_tilt
from .tilt import projection
from .tilt import tilt_increment

def make_projection(ordering, traj, shape, accel, rotargs):    
    """
    """
    # dummy
    dummy = rotargs["dummy"]
    
    # parse sizes
    nintl, nplanes, nechoes, nframes = shape
        
    # parse accel
    Rplane, Rangular = accel
        
    # rotation angles and shapes
    phi = make_tilt(rotargs["tilt_type"][0], nintl + int(dummy), Rplane, nframes, 1, False)
    nphi = phi.shape[0]
    if dummy:
        phi[:nframes] = 0.0
    if "shuffl" in ordering:
        nreps = int(nphi / nframes * nplanes / Rangular)
        frame = np.repeat(np.arange(nframes), nreps, axis=0)
        rep = np.tile(np.arange(nreps), nframes)
        increment = tilt_increment(rotargs["tilt_type"][1], nreps * nframes)
        theta = increment * (frame + rep)
    else:
        theta = make_tilt(rotargs["tilt_type"][1], nplanes, Rangular, nframes, 1, False)    
    ntheta = theta.shape[0]

    # expand
    if ordering == "sequential": # all the planes, then move to next interleave
        theta = np.tile(theta, int(nphi / nframes))
        theta = np.repeat(theta, nechoes, axis=0)
        phi = np.repeat(phi, int(ntheta / nframes), axis=0)
        phi = np.repeat(phi, nechoes, axis=0)
    elif ordering == "interleaved": # all the interleaves, then move to next plane
        theta = np.repeat(theta, int(nphi / nframes), axis=0)
        theta = np.repeat(theta, nechoes, axis=0)
        phi = np.tile(phi, int(ntheta / nframes))
        phi = np.repeat(phi, nechoes, axis=0)
    elif "shuffl" in ordering: # planes have the correct length already, expand interleave rotation
        phi = np.repeat(phi, int(ntheta / nphi), axis=0)
        phi = np.repeat(phi, nechoes, axis=0)

    # parse base coord
    kr = traj["kr"]
    acs_kr = traj["acs"]["kr"]
    
    # expand for 3D rot about x axis
    kr = np.stack((kr[0], kr[1], 0 * kr[1]), axis=0)
    acs_kr = np.stack((acs_kr[0], acs_kr[1], 0 * acs_kr[1]), axis=0)
    
    # perform rotation
    axis = np.zeros_like(theta) # rotation axis
    Rx = angleaxis2rotmat(theta, [1, 0, 0]) # whole-plane rotation about x
    if "multiaxis" in ordering:
        Ry = angleaxis2rotmat(theta, [0, 1, 0]) # whole-plane rotation about y
        R0 = angleaxis2rotmat(0.5 * np.pi * np.ones_like(theta), [0, 1, 0])
        Rz = np.einsum('...ij,...jk->...ik', Rx, R0) # bring plane on y-z, then whole-plane rotation about z
        Rx = np.concatenate((Rx, Ry, Rz), axis=0) # whole-plane rotation
        axis = np.concatenate((axis, axis+1, axis+2), axis=0)
        
    Rz = angleaxis2rotmat(phi, [0, 0, 1]) # in-plane rotation about z
    if "multiaxis" in ordering:
        Rz = np.concatenate((Rz, Rz, Rz), axis=0)
    # put together full rotation matrix
    R = np.einsum('...ij,...jk->...ik', Rx, Rz)
    
    # rotate
    kr = projection(kr, R).astype(np.float32).transpose(1, 2, 0)
    acs_kr = projection(acs_kr, R).astype(np.float32).transpose(1, 2, 0)

    # compute dcf - > angular component
    dcf = traj["dcf"]
    dcf = angular_compensation(dcf, kr, axis)
        
    # put together
    if "kt" not in traj:
        kt = None
        nframes = 1
    if "kecho" not in traj:
        kecho = None
        nechoes = 1
    
    # extract values for full trajectory structure
    t = traj["t"]
    te = traj["te"]
    mtx = traj["mtx"]
    adc = traj["adc"]
    acs = traj["acs"]
    acs["kr"] = acs_kr
    compressed = traj
    compressed["rot"] = R
    compressed["rot_axis"] = axis

    # actual packing
    traj = {"kr": scale_traj(kr), "kecho": kecho, "kt": kt, "t": t, "te": te, "mtx": [mtx, mtx], "dcf": dcf, "adc": adc, "acs": acs, "compressed": compressed}
    
    # remove echoes and frames if they are trivial 
    if nechoes == 1:
        traj.pop("kecho", None)
        traj["compressed"].pop("kecho", None)
    if nframes == 1:
        traj.pop("kt", None)
        traj["compressed"].pop("kt", None)
    
    return traj
    

def make_stack(ordering, traj, kz, plan, kwargs):
    
    # parse flags
    tilt = kwargs["tilt"]
    
    # get final number of slices
    nangles = traj["kr"].shape[0]
    nslices = kz.shape[0]
                
    # expand
    if tilt: # phi, kr, kecho, kt already account for nslices, I only have to expand plan and kz
        nreps = int(nangles // nslices)
        if ordering == "sequential": # all the z, then move to next interleave
            plan = np.tile(plan, nreps)
        elif ordering == "interleaved": # all the interleaves, then move to next z
            plan = np.repeat(plan, nreps, axis=0)
    else: # phi, kr, kecho, kt must be expanded nslices times, while plan and kz must be repeated nangles time
        if ordering == "sequential": # all the z, then move to next interleave
            plan = np.tile(plan, nangles)
            traj["kr"] = np.repeat(traj["kr"], nslices, axis=-1)
            if "kt" in traj:
                traj["kt"] = np.repeat(traj["kt"], nslices, axis=-1)
            if "kecho" in traj:
                traj["kecho"] = np.repeat(traj["kecho"], nslices, axis=-1)
        elif ordering == "interleaved": # all the interleaves, then move to next z
            plan = np.repeat(plan, nangles, axis=0)
            traj["kr"] = np.apply_along_axis(np.tile, -1, traj["kr"], nslices)
            if "kt" in traj:
                traj["kt"] = np.tile(traj["kt"], nslices)
            if "kecho" in traj:
                traj["kecho"] = np.tile(traj["kecho"], nslices)
    
    # append
    traj["kz"] = kz
    
    return traj, plan
    
def make_multiecho(traj, plan, nechoes, key="k"):
    # check
    if traj["kt"] is None:
        traj.pop("kt", None)
    
    # process
    if nechoes > 1:
        # parse  
        k = traj[key]
        nshots = k.shape[0]
        
        # expand plan
        if plan is not None:
            plan = np.repeat(plan, nechoes, axis=-1)
    
        # expand trajectory
        k = np.repeat(k, nechoes, axis=0)
        if "kt" in traj:
            traj["kt"] = np.repeat(traj["kt"], nechoes)
        if "kz" in traj:
            traj["kz"] = np.repeat(traj["kz"], nechoes)
        
        # add echo index
        kecho = np.arange(nechoes).astype(int)
        kecho = np.tile(kecho, nshots)
        
        # prepare out
        traj[key] = k
        traj["kecho"] = kecho
    
    return traj, plan
    

