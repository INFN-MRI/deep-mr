"""I/O routines for MATLAB acquisition header."""

__all__ = ["read_matlab_acqhead"]

import numpy as np

from ..generic import matlab
from ..types.header import Header

def read_matlab_acqhead(filename, dcfname=None, methodname=None, sliceprofname=None):
    """
    Read acquistion header from matlab file.

    Parameters
    ----------
    filename : str
        Path to the file.
    dcfname : str, optional
        Path to the dcf file.
        The default is None.
    methodname : str, optional
        Path to the schedule description file.
        The default is None.
    sliceprofname : str, optional
        Path to the slice profile file.
        The default is None.

    Returns
    -------
    head : deepmr.Header
        Deserialized acqusition header.
        
    """
    # load dcf
    matfile, filename = matlab.read_matfile(filename, True)
    
    # get k space trajectory
    k, reshape = _get_trajectory(matfile)
    ndim = k.shape[-1]
    
    # get adc
    adc = _get_adc(matfile)
     
    # get dcf
    dcf = _get_dcf(matfile, k, filename, dcfname)
            
    # get sampling time
    t = _get_sampling_time(matfile)
    
    # get matrix
    shape = _get_shape(matfile, ndim)
    
    # get resolution
    resolution, spacing = _get_resolution_and_spacing(matfile, shape, ndim)
            
    # initialize header
    head = Header(shape, t, _resolution=resolution, _spacing=spacing)
    head.traj = k
    head.dcf = dcf
    head._adc = adc
   
    # get schedule file
    head = _get_schedule(head, matfile, methodname)
    
    # reformat trajectory
    acq_type = _estimate_acq_type(k, shape)
    
    # reformat
    head, _reformat_trajectory(head, acq_type, reshape)
    
    # get slice profile
    head = _get_slice_profile(head, matfile, filename, sliceprofname)
    
    # get basis
    head = _get_basis(head, matfile)
    
    return head
 
# %% subroutines
def _get_trajectory(matfile):
    
    # flags
    reshape = True
    
    if "traj" in matfile:
        k = matfile["traj"]
        reshape = False
    elif "k" in matfile:
        k = matfile["k"]
        # get shape
        if "t" in matfile:
            npts = matfile["t"].shape[-1]
        elif "ts" in matfile:
            npts = matfile["ts"].shape[-1]
        else:
            raise RuntimeError("Time not found!")
        nviews = int(k.shape[1] / npts)
        
        # reshape
        k = k.reshape(nviews, npts, -1)
    elif "ks" in matfile and "phi" in matfile:
        ks = matfile["ks"]
        phi = matfile["phi"].T
        
        # get shape
        npts, nviews = ks.shape[1], phi.shape[1]
        
        # rotate in plane
        sinphi = np.sin(np.deg2rad(phi))
        cosphi = np.cos(np.deg2rad(phi))
        
        if "theta" in matfile:
            theta = matfile["theta"].T
            sintheta = np.sin(np.deg2rad(theta))
            costheta = np.cos(np.deg2rad(theta))
            k = np.stack(
                [ks[..., 0] * costheta * cosphi + ks[..., 1] * sinphi - ks[..., 2] * sintheta * cosphi,
                 -ks[..., 0] * costheta * sinphi + ks[..., 1] * cosphi - ks[..., 2] * sintheta * sinphi,
                 ks[..., 0] * sintheta + ks[..., 2] * costheta],
                axis=-1)
        else:
            k = np.stack(
                [ks[..., 0] * cosphi + ks[..., 1] * sinphi,
                 -ks[..., 0] * sinphi + ks[..., 1] * cosphi]
                )
            if "kz" in matfile and matfile["kz"]:
                kz = np.repeat(matfile["theta"].T, npts, -1)
                nz = kz.shape[0]
                k = np.concatenate(
                    [np.apply_along_axis(np.tile, 0, k, nz),
                     np.apply_along_axis(np.tile, 0, kz, nviews)], 
                     axis=-1)
    else:
        raise RuntimeError("K-space trajectory not found!")
        
    return k, reshape

def _get_adc(matfile):
    if "adc" in matfile:
        adc = matfile["ind"]
    else:
        if "ind" in matfile:
            adc = matfile["ind"][0].astype(bool)
        elif "inds" in matfile:
            adc = matfile["inds"].squeeze().astype(bool)
        else:
            raise RuntimeError("ADC indexes not found!")
        adc = np.argwhere(adc)[[0, -1]].squeeze()
    return adc

def _get_dcf(matfile, k, filename, dcfname):
    
    if "dcf" in matfile or "dcfs" in matfile:
        struct = matfile
    else:
        try:
            if dcfname is None:
                dcfname = ".".join([filename.split(".")[0] + "_dcf", "mat"])
            struct = matlab.read_matfile(dcfname)
        except Exception:
            struct = {}
        
    if "dcf" in struct:
        dcf = struct["dcf"].reshape(k.shape[:-1])
    elif "dcfs" in struct:
        dcf = struct["dcfs"].squeeze() # (npts,)
    else:
        dcf = None
   
    return dcf

def _get_sampling_time(matfile):
    if "t" in matfile:
        t = np.atleast_2d(matfile["t"][0])
    elif "ts" in matfile:
        t = matfile["ts"].squeeze()
    else:
        raise RuntimeError("Sampling time not found!")
    
    # s to ms
    if np.round(t).max() == 0:
        t *= 1e3
        
    # remove offset from sampling time
    t -= t[0]
    
    return t

def _get_shape(matfile, ndim):
    if "mtx" in matfile:
        shape = matfile["mtx"].squeeze()[::-1]
    elif "npix" in matfile:
        shape = matfile["npix"].squeeze()[::-1]
    elif "shape" in matfile:
        shape = matfile["shape"]
    else:
        raise RuntimeError("Matrix size not found!")
        
    # expand scalar
    if len(shape) == 1:
        shape = [int(shape)] * ndim
        
    return shape

def _get_resolution_and_spacing(matfile, shape, ndim):
    if "resolution" in matfile:
        resolution = matfile["resolution"]
    else:
        if "fov" in matfile:
            fov = [float(matfile["fov"].squeeze()) * 1e3] * ndim # mm
        else:
            raise RuntimeError("Field of View not found!")
        resolution = (np.asarray(fov) / np.asarray(shape)).tolist() 
    
    # get spacing
    if "spacing" in matfile:
        spacing = matfile["spacing"]
    else:
        spacing = resolution[0]
        
    return resolution, spacing

def _get_schedule(head, matfile, schedulename):
    
    if "method" in matfile:
        schedule = matfile["method"]
    else:
        try:
            if schedulename is None:
                schedulename = ".".join(schedulename.split(".")[-1].extend(["_method", "mat"]))
            schedule = matlab.read_matfile(schedulename)
            schedule = schedule["method"]
        except Exception:
            schedule = None
            
    if schedule is not None:
        if "VariableFlip" in schedule.dtype.fields:
            FA = schedule["VariableFlip"][0][0].squeeze().astype(np.float32)
        else:
            FA = 0.0
        if "VariablePhase" in schedule.dtype.fields:
            phase = schedule["VariableFlip"][0][0].squeeze().astype(np.float32)
            FA = FA * np.exp(1j * np.deg2rad(phase))
        else:
            FA = 0.0
        if "VariableTE" in schedule.dtype.fields:
            TE = schedule["VariableTE"][0][0].squeeze().astype(np.float32)
        elif "TE" in schedule.dtype.fields:
            TE = schedule["TE"][0][0].squeeze().astype(np.float32)
        if "VariableTR" in schedule.dtype.fields:
            TR = schedule["VariableTR"][0][0].squeeze().astype(np.float32)
        else:
            TR = 0.0
        if "InversionTime" in schedule.dtype.fields:
            TI = schedule["InversionTime"][0][0].squeeze().astype(np.float32)
        else:
            TI = 0.0
    else:
        if "FA" in matfile:
            FA = matfile["FA"]
        else:
            FA = 0.0
        if "TE" in matfile:
            TE = matfile["TE"]
        else:
            TE = 0.0
        if "TR" in matfile:
            TR = matfile["TR"]
        else:
            TR = 0.0
        if "TI" in matfile:
            TI = matfile["TI"]
        else:
            TI = 0.0
        
        # rf phase
        if "user" in matfile and "rf_phase" in matfile["user"]:
            phase = np.frombuffer(matfile["user"]["rf_phase"], dtype=np.float32)
            FA = FA * np.exp(1j * np.deg2rad(phase))
            
    # broadcast contrasts
    FA, TI, TE, TR = np.broadcast_arrays(FA, TI, TE, TR)
    
    # update header
    head.TI = TI
    head.TE = TE
    head.TR = TR
    head.FA = FA
    
    return head

def _get_slice_profile(head, matfile, filename, sliceprofname):
    if "sliceB1" in matfile or "slice_profile" in matfile:
        struct = matfile
    else:
        try:
            if sliceprofname is None:
                sliceprofname = ".".join([filename.split(".")[0] + "_sliceprof", "mat"])
            struct = matlab.read_matfile(sliceprofname)
        except Exception:
            struct = {}
        
    if "sliceB1" in struct:
        slice_prof = struct["sliceB1"].squeeze()
    elif "slice_profile" in struct:
        slice_prof = struct["slice_profile"].squeeze() # (npts,)
    else:
        slice_prof = None
    
    if slice_prof is not None:
        head.user["slice_profile"] = slice_prof
        
    return head

def _get_basis(head, matfile):
    
    if "Vk" in matfile:
        basis = matfile["Vk"]
    elif "basis" in matfile:
        basis = matfile["basis"]
    else:
        basis = None
        
    if basis is not None:
        basis = basis.astype(np.complex64)
    
        # compress
        if np.isreal(basis).all():
            basis = basis.real
        elif np.isreal(basis.imag + 1j * basis.real).all():
            basis = basis.imag
        
        head.user["basis"] = basis
    
    return head
    

def _estimate_acq_type(k, shape):
    
    ndim = k.shape[-1]
    ktmp = k * shape
    iscart = [np.allclose(ktmp[..., n].astype(float), ktmp[..., n].astype(int)) for n in range(ndim)]
    if np.all(iscart):
        acq_type = "cart"
    elif np.any(iscart):
        acq_type = "hybrid"
    else:
        acq_type = "noncart"

    return acq_type

def _reformat_trajectory(head, acq_type, reshape):
    
    separable = False
    k = head.traj
    dcf = head.dcf
    shape = head.shape[::-1]
    
    # get number of dimensions
    ndim = k.shape[-1]
    
    if reshape:
        if acq_type == "hybrid" or acq_type == "cart":
            nz = shape[-1]
            kz = k[:, 0, -1].astype(float) * nz + nz // 2 # (nviews,)
            
            # check if it is fully sampled
            if np.allclose(np.unique(kz), np.arange(nz)):
                kint = k.reshape(nz, -1, k.shape[-2], k.shape[-1])[..., :2]    
                kseq = k.reshape(-1, nz, k.shape[-2], k.shape[-1])[..., :2]
                kseq = kseq.swapaxes(0, 1)
                        
                # test for interleaved
                if np.all([np.allclose(kint[0], kint[n]) for n in range(nz)]):
                    separable = "interleaved"
                    k = kint[0]
                elif np.all([np.allclose(kseq[0], kseq[n]) for n in range(nz)]):
                    separable = "sequential"
                    k = kseq[0]
            
        # now reshape
        ncontrasts = len(head.FA)
        nviews = int(k.shape[0] / ncontrasts)  
        
        k = k.reshape(nviews, ncontrasts, -1, ndim).swapaxes(0, 1)
        k = np.ascontiguousarray(k).astype(np.float32)
        if dcf is not None:
            dcf = dcf.reshape(nviews, ncontrasts, -1).swapaxes(0, 1)
            dcf = np.ascontiguousarray(dcf).astype(np.float32)
    else:
        if acq_type == "hybrid":
            acq_type = "noncart"
            
    # assign
    head.traj = k
    head.dcf = dcf
    head.user["separable"] = separable
    head.user["mode"] = str(k.shape[-1]) + "D" + acq_type
        
    return head
        
        
