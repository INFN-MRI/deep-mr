"""I/O Routines for MRD files."""

__all__ = ["read_mrd"]

import numpy as np

import ismrmrd as mrd

from ..utils.pathlib import get_filepath
from ..utils.header import Header

def read_mrd(filename: str) -> (dict, np.ndarray):
    
    # get full path
    filename = get_filepath(filename, True, "h5")
    
    # open file
    dset = mrd.Dataset(filename)
    
    # read header
    mrdhead = _read_header(dset)
    
    # read acquisitions
    nacq = dset.number_of_acquisitions()
    acq = [dset.read_acquisition(n) for n in range(nacq)]
    
    # build header
    header = Header.from_mrd(mrdhead, acq) 
    
    # close
    dset.close()
    
    return acq, header, mrdhead
        
    
def _read_header(dset):
    xml_header = dset.read_xml_header()
    xml_header = xml_header.decode("utf-8")
    return mrd.xsd.CreateFromDocument(xml_header)



def _deserialize(mrdprot):
    
    # get all waveforms, dcfs and indexes
    traj0 = []
    dcf0 = []
    t0 = []
    icontrast = []
    iz = []
    iview = []
    
    # get number of profiles
    nprofiles = len(mrdprot["profiles"])
    for n in range(nprofiles):
        ctraj = mrdprot["profiles"][n].traj
        cdcf = mrdprot["profiles"][n].data
        
        npts = cdcf.shape[-1]
        dt = mrdprot["profiles"][n].head.sample_time_us
        ct = np.arange(npts) * dt * 1e-3 # ms
        
        cicontrast = mrdprot["profiles"][n].head.idx.contrast
        ciz = mrdprot["profiles"][n].head.idx.slice
        ciview = mrdprot["profiles"][n].head.idx.kspace_encode_step_1
        
        # append
        traj0.append(ctraj)
        dcf0.append(cdcf)
        t0.append(ct) 
        icontrast.append(cicontrast)
        iz.append(ciz)
        iview.append(ciview)
    
    # stack / concatenate
    traj0 = np.stack(traj0, axis=0)
    dcf0 = np.stack(dcf0, axis=0)
    t0 = np.stack(t0, axis=0)
    icontrast = np.asarray(icontrast)
    iz = np.asarray(iz)
    iview = np.asarray(iview)
        
    # get geometry from header
    if isinstance(mrdprot["hdr"].encoding, list):
        fov = mrdprot["hdr"].encoding[0].encodedSpace.fieldOfView_mm
        ishape = mrdprot["hdr"].encoding[0].encodedSpace.matrixSize
        kshape = mrdprot["hdr"].encoding[0].encodingLimits
    else:
        fov = mrdprot["hdr"].encoding.encodedSpace.fieldOfView_mm
        ishape = mrdprot["hdr"].encoding.encodedSpace.matrixSize
        kshape = mrdprot["hdr"].encoding.encodingLimits
        
    # get fov, matrix size and kspace size
    fov = [fov.x, fov.y, fov.z]
    ishape = [ishape.x, ishape.y, ishape.z]
    kshape = [kshape.contrast.maximum+1, kshape.slice.maximum+1, kshape.kspace_encoding_step_1.maximum+1, npts]
    
    # sort trajectory, dcf and t
    traj = np.zeros(kshape + [traj0.shape[-1]], dtype=np.float32)
    dcf = np.zeros(kshape, dtype=np.float32)
    t = np.zeros(kshape, dtype=np.float32)
    
    # cast dcf to real
    dcf0 = dcf0.real
    
    for n in range(nprofiles):
        traj[icontrast[n], iz[n], iview[n]] = traj0[n]
        dcf[icontrast[n], iz[n], iview[n]] = dcf0[n]
        t[icontrast[n], iz[n], iview[n]] = t0[n]
            
    # get flip angle, TE, TR, TI
    flip = mrdprot["hdr"].sequenceParameters.flipAngle_deg
    TE = mrdprot["hdr"].sequenceParameters.TE
    TI = mrdprot["hdr"].sequenceParameters.TI
    TR = mrdprot["hdr"].sequenceParameters.TR
    
    # get custom parameters
    params = {}
    for n in range(len(mrdprot["hdr"].userParameters.userParameterLong)):
        try:
            key = mrdprot["hdr"].userParameters.userParameterLong[n].name.name
        except:
            key = mrdprot["hdr"].userParameters.userParameterLong[n].name                
        if key in params:
            if isinstance(params[key], list) is False:
                params[key] = [params[key]]
            try:
                params[key].append(mrdprot["hdr"].userParameters.userParameterLong[n].name.value)
            except:
                params[key].append(mrdprot["hdr"].userParameters.userParameterLong[n].value)
        else:
            try:
                params[key] = mrdprot["hdr"].userParameters.userParameterLong[n].name.value
            except:
                params[key] = mrdprot["hdr"].userParameters.userParameterLong[n].value
    for n in range(len(mrdprot["hdr"].userParameters.userParameterDouble)):
        try:
            key = mrdprot["hdr"].userParameters.userParameterDouble[n].name.name
        except:
            key = mrdprot["hdr"].userParameters.userParameterDouble[n].name
        if key in params:
            if isinstance(params[key], list) is False:
                params[key] = [params[key]]
            try:
                params[key].append(mrdprot["hdr"].userParameters.userParameterDouble[n].name.value)
            except:
                params[key].append(mrdprot["hdr"].userParameters.userParameterDouble[n].value)
        else:
            try:
                params[key] = mrdprot["hdr"].userParameters.userParameterDouble[n].name.value
            except:
                params[key] = mrdprot["hdr"].userParameters.userParameterDouble[n].value

    return {"kr": traj, "dcf": dcf, "t": t, "fov": fov, "shape": ishape, "flip": flip, "TE": TE, "TR": TR, "TI": TI, "params": params}
    


