"""High level rawdata reading routines."""

__all__ = ["read_rawdata"]

import torch
from ..utils.mrd.read import read as read_mrd

try:
    import gehc
    __GEHC_AVAILABLE__ = True
except:
    __GEHC_AVAILABLE__ = False
    

def read_rawdata(fname: str, vendor: str = "GEHC") -> (torch.Tensor, dict):
    """
    Read kspace rawdata.

    Parameters
    ----------
    fname : str
        Path to raw data file.
    vendor : str, optional
        Target vendor. The default is "GEHC".

    Returns
    -------
    torch.Tensor
        K-Space rawdata.
    dict
        Structure containing data description.

    """
    # TODO: add auto feature
    # TODO: add conversion to common header type
    if vendor == "GEHC":
        if __GEHC_AVAILABLE__:
            data, header = gehc.read_rawdata(fname)
            data = torch.from_numpy(data)
            return data, header
        else:
            print("GEHC reader is private - ask for access")
            return None, None
            
    if vendor == "Siemens":
        print("Not Implemented")
    
    if vendor == "Philips":
        print("Not Implemented")
        
    if vendor == "MRD":
        data, header = read_mrd(fname)
        data = torch.from_numpy(data)
        return data, header
        

