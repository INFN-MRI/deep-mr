"""Signal filtering routines."""

__all__ = ["fermi"]

import torch

def fermi(ndim, size, width=None):
    """
    Build a n-dimensional Fermi filter.
    
    This routine can handle isotropic ND matrices. User
    can specify the size of window support and the FWHM.
    The filter can be used in the context of image processing
    to mitigate ringing artifact [1].

    Parameters
    ----------
    ndim : int
        Number of dimensions (e.g., 1=1D, 2=2D, 3=3D).
    size : int
        Support of the window. Filter size will be ndim * [size].
    width : int, optional
        Full width half maximum of the filter.
        If "None", it is automatically set to "size". The default is "None".

    Returns
    -------
    filt : torch.Tensor
        Fermi window of shape ndim * [size] and FWHM = width.
        
    Example
    -------
    >>> import deepmr
    
    We can design e.g., 1D, 2D or 3D filters as:
        
    >>> filt1d = deepmr.fermi(1, 128)
    >>> filt1d.shape
    torch.Size([128])
    >>> filt2d = deepmr.fermi(2, 128)
    torch.Size([128, 128])
    >>> filt3d = deepmr.fermi(3, 128)
    torch.Size([128, 128])
    
    Bu default, FWHM is equal to the support size:
        
    >>> (filt1d >= 0.5).sum()
    tensor(128)
    
    User can specify a smaller FWHM via "width" parameter:
        
    >>> filt1d = deepmr.fermi(1, 128, width=32)
    >>> filt1d.shape
    torch.Size([128])
    >>> (filt1d >= 0.5).sum()
    tensor(47)
    
    The discrepancy between nominal and actual FWHM is due to signal
    discretization.
    
    References
    ----------
    [1] Bernstein, M.A., Fain, S.B. and Riederer, S.J. (2001), 
    Effect of windowing and zero-filled reconstruction of MRI data 
    on spatial resolution and acquisition strategy. 
    J. Magn. Reson. Imaging, 14: 270-280. 
    https://doi.org/10.1002/jmri.1183

    """
    # default width
    if width is None:
        width = size
    
    # get radius
    radius = int(width // 2)
        
    # build grid, normalized so that u = 1 corresponds to window FWHM
    R = [torch.arange(int(-size // 2), int(size // 2), dtype=torch.float32) for n in range(ndim)]
    
    # get transition width
    T = 20 * size # width / 128
    
    # build center-out grid
    R = torch.meshgrid(*R, indexing="xy")
    R = torch.stack(R, dim=0)
    R = (R**2).sum(dim=0)**0.5
    
    # build filter
    filt = 1 / (1 + torch.exp((R - radius)) / T)
    filt /= filt.max()
    
    return filt

