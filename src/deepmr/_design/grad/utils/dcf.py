"""Estimation of the density compensation array methods."""

__all__ = ["analytical_dcf", "voronoi", "angular_compensation"]

import warnings

import torch

import numpy as np
import numba as nb

from scipy.spatial import Voronoi

from . import misc
from . import tilt

warnings.filterwarnings(category=nb.NumbaTypeSafetyWarning, action="ignore")

def analytical_dcf(coord):
    """
    Estimate  density compensation weight using the square of k-space radius.

    Args:
        coord (array-like): array of shape (..., ndim) containing the coordinates of the points.
            
    Returns:
        (array-like): dcf array of shape (...,).
            
    """
    return (coord**2).sum(axis=-1)**0.5 # shape (..., )

    
def angular_compensation(wi, coord, rotation_axis):
    """
    Correct dcf for 3D projection case.

    Args:
        wi (array-like): input in-plane density compensation factors.
        coord (array-like): array of shape (nproj, npts, 3) containing the coordinates of the points.
        rotation_axis (array-like): array of shape (nproj,) describing the through-plane rotation axis
            for each shot (0=x, 1=y, 2=z).
            
    Returns:
        (array-like): corrected dcf array of shape (nproj, npts, 3).
            
    """
    # versors
    ux = np.asarray([1, 0, 0])
    uy = np.asarray([0, 1, 0])
    uz = np.asarray([0, 0, 1])
    umat = np.stack((ux, uy, uz), axis=0)
    
    # versor for each proj
    u = umat[rotation_axis.astype(int)].astype(np.float32)
    
    # angular component
    cangular = np.cross(coord.transpose(1, 0, 2), u).transpose(1, 0, 2)
    wi_angular = analytical_dcf(cangular) # shape (npts)
    
    # apply compensation
    wi = wi[None, :] * wi_angular
    
    # normalize
    wi /= np.sum(wi)
    
    return wi
    
def voronoi(coord, nshots=1, fix_edge=False, ratio=0.1):
    """
    Estimate  density compensation weight using voronoi parcellation.

    In case of multiple point in the a given kspace location, the weight is split evenly.

    Args:
        coord (array-like): array of shape (M, 2) or (M, 3) containing the coordinates of the points.
        fix_edge (bool): interpolate point at edges where voronoi cell is open (default to zero).
        ratio (float): fudge factor to remove spike points in DCF.
            
    Returns:
        (array-like): dcf array of shape (M, 2) or (M, 3).
            
    """
    # convert to numpy array
    if isinstance(coord, torch.Tensor):
        istorch = True
        device = coord.device
        coord = coord.clone()
        coord = coord.detach().numpy
    else:
        istorch = False
        coord = coord.copy()
        
    # stack
    if np.iscomplexobj(coord):
        coord = misc.traj_complex_to_array(coord)
        
    # build full disk
    if nshots > 1:
        angles = tilt.make_tilt("uniform", nshots)
        rotmat = tilt.angleaxis2rotmat(angles, [0, 0, 1])
        coord = tilt.projection(coord.T, rotmat).astype(np.float32)
        coord = coord.transpose(1, 2, 0)
        
    # get index of k-space center
    if nshots > 1:
        cabs = (coord[0]**2).sum(axis=-1)**0.5
    else:
        cabs = (coord**2).sum(axis=-1)**0.5
    k0_idx = np.argmin(cabs)

    # flatten
    ishape = coord.shape[:-1]
    coord = coord.reshape(-1, coord.shape[-1])
    
    # find duplicated
    isduplicated, first, duplicated = _repeated(coord)
    
    if isduplicated:  # multiple DC points?
        # preallocate
        wi = np.zeros(coord.shape[0], dtype=np.float32)
        
        # calculate unique + first of duplicated
        wi[first] = _voronoi(coord[first], fix_edge, ratio)
        
        _fill_duplicated(wi, duplicated)
        
    else:
        wi = _voronoi(coord, fix_edge, ratio)
                
    # correct for numerical errors
    wi = np.nan_to_num(wi, posinf=0.0, neginf=0.0)
        
    # restore
    if istorch:
        wi = torch.as_tensor(wi, dtype=torch.float32, device=device)
    
    # reshape
    wi = wi.reshape(ishape)
    
    # get first interleaf
    if nshots > 1:
        wi = wi[0]
        
    # normalize
    wi = _normalize_weight(wi)
    
    # impose that center of k-space weight is 1 / nshots
    scale = 1 / wi[k0_idx] / nshots 
    wi = scale * wi
    
    return wi.squeeze()

#%% local utils
def _normalize_weight(weights):
    """From MRI-NUFFT."""
    sum_weights = np.sum(weights)
    weights[np.isclose(weights, 0.0)] = 1.0
    inv_weights = sum_weights / weights
    inv_weights = inv_weights / np.sum(inv_weights)
    inv_weights[np.isclose(inv_weights, 0.0)] = 1.0
    return 1 / inv_weights

def _repeated(input):
    """Find list of repeated points (e.g., k=(0,0,0))"""
    _, ind1, ind2, count = np.unique(input, return_index=True, return_inverse=True, return_counts=True, axis=0)
    
    # check for duplicated 
    isduplicated = np.any(count > 1)
    
    if isduplicated:
        duplicated = np.where(count[ind2] > 1)[0]
        ind2 = _get_duplicated(ind2, duplicated)
    
    return isduplicated, ind1, ind2

@nb.njit(fastmath=True, cache=True)
def _get_duplicated(input, duplicated):
    
    # get shape
    M = len(duplicated)
    
    # prepare output
    ind = []
    
    # actual loop
    for n in range(M):
        ind.append(np.where(input == duplicated[n]))
        
    # cast
    ind = nb.typed.List(ind)
        
    return ind
               
def _voronoi(coord, fix_edge, ratio):
    """
    Estimate  density compensation weight using voronoi parcellation.

    This assume unicity of the point in the kspace.
    """
    # get sizes
    if coord.shape[-1] == 2:
        vol = _compute_area
    else:
        vol = _compute_volume
     
    # preallocate ouput
    wi = np.zeros(coord.shape[0], dtype=np.float32)
    
    # calculate voronoi parcellation
    v, c = _voronoin(coord)
        
    # actual calculation
    vol(wi, v, c)
    
    # clean up (from toppe)
    fufa = 0.98
    go_ahead = True
    while go_ahead:
      r = 1 - (1 - ratio) * fufa
      thld = _select_nth(wi, round(r * wi.size)) # 0.1 chosen arbitrarily
      if not(np.isfinite(thld)):
          fufa **= 2
      else:
          go_ahead = False
    
    # trim
    wi[wi > thld] = thld
    
    # optionally, fix edges
    if fix_edge:
        # For edge point (infinite voronoi cells) we extrapolate from neighbours
        # Initial implementation in Jeff Fessler's MIRT
        rho = np.sum(coord**2, axis=-1)          
        igood = rho > 0.6 * np.max(rho)
        if len(igood) < 10:
            print(f"dubious extrapolation with {len(igood)} points")
        poly = np.polynomial.Polynomial.fit(rho[igood], wi[igood], 3)
        wi[wi == 0] = poly(rho[wi == 0])
        
    return wi

@nb.njit(parallel=True, fastmath=True, cache=True)
def _compute_area(wi, v, c):
    # get shape
    M = wi.shape[0]
    
    # actual calculation
    for mm in nb.prange(M):
        if np.all(c[mm] != -1):
            wi[mm] = _vol2d(v[c[mm]])
            
@nb.njit(fastmath=True, cache=True)
def _vol2d(points):
    """
    Compute the area of a convex 2D polygon.

    Args:
        points (array_like): array of shape (N, 2) containing the coordinates of the points.

    Returns
        (float): area of the polygon.
    """
    # https://stackoverflow.com/questions/451426/how-do-i-calculate-the-area-of-a-2d-polygon
    area = 0
    for i in range(1, len(points) - 1):
        area += points[i, 0] * (points[i + 1, 1] - points[i - 1, 1])
    area += points[-1, 0] * (points[0, 1] - points[-2, 1])
    
    # we actually don't provide the last point, so we have to do another edge case.
    area += points[0, 0] * (points[1, 1] - points[-1, 1])
    return abs(area) / 2.0

@nb.njit(parallel=True, fastmath=True, cache=True)
def _compute_volume(wi, v, c):
    # get shape
    M = wi.shape[0]
    
    # actual calculation
    for mm in nb.prange(M):
        if np.all(c[mm] != -1):
            wi[mm] = _vol3d(v[c[mm]])
            
@nb.njit(fastmath=True, cache=True)
def _vol3d(points):
    """
    Compute the volume of a convex 3D polygon.

    Args:
        points (array_like): array of shape (N, 3) containing the coordinates of the points.

    Returns
        (float): volume of the polygon.
    """
    base_point = points[0]
    A = points[:-2] - base_point
    B = points[1:-1] - base_point
    C = points[2:] - base_point
    return np.sum(np.abs(np.dot(np.cross(B, C), A.T))) / 6.0

def _voronoin(input):
    """
    Get the same as MATLAB voronoin.
    
    source:
        https://stackoverflow.com/questions/59090443/how-to-get-the-same-output-of-voronoin-of-matlab-by-scipy-spatial-voronoi-of-pyt
    """
    vor = Voronoi(input)
    
    # get vertices
    v = vor.vertices
    
    # get indices
    c = [np.asarray(vor.regions[vor.point_region[n]], dtype=np.int64) for n in range(input.shape[0])]
    
    # avoid error
    c = nb.typed.List(c)
    
    return v, c

@nb.njit(fastmath=True, cache=True)
def _fill_duplicated(input, idx):
    """distribute weights across repeatedly sampled k-space locations."""
    # get shape
    M = len(idx)
    
    for n in range(M):
        ii = idx[n]
        ni = len(ii)
        val = input[ii[0]] / ni
        for m in range(ni):
            input[ii[m]] = val
            
def _select_nth(input, n, direction="descend"):
    """
    Select the Nth ordered element from x, largest or smallest by 'direction'.
    """
    ind = np.argsort(input)
    if direction == "descend":
        ind = ind[::-1]
        
    return input[ind[n]]


            