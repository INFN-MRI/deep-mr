"""
A small package to perform pattern matching fitting.
"""

__all__ = ["tsmi2map", "map2tsmi", "BlochDictionary"]

from dataclasses import dataclass

import numpy as np
import numba as nb

def tsmi2map(bloch_dict, time_series):
    """
    Takes as input the time series in image space (or the set of temporal subspace coefficients)
    and returns quantitative maps / proton density.
    
    Input:
        bloch_dict (BlochDictionary): dictionary of simulated tsmi  with the following fields:
            - atoms (ndarray): simulated and normalized tsmi with shape (ncoeff, natoms)
            - norm (ndarray): norm of the simulated tsmi of shape (natoms,)
            - lookup_table (ndarray): quantitative parameters generating the tsmi of shape (nparams, natoms)
            - labels (list): names of the parameters of interest (e.g., T1, T2)
        time_series (ndarray): Input tsmi of shape (ncoeff, nz, ny, nx).
        
    Output:
        m0 (ndarray): tissue proton density of shape (nz, ny, nx).
        qmaps (dict): dictionary with the estimated tissue parametric maps, each of shape (nz, ny, nx).
    """
    # get shape
    shape = time_series.shape[1:]
    time_series = time_series.reshape((time_series.shape[0], np.prod(shape)))
    time_series = np.ascontiguousarray(time_series.transpose().conj())
    
    # get atoms
    atoms = np.ascontiguousarray(bloch_dict.atoms.transpose())
    labels = bloch_dict.lookup_table
    
    # get quantitative maps and proton density
    qmaps, cost, idx = _matching(time_series, atoms, labels)
    qmaps = qmaps.reshape([qmaps.shape[0]] + list(shape))
    qmaps = [qmap for qmap in qmaps]
    m0 = (cost / bloch_dict.norm[idx]).reshape(shape)
    
    return m0, dict(zip(bloch_dict.labels, qmaps))

def map2tsmi(bloch_dict, qmaps, m0=None):
    """
    Takes as input quantitative maps / proton density
    and returns the time series in image space (or the set of temporal subspace coefficients).
    
    Input:
        bloch_dict (BlochDictionary): dictionary of simulated tsmi  with the following fields:
            - atoms (ndarray): simulated and normalized tsmi with shape (ncoeff, natoms)
            - norm (ndarray): norm of the simulated tsmi of shape (natoms,)
            - lookup_table (ndarray): quantitative parameters generating the tsmi of shape (nparams, natoms)
            - labels (list): names of the parameters of interest (e.g., T1, T2)
        qmaps (dict): dictionary with the estimated tissue parametric maps, each of shape (nz, ny, nx).
        m0 (ndarray): tissue proton density of shape (nz, ny, nx).
        
    Output:
        time_series (ndarray): Output tsmi of shape (ncoeff, nz, ny, nx).

    """
    # convert quantitative maps to ndarray
    qmaps = np.stack(list(qmaps.values()), axis=0)
    
    # get shape
    shape = qmaps.shape[1:]
    qmaps = qmaps.reshape((qmaps.shape[0], np.prod(shape)))
    qmaps = np.ascontiguousarray(qmaps.transpose()) # (nvoxels, ncoeffs)
    
    # get lookup_table
    lookup_table = np.ascontiguousarray(bloch_dict.lookup_table.transpose()) # (natoms, nparams)
    
    # preallocate
    idx = np.zeros(qmaps.shape[0], dtype=int)
    
    # do actual matching
    _norm_search(qmaps, lookup_table, idx)
    
    # get quantitative maps and proton density
    time_series = bloch_dict.atoms[:, idx]
    time_series = time_series.reshape([time_series.shape[0]] + list(shape))
    
    if m0 is not None:
        time_series = m0 * time_series
    
    return time_series

#%% local utils
@dataclass
class BlochDictionary:
    atoms: np.ndarray
    lookup_table: np.ndarray
    labels: list
    
    def __post_init__(self):
        self.atoms = np.ascontiguousarray(self.atoms.transpose())
        self.norm = np.linalg.norm(self.atoms, axis=0)
        self.atoms = self.atoms / self.norm
        self.lookup_table = np.ascontiguousarray(self.lookup_table.transpose())
        self.labels = list(self.labels)
        
@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _get_norm(x, y):
    z = 0.0 
    for n in range(x.shape[0]):
        z += np.abs(x[n] - y[n])
        
    return z


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _norm_search(quantitative_maps, lookup_table, idx):     
    for n in nb.prange(quantitative_maps.shape[0]): 
        c0 = _get_norm(quantitative_maps[n], lookup_table[0])

        for p in range(1, lookup_table.shape[0]):
            c = _get_norm(quantitative_maps[n], lookup_table[p])

            # keep maximum value
            if c < c0:
                c0 = c
                idx[n] = p


def _matching(signals, atoms, labels):
    """
    performs pattern matching step.
    """      
    # preallocate
    cost = np.zeros(signals.shape[0], dtype=np.complex64)
    idx = np.zeros(signals.shape[0], dtype=int)
    
    # do actual matching
    _dot_search(signals, atoms, cost, idx)

    return labels[:, idx], cost, idx


@nb.njit(fastmath=True, parallel=True)  # pragma: no cover
def _dot_search(time_series, dictionary, cost, idx):      
    for n in nb.prange(time_series.shape[0]):        
        for a in range(dictionary.shape[0]):
            value = _dot_product(time_series[n], dictionary[a])
            
            # keep maximum value
            if np.abs(value) > np.abs(cost[n]):
                cost[n] = value
                idx[n] = a


@nb.njit(fastmath=True, cache=True)  # pragma: no cover
def _dot_product(x, y):
    z = 0.0 
    for n in range(x.shape[0]):
        z += x[n] * y[n]
        
    return z
