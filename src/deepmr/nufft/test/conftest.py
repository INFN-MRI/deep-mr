""" Configuration utils for test suite. """
# pylint: disable=no-member
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments


import torch


# %% data generator
class _kt_space_trajectory:

    def __init__(self, ndim, nframes, npix):

        # data type
        dtype = torch.float32
        
        # build coordinates
        nodes = torch.arange(npix) - (npix // 2)

        if ndim == 2:
            x_i, y_i = torch.meshgrid(nodes, nodes, indexing='ij')
            x_i = x_i.flatten()
            y_i = y_i.flatten()
            coord = torch.stack((x_i, y_i), axis=-1).to(dtype)

        elif ndim == 3:
            x_i, y_i, z_i = torch.meshgrid(nodes, nodes, nodes, indexing='ij')
            x_i = x_i.flatten()
            y_i = y_i.flatten()
            z_i = z_i.flatten()
            coord = torch.stack((x_i, y_i, z_i), axis=-1).to(dtype)

        coord = torch.repeat_interleave(coord[None, :, :], nframes, axis=0)

        # reshape coordinates and build dcf / matrix size
        self.coordinates = coord
        
        if ndim == 2:
            self.density_comp_factor = torch.ones(self.coordinates.shape[:-1], dtype=dtype)[:, None, None, None, :]
            
        elif ndim == 3:
            self.density_comp_factor = torch.ones(self.coordinates.shape[:-1], dtype=dtype)[:, None, None, :]

        self.acquisition_matrix = npix


def _kt_space_data(ndim, nframes, nechoes, ncoils, nslices, npix, device):

    # data type
    dtype = torch.complex64

    if ndim == 2:
        data = torch.ones((nframes, nechoes, ncoils, nslices, (npix**ndim)), dtype=dtype, device=device)

    elif ndim == 3:
        data = torch.ones((nframes, nechoes, ncoils, (npix**ndim)), dtype=dtype, device=device)

    return data


def _image(ndim, nframes, nechoes, ncoils, nslices, npix, device):

    # data type
    dtype = torch.complex64
    center = int(npix // 2)

    if ndim == 2:
        img = torch.zeros((nframes, nechoes, ncoils, nslices, npix, npix), dtype=dtype, device=device)
        img[:, :, :, :, center, center] = 1

    elif ndim == 3:
        img = torch.zeros((nframes, nechoes, ncoils, npix, npix, npix), dtype=dtype, device=device)
        img[:, :, :, center, center, center] = 1

    return img


def _lowrank_subspace_projection(dtype, nframes):
    return torch.eye(nframes, dtype=dtype)