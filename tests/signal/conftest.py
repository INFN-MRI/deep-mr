""" Configuration utils for test suite. """

import torch


# %% FFT
def _fftc(x, ax):
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=ax), dim=ax, norm="ortho"), dim=ax
    )


def _ifftc(x, ax):
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x, dim=ax), dim=ax, norm="ortho"), dim=ax
    )


# %% data generator
class _kt_space_trajectory:
    def __init__(self, ndim, ncontrasts, npix):
        # data type
        dtype = torch.float32

        # build coordinates
        nodes = torch.arange(npix) - (npix // 2)

        if ndim == 1:
            coord = nodes[..., None]
        elif ndim == 2:
            x_i, y_i = torch.meshgrid(nodes, nodes, indexing="ij")
            x_i = x_i.flatten()
            y_i = y_i.flatten()
            coord = torch.stack((x_i, y_i), axis=-1).to(dtype)
        elif ndim == 3:
            x_i, y_i, z_i = torch.meshgrid(nodes, nodes, nodes, indexing="ij")
            x_i = x_i.flatten()
            y_i = y_i.flatten()
            z_i = z_i.flatten()
            coord = torch.stack((x_i, y_i, z_i), axis=-1).to(dtype)

        # assume single shot trajectory
        coord = coord[None, ...]  # (nview=1, nsamples=npix**ndim, ndim=ndim)
        if ncontrasts > 1:
            coord = torch.repeat_interleave(coord[None, ...], ncontrasts, axis=0)
            
        # normalize
        cmax = (coord**2).sum(axis=-1)**0.5
        coord = coord / cmax.max() / 2

        # reshape coordinates and build dcf / matrix size
        self.coordinates = coord
        self.density_comp_factor = torch.ones(self.coordinates.shape[:-1], dtype=dtype)
        self.acquisition_matrix = npix

def _kt_space_data(ndim, ncontrasts, ncoils, nslices, npix, device):
    # data type
    dtype = torch.complex64

    if ncontrasts == 1:
        if ndim == 2:
            data = torch.ones(
                (nslices, ncoils, 1, (npix**ndim)), dtype=dtype, device=device
            )
        elif ndim == 3:
            data = torch.ones((ncoils, 1, (npix**ndim)), dtype=dtype, device=device)
    else:
        if ndim == 2:
            data = torch.ones(
                (nslices, ncoils, ncontrasts, 1, (npix**ndim)),
                dtype=dtype,
                device=device,
            )
        elif ndim == 3:
            data = torch.ones(
                (ncoils, ncontrasts, 1, (npix**ndim)), dtype=dtype, device=device
            )

    return data


def _image(ndim, ncontrasts, ncoils, nslices, npix, device):
    # data type
    dtype = torch.complex64
    center = int(npix // 2)

    if ncontrasts == 1:
        if ndim == 2:
            img = torch.zeros((nslices, ncoils, npix, npix), dtype=dtype, device=device)
            img[:, :, center, center] = 1

        elif ndim == 3:
            img = torch.zeros((ncoils, npix, npix, npix), dtype=dtype, device=device)
            img[:, center, center, center] = 1
    else:
        if ndim == 2:
            img = torch.zeros(
                (nslices, ncoils, ncontrasts, npix, npix), dtype=dtype, device=device
            )
            img[:, :, :, center, center] = 1

        elif ndim == 3:
            img = torch.zeros(
                (ncoils, ncontrasts, npix, npix, npix), dtype=dtype, device=device
            )
            img[:, :, center, center, center] = 1

    return img


def _lowrank_subspace_projection(dtype, nframes):
    return torch.eye(nframes, dtype=dtype)
