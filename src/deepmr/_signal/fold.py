"""Patch extraction routines."""

__all__ = ["tensor2patches", "patches2tensor"]

import numpy as np
import torch

from .._external.unfoldNd.fold import foldNd


def tensor2patches(image, patch_shape, patch_stride=None):
    """
    View tensor as overlapping hyperectangular patches, with a given stride.

    Adapted from [1, 2].

    Parameters
    ----------
    image : torch.Tensor
        N-dimensional image tensor, with the last ``ndim`` dimensions
        being the image dimensions.
    patch_shape : Iterable[int]
        Shape of the patch of length ``ndim``.
    patch_stride : Iterable[int], optional
        Stride of the windows of length ``ndim``.
        The default it is the patch size (i.e., non overlapping).

    Returns
    -------
    patches : torch.Tensor
        Tensor of (overlapping) patches of shape:

        * ``1D: (..., npatches_z, patch_size_x)``
        * ``2D: (..., npatches_z, npatches_y, patch_size_y, patch_size_x)``
        * ``3D: (..., npatches_z, npatches_y, npatches_x, patch_size_z, patch_size_y, patch_size_x)``

    References
    ----------
    [1] https://stackoverflow.com/questions/64462917/view-as-windows-from-skimage-but-in-pytorch \n
    [2] https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/10

    """
    # be sure it is a tensor
    image = torch.as_tensor(image)

    # default stride
    if patch_stride is None:
        patch_stride = patch_shape

    # cast to array
    patch_shape = np.asarray(patch_shape)
    patch_stride = np.asarray(patch_stride)

    # verify that strides and shapes are > 0
    assert np.all(patch_shape > 0), f"Patch shape must be > 0; got {patch_shape}"
    assert np.all(patch_stride > 0), f"Patch stride must be > 0; got {patch_stride}"
    assert np.all(
        patch_stride <= patch_shape
    ), "We do not support non-overlapping or non-contiguous patches."

    # get number of dimensions
    ndim = len(patch_shape)
    batch_shape = image.shape[:-ndim]

    # count number of patches for each dimension
    ishape = np.asarray(image.shape[-ndim:])
    num_patches = np.ceil(ishape / patch_stride)
    num_patches = num_patches.astype(int)

    # pad if required
    padsize = ((num_patches - 1) * patch_stride + patch_shape) - ishape
    padsize = np.stack((0 * padsize, padsize), axis=-1)
    padsize = padsize.ravel()
    patches = torch.nn.functional.pad(image, tuple(padsize))

    # get reshape to (b, nz, ny, nx), (b, ny, nx), (b, nx) for 3, 2, and 1D, respectively
    patches = patches.view(int(np.prod(batch_shape)), *patches.shape[-ndim:])

    if ndim == 3:
        kc, kh, kw = patch_shape  # kernel size
        dc, dh, dw = patch_stride  # stride
        patches = patches.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    elif ndim == 2:
        kh, kw = patch_shape  # kernel size
        dh, dw = patch_stride  # stride
        patches = patches.unfold(1, kh, dh).unfold(2, kw, dw)
    elif ndim == 1:
        kw = patch_shape  # kernel size
        dw = patch_stride  # stride
        patches = patches.unfold(1, kw, dw)
    else:
        raise ValueError(f"Only support ndim=1, 2, or 3, got {ndim}")

    # reformat
    patches = patches.reshape(*batch_shape, *patches.shape[1:])

    return patches


def patches2tensor(patches, shape, patch_shape, patch_stride=None):
    """
    Accumulate patches into a tensor.

    Adapted from [1] using [2].

    Parameters
    ----------
    patches : torch.Tensor
        Tensor of (overlapping) patches of shapes:

        * ``1D: (..., npatches_z, patch_size_x)``
        * ``2D: (..., npatches_z, npatches_y, patch_size_y, patch_size_x)``
        * ``3D: (..., npatches_z, npatches_y, npatches_x, patch_size_z, patch_size_y, patch_size_x)``

    shape : Iterable[int]
        Output shape of length ``ndim``.
        If scalar, assume isotropic matrix of shape ``ndim * [shape]``.
    patch_shape : Iterable[int]
        Shape of the patch of length ``ndim``.
    patch_stride : Iterable[int], optional
        Stride of the windows of length ``ndim``.
        The default it is the patch size (i.e., non overlapping).

    Returns
    -------
    image : torch.Tensor
        N-dimensional image tensor, with the last ``ndim`` dimensions
        being the image dimensions.

    References
    ----------
    [1] https://discuss.pytorch.org/t/how-to-split-tensors-with-overlap-and-then-reconstruct-the-original-tensor/70261 \n
    [2] https://github.com/f-dangel/unfoldNd

    """
    # be sure it is a tensor
    patches = torch.as_tensor(patches)

    # default stride
    if patch_stride is None:
        patch_stride = patch_shape

    # cast to array
    patch_shape = np.asarray(patch_shape)
    patch_stride = np.asarray(patch_stride)

    # verify that strides and shapes are > 0
    assert np.all(patch_shape > 0), f"Patch shape must be > 0; got {patch_shape}"
    assert np.all(patch_stride > 0), f"Patch stride must be > 0; got {patch_stride}"
    assert np.all(
        patch_stride <= patch_shape
    ), "We do not support non-overlapping or non-contiguous patches."

    # get number of dimensions
    ndim = len(shape)
    batch_shape = patches.shape[: -2 * ndim]

    # count number of patches for each dimension
    ishape = np.asarray(shape)
    num_patches = np.ceil(ishape / patch_stride)
    num_patches = num_patches.astype(int)

    # pad if required
    padsize = ((num_patches - 1) * patch_stride + patch_shape) - ishape
    padded_shape = shape + padsize

    # perform unfolding
    if np.allclose(patch_shape, patch_stride):
        image = _fold_nonoverlapping(patches, ndim, padded_shape, batch_shape)
    else:
        image = _fold_overlapping(
            patches, padded_shape, batch_shape, patch_shape, patch_stride
        )

    # crop
    if ndim == 1:
        image = image[:, : shape[0]]
    elif ndim == 2:
        image = image[:, : shape[0], : shape[1]]
    elif ndim == 3:
        image = image[:, : shape[0], : shape[1], : shape[2]]
    else:
        raise ValueError(f"Only support ndim=1, 2, or 3, got {ndim}")

    # final reshape
    image = image.reshape(*batch_shape, *shape)

    return image


# %% local subroutines
def _fold_overlapping(patches, padded_shape, batch_shape, patch_shape, patch_stride):
    # get reshape to (b, nz, ny, nx), (b, ny, nx), (b, nx) for 3, 2, and 1D, respectively
    patches = patches.reshape(int(np.prod(batch_shape)), -1, int(np.prod(patch_shape)))
    patches = patches.permute(0, 2, 1)

    # get image
    weight = foldNd(
        torch.ones_like(patches[[0]]),
        tuple(padded_shape),
        tuple(patch_shape),
        stride=tuple(patch_stride),
    )
    image = foldNd(
        patches, tuple(padded_shape), tuple(patch_shape), stride=tuple(patch_stride)
    )

    # get rid of channel dim
    weight = weight[0, 0]
    image = image[:, 0]

    # final reshape
    image = image.reshape(-1, *padded_shape)
    weight = weight.reshape(*padded_shape)

    return (image / weight).to(patches.dtype)


def _fold_nonoverlapping(patches, ndim, padded_shape, batch_shape):
    # get reshape to (b, nz, ny, nx), (b, ny, nx), (b, nx) for 3, 2, and 1D, respectively
    unfold_shape = patches.shape[-2 * ndim :]
    patches = patches.view(int(np.prod(batch_shape)), *unfold_shape)

    if ndim == 3:
        nz = unfold_shape[0] * unfold_shape[3]
        ny = unfold_shape[1] * unfold_shape[4]
        nx = unfold_shape[2] * unfold_shape[5]
        patches = patches.permute(0, 1, 4, 2, 5, 3, 6)
        image = patches.reshape(-1, nz, ny, nx)
    elif ndim == 2:
        ny = unfold_shape[0] * unfold_shape[2]
        nx = unfold_shape[1] * unfold_shape[3]
        patches = patches.permute(0, 1, 3, 2, 4)
        image = patches.reshape(-1, ny, nx)
    elif ndim == 1:
        nx = unfold_shape[0] * unfold_shape[1]
        image = patches.reshape(-1, nx)
    else:
        raise ValueError(f"Only support ndim=1, 2, or 3, got {ndim}")

    # final reshape
    image = image.reshape(-1, *padded_shape)

    return image
