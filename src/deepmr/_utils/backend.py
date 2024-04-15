"""Support PyTorch-Numba conversion subroutines.."""

__all__ = ["numba2pytorch", "pytorch2numba"]

import torch
import numba as nb


def numba2pytorch(array, requires_grad=False):
    """
    Zero-copy conversion from Numpy/Numba CUDAarray to PyTorch tensor.

    Parameters
    ----------
    array : np.ndarray
        Input array.
    requires_grad : bool, optional
        Set requires_grad output tensor. The default is False.

    Returns
    -------
    output : torch.Tensor
        Output tensor.

    """
    if torch.cuda.is_available() is True:
        if nb.cuda.is_cuda_array(array) is True:
            index = nb.cuda.get_current_device().id
            tensor = torch.as_tensor(array, device="cuda:" + str(index))
        else:
            tensor = torch.as_tensor(array)  # pylint: disable=no-member
    else:
        tensor = torch.as_tensor(array)  # pylint: disable=no-member

    tensor.requires_grad = requires_grad
    return tensor.contiguous()


def pytorch2numba(tensor):
    """
    Zero-copy conversion from PyTorch tensor to Numpy/Numba CUDA array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    output : np.ndarray
        Output array.

    """
    device = tensor.device
    if device.type == "cpu":
        array = tensor.detach().contiguous().numpy()
    else:
        with nb.cuda.devices.gpus[device.index]:
            array = nb.cuda.as_cuda_array(tensor.detach().contiguous())

    return array
