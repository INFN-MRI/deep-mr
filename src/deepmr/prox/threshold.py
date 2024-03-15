
__all__ = ["soft_thresh", "hard_thresh"]

import torch

def soft_thresh(input, ths=0.1):
    r"""
    Soft thresholding function.

    Arguments
    ---------
    input : torch.Tensor 
        Input data.
    ths : float, optional
        Threshold. It can be element-wise, in which case
        it is assumed to be broadcastable with ``input``.
        The default is ``0.1``.
        
    Returns
    -------
    output : torch.Tensor
        Output data.
        
    """
    sign = torch.sign(input)

    output = abs(input) - ths
    output = (abs(output) + output) / 2.0

    return output * sign
        
def hard_thresh(input, ths=0.1):
    r"""
    Hard thresholding function.

    Arguments
    ---------
    input : torch.Tensor 
        Input data.
    ths : float, optional
        Threshold. It can be element-wise, in which case
        it is assumed to be broadcastable with ``input``.
        The default is ``0.1``.
        
    Returns
    -------
    output : torch.Tensor
        Output data.
        
    """
    output = input.clone()
    output[abs(output) < ths] = 0.0
    return output



