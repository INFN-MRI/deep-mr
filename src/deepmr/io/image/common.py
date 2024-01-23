""" DICOM/NIfTI common IO subroutines."""

import numpy as np
import torch

def _prepare_image(image, transpose=None, flip=None, rescale=False):
    
    # cast image to numpy
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    # maske sure we are working with magnitude
    if np.iscomplexobj(image):
        image = abs(image)
    
    # reformat image
    if transpose is not None:
        image = image.transpose(*transpose)
    if flip is not None:
        image = np.flip(image, axis=flip)
    
    # rescale
    minval = np.iinfo(np.int16).min
    maxval = np.iinfo(np.int16).max
    
    if rescale:
        image -= image.min() # (min, max) -> (0, max-min)
        image /= image.max() # (0, max-min) -> (0, 1.0)
        image *= 0.95 * maxval
        
    # else:
    #     if np.any(image < minval):
    #         n = np.sum(image < minval)
    #         ntot = image.size
    #         raise UserWarning(f"{n} out of {ntot} pixels are below the minimum intensity value {minval}. If you are working with quantitative values, make sure you are using a suitable measurement unit (e.g, 'ms' instead of 's').")
    
    # clip values outside range and cast
    image[image < minval] = minval
    image[image > maxval] = maxval
    image = image.astype(np.int16)
    
    try:
        windowMin = round(0.5 * np.percentile(image[image < 0], 95))
    except:
        windowMin = 0
    try:
        windowMax = round(0.5 * np.percentile(image[image > 0], 95))
    except:
        windowMax = 0   
        
    return image, (windowMin, windowMax)
           
def _anonymize(head):
    head.ref_dicom.PatientName = ""
    head.ref_dicom.PatientBirthDate = ""            
    head.ref_dicom.PatientWeight = "" 
    head.ref_dicom.PatientID = "Anon" 
    return head