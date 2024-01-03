"""This module contains DICOM writing routines."""
__all__ = ["write"]

import copy
import multiprocessing
import os

from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pydicom

from . import _subroutines

def write(
    image: np.ndarray,
    info: dict,
    series_description: str,
    outpath: str = "./output",
    series_number_scale=1000,
    series_number_offset=0,
):
    """
    Write parametric map to dicom.

    Args:
        image: ndarray of image data to be written.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - TI: ndarray of Inversion Times [ms].
            - TE: ndarray of Echo Times [ms].
            - TR: ndarray of Repetition Times [ms].
            - FA: ndarray of Flip Angles [deg].
        outpath: desired output path
    """
    if info["dcm_template"]:
        # generate UIDs
        SeriesInstanceUID = pydicom.uid.generate_uid()

        # count number of instances
        ninstances = image.shape[0]

        # init dsets
        dsets = copy.deepcopy(info["dcm_template"])

        # generate series number
        series_number = str(
            series_number_scale * int(dsets[0].SeriesNumber) + series_number_offset
        )

        # cast image
        minval = np.iinfo(np.int16).min
        maxval = np.iinfo(np.int16).max
        image[image < minval] = minval
        image[image > maxval] = maxval
        image = image.astype(np.int16)

        # get level
        windowMin = np.percentile(image, 5)
        windowMax = np.percentile(image, 95)
        windowWidth = windowMax - windowMin

        # set properties
        for n in range(ninstances):
            dsets[n].pixel_array[:] = image[n]
            dsets[n].PixelData = image[n].tobytes()

            dsets[n].WindowWidth = str(windowWidth)
            dsets[n].WindowCenter = str(0.5 * windowWidth)

            dsets[n].SeriesDescription = series_description
            dsets[n].SeriesNumber = series_number
            dsets[n].SeriesInstanceUID = SeriesInstanceUID

            dsets[n].SOPInstanceUID = pydicom.uid.generate_uid()
            dsets[n].InstanceNumber = str(n + 1)

            try:
                dsets[n].ImagesInAcquisition = ninstances
            except:
                pass
            try:
                dsets[n][0x0025, 0x1007].value = ninstances
            except:
                pass
            try:
                dsets[n][0x0025, 0x1019].value = ninstances
            except:
                pass

        # generate file names
        filename = ["img-" + str(n).zfill(3) + ".dcm" for n in range(ninstances)]

        # generate output path
        outpath = os.path.abspath(outpath)

        # create output folder
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # get dicompath
        dcm_paths = [os.path.join(outpath, file) for file in filename]

        # generate path / data pair
        path_data = [[dcm_paths[n], dsets[n]] for n in range(ninstances)]

        # make pool of workers
        pool = ThreadPool(multiprocessing.cpu_count())

        # each thread write a dicom
        dsets = pool.map(_subroutines._dcmwrite, path_data)

        # cloose pool and wait finish
        pool.close()
        pool.join()
