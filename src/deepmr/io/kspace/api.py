"""KSpace IO API."""

__all__ = ["read_rawdata"]

import math
import time

import numpy as np
import torch

from . import gehc as _gehc
from . import mrd as _mrd
# from . import siemens as _siemens

def read_rawdata(filepath, acqheader=None, device="cpu", verbose=0):
    """
    Read kspace data from file.
    
    Currently, handles data written in ISMRMD format [1] (vendor agnostic)
    and GEHC proprietary raw data (requires access to a private repository).

    Parameters
    ----------
    filepath : str
        Path to kspace file.
    acqheader : Header, deepmr.optional
        Acquisition header loaded from trajectory.
        If not provided, assume Cartesian acquisition and infer from data.
        The default is None.
    device : str, optional
        Computational device for internal attributes. The default is "cpu".
    verbose : int, optional
        Verbosity level (0=Silent, 1=Less, 2=More). The default is 0.

    Returns
    -------
    data : torch.tensor
        Complex k-space data.
    head : deepmr.Header
        Metadata for image reconstruction.
        
    Example
    -------
    >>> import deepmr

    Get the filename for an example .mrd file.

    >>> filepath = deepmr.testdata("mrd")

    Load the file contents.

    >>> data, head = deepmr.io.read_rawdata(filepath)

    The result is a data/header pair. 'Data' contains k-space data.
    Here, it represents a 2D spiral acquisition with 1 slice, 36 coils, 32 arms and 1284 samples per arm:
    
    >>> data.shape
    torch.Size([1, 36, 1, 32, 1284])
    
    'Head' contains the acquisition information. We can inspect the k-space trajectory and dcf size,
    the expected image shape and resolution:
    
    >>> head.traj.shape
    torch.Size([1, 32, 1284, 2])
    >>> head.dcf.shape
    torch.Size([1, 32, 1284])
    >>> head.shape
    tensor([  1, 192, 192])
    >>> head.ref_dicom.SliceThickness
    '5.0'
    >>> head.ref_dicom.PixelSpacing
    [1.56, 1.56]

    Notes
    -----
    The returned 'data' tensor contains raw k-space data. Dimensions are defined as following:
        
        * **2Dcart:** (nslices, ncoils, ncontrasts, ny, nx).
        * **2Dnoncart:** (nslices, ncoils, ncontrasts, nviews, nsamples).
        * **3Dcart:** (nx, ncoils, ncontrasts, nz, ny).
        * **3Dnoncart:** (ncoils, ncontrasts, nviews, nsamples).
        
    When possible, data are already pre-processed:
        
        * For Cartesian data (2D and 3D) readout oversampling is removed if the number of samples along readout is larger than the number of rows in the image space (shape[-1]).
        * For Non-Cartesian (2D and 3D), fov is centered according to trajectory and isocenter info from the header.
        * For separable acquisitions (3D stack-of-Non-Cartesians and 3D Cartesians), k-space is decoupled via FFT (along slice and readout axes, respectively).
            
    The returned 'head' (deepmr.io.types.Header) is a structure with the following fields:
    
        * shape (torch.Tensor):
            This is the expected image size of shape (nz, ny, nx).
        * t (torch.Tensor): 
            This is the readout sampling time (0, t_read) in ms.
            with shape (nsamples,).
        * traj (torch.Tensor): 
            This is the k-space trajectory normalized as (-0.5, 0.5) 
            with shape (ncontrasts, nviews, nsamples, ndims).
        * dcf (torch.Tensor): 
            This is the k-space sampling density compensation factor
            with shape (ncontrasts, nviews, nsamples).
        * FA (torch.Tensor, float): 
            This is either the acquisition flip angle in degrees or the list
            of flip angles of shape (ncontrasts,) for each image in the series.
        * TR (torch.Tensor, float): 
            This is either the repetition time in ms or the list
            of repetition times of shape (ncontrasts,) for each image in the series.
        * TE (torch.Tensor, float): 
            This is either the echo time in ms or the list
            of echo times of shape (ncontrasts,) for each image in the series.
        * TI (torch.Tensor, float): 
            This is either the inversion time in ms or the list
            of inversion times of shape (ncontrasts,) for each image in the series.
        * user (dict):
            User parameters. Some examples are:
                
                * ordering (torch.Tensor): 
                    Indices for reordering (acquisition to reconstruction)
                    of acquired k-space data, shaped (3, nslices * ncontrasts * nview), whose rows are
                    'contrast_index', 'slice_index' and 'view_index', respectively.
                * mode (str): 
                    Acquisition mode ('2Dcart', '3Dcart', '2Dnoncart', '3Dnoncart').
                * separable (bool): 
                    Whether the acquisition can be decoupled by fft along slice / readout directions
                    (3D stack-of-noncartesian / 3D cartesian, respectively) or not (3D noncartesian and 2D acquisitions).
                * slice_profile (torch.Tensor): 
                    Flip angle scaling along slice profile of shape (nlocs,).
                * basis (torch.Tensor): 
                    Low rank subspace basis for subspace reconstruction of shape (ncoeff, ncontrasts).
        * affine (np.ndarray): 
            Affine matrix describing image spacing, orientation and origin of shape (4, 4).
        * ref_dicom (pydicom.Dataset): 
            Template dicom for image export.
        * flip (list): 
            List of spatial axis to be flipped after image reconstruction.
            The default is an empty list (no flipping).
        * transpose (list): 
             Permutation of image dimensions after reconstruction, depending on acquisition mode:
                 
                * **2Dcart:** reconstructed image has (nslices, ncontrasts, ny, nx) -> transpose = [1, 0, 2, 3] 
                * **2Dnoncart:** reconstructed image has (nslices, ncontrasts, ny, nx) -> transpose = [1, 0, 2, 3] 
                * **3Dcart:** reconstructed image has (ncontrasts, nz, ny, nx) -> transpose = [0, 1, 2, 3] 
                * **3Dnoncart:** reconstructed image has (nx, ncontrasts, nz, ny) -> transpose = [1, 2, 3, 0] 
            The default is an empty list (no transposition).
            
    References
    ----------
    [1]: Inati, S.J., Naegele, J.D., Zwart, N.R., Roopchansingh, V., Lizak, M.J., Hansen, D.C., Liu, C.-Y., Atkinson, D., 
    Kellman, P., Kozerke, S., Xue, H., Campbell-Washburn, A.E., Sørensen, T.S. and Hansen, M.S. (2017), 
    ISMRM Raw data format: A proposed standard for MRI raw datasets. Magn. Reson. Med., 77: 411-421. 
    https://doi.org/10.1002/mrm.26089
    
    """
    tstart = time.time()
    if verbose >= 1:
        print(f"Reading raw k-space from file {filepath}...", end="\t")

    done = False

    # convert header to numpy
    if acqheader is not None:
        acqheader.numpy()

    # mrd
    if verbose == 2:
        t0 = time.time()
    try:
        data, head = _mrd.read_mrd_rawdata(filepath)
        done = True
    except Exception:
        pass

    # gehc
    if not (done):
        try:
            data, head = _gehc.read_gehc_rawdata(filepath, acqheader)
            done = True
        except Exception:
            pass

    # siemens
    # if not(done):
    #     try:
    #         head = _siemens.read_siemens_rawdata(filepath, acqheader)
    #         done = True
    #     except Exception:
    #         pass

    # check if we loaded data
    if not (done):
        raise RuntimeError(f"File (={filepath}) not recognized!")
    if verbose == 2:
        t1 = time.time()
        print(f"done! Elapsed time: {round(t1-t0, 2)} s")

    # transpose
    data = data.transpose(2, 0, 1, 3, 4)  # (slice, coil, contrast, view, sample)

    # select actual readout
    if verbose == 2:
        nsamples = data.shape[-1]
        print("Selecting actual readout samples...", end="\t")
        t0 = time.time()
    data = _select_readout(data, head)
    if verbose == 2:
        t1 = time.time()
        print(
            f"done! Selected {data.shape[-1]} out of {nsamples} samples. Elapsed time: {round(t1-t0, 2)} s"
        )

    # center fov
    if verbose == 2:
        if head.traj is not None:
            t0 = time.time()
            ndim = head.traj.shape[-1]
            shift = head._shift[:ndim]
            if ndim == 2:
                print(f"Shifting FoV by (dx={shift[0]}, dy={shift[1]}) mm", end="\t")
            if ndim == 3:
                print(
                    f"Shifting FoV by (dx={shift[0]}, dy={shift[1]}, dz={shift[2]}) mm",
                    end="\t",
                )
    data = _fov_centering(data, head)
    if verbose == 2:
        if head.traj is not None:
            t1 = time.time()
            print(f"done! Elapsed time: {round(t1-t0, 2)} s")

    # remove oversampling for Cartesian
    if "mode" in head.user:
        if head.user["mode"][2:] == "cart":
            if verbose == 2:
                t0 = time.time()
                ns1 = data.shape[0]
                ns2 = head.shape[0]
                print(
                    f"Removing oversampling along readout ({round(ns1/ns2, 2)})...",
                    end="\t",
                )
            data, head = _remove_oversampling(data, head)
            if verbose == 2:
                t1 = time.time()
                print(f"done! Elapsed time: {round(t1-t0, 2)} s")

    # transpose readout in slice direction for 3D Cartesian
    if "mode" in head.user:
        if head.user["mode"] == "3Dcart":
            data = data.transpose(
                -1, 1, 2, 0, 3
            )  # (z, ch, e, y, x) -> (x, ch, e, z, y)

    # decouple separable acquisition
    if "separable" in head.user and head.user["separable"]:
        if verbose == 2:
            t0 = time.time()
            print("Separable 3D acquisition, performing FFT along slice...", end="\t")
        data = _fft(data, 0)
        if verbose == 2:
            t1 = time.time()
            print(f"done! Elapsed time: {round(t1-t0, 4)} s")

    # set-up transposition
    if "mode" in head.user:
        if head.user["mode"] == "2Dcart":
            head.transpose = [1, 0, 2, 3]
            if verbose == 2:
                print("Acquisition mode: 2D Cartesian")
                print(
                    f"K-space shape: (nslices={data.shape[0]}, nchannels={data.shape[1]}, ncontrasts={data.shape[2]}, ny={data.shape[3]}, nx={data.shape[4]})"
                )
                print(
                    f"Expected image shape: (nslices={data.shape[0]}, nchannels={data.shape[1]}, ncontrasts={data.shape[2]}, ny={head.shape[1]}, nx={head.shape[2]})"
                )
        elif head.user["mode"] == "2Dnoncart":
            head.transpose = [1, 0, 2, 3]
            if verbose == 2:
                print("Acquisition mode: 2D Non-Cartesian")
                print(
                    f"K-space shape: (nslices={data.shape[0]}, nchannels={data.shape[1]}, ncontrasts={data.shape[2]}, nviews={data.shape[3]}, nsamples={data.shape[4]})"
                )
                print(
                    f"Expected image shape: (nslices={data.shape[0]}, nchannels={data.shape[0]}, ncontrasts={data.shape[1]}, ny={head.shape[1]}, nx={head.shape[2]})"
                )
        elif head.user["mode"] == "3Dnoncart":
            data = data[0]
            head.transpose = [1, 0, 2, 3]
            if verbose == 2:
                print("Acquisition mode: 3D Non-Cartesian")
                print(
                    f"K-space shape: (nchannels={data.shape[0]}, ncontrasts={data.shape[1]}, nviews={data.shape[2]}, nsamples={data.shape[3]})"
                )
                print(
                    f"Expected image shape: (nchannels={data.shape[0]}, ncontrasts={data.shape[1]}, nz={head.shape[0]}, ny={head.shape[1]}, nx={head.shape[2]})"
                )
        elif head.user["mode"] == "3Dcart":
            head.transpose = [1, 2, 3, 0]
            if verbose == 2:
                print("Acquisition mode: 3D Cartesian")
                print(
                    f"K-space shape: (nx={data.shape[0]}, nchannels={data.shape[1]}, ncontrasts={data.shape[2]}, nz={data.shape[3]}, ny={data.shape[4]})"
                )
                print(
                    f"Expected image shape: (nx={head.shape[2]}, nchannels={data.shape[1]}, ncontrasts={data.shape[2]}, nz={head.shape[0]}, ny={head.shape[1]})"
                )

        # remove unused trajectory for cartesian
        if head.user["mode"][2:] == "cart":
            head.traj = None
            head.dcf = None

    # clean header
    head.user.pop("mode", None)
    head.user.pop("separable", None)

    # final report
    if verbose == 2:
        print(f"Readout time: {round(float(head.t[-1]), 2)} ms")
        if head.traj is not None:
            print(f"Trajectory range: ({head.traj.min()},{head.traj.max()})")
            print(
                f"Trajectory shape: (ncontrasts={head.traj.shape[0]}, nviews={head.traj.shape[1]}, nsamples={head.traj.shape[2]}, ndim={head.traj.shape[-1]})"
            )
        if head.dcf is not None:
            print(
                f"DCF shape: (ncontrasts={head.dcf.shape[0]}, nviews={head.dcf.shape[1]}, nsamples={head.dcf.shape[2]})"
            )
    if head.FA is not None:
        if len(np.unique(head.FA)) > 1:
            if verbose == 2:
                print(f"Flip Angle train length: {len(head.FA)}")
        else:
            FA = float(np.unique(head.FA)[0])
            head.FA = FA
            if verbose == 2:
                print(f"Constant FA: {round(abs(FA), 2)} deg")
    if head.TR is not None:
        if len(np.unique(head.TR)) > 1:
            if verbose == 2:
                print(f"TR train length: {len(head.TR)}")
        else:
            TR = float(np.unique(head.TR)[0])
            head.TR = TR
            if verbose == 2:
                print(f"Constant TR: {round(TR, 2)} ms")
    if head.TE is not None:
        if len(np.unique(head.TE)) > 1:
            if verbose == 2:
                print(f"Echo train length: {len(head.TE)}")
        else:
            TE = float(np.unique(head.TE)[0])
            head.TE = TE
            if verbose == 2:
                print(f"Constant TE: {round(TE, 2)} ms")
    if head.TI is not None and np.allclose(head.TI, 0.0) is False:
        if len(np.unique(head.TI)) > 1:
            if verbose == 2:
                print(f"Inversion train length: {len(head.TI)}")
        else:
            TI = float(np.unique(head.TI)[0])
            head.TI = TI
            if verbose == 2:
                print(f"Constant TI: {round(TI, 2)} ms")

    # cast
    data = torch.as_tensor(
        np.ascontiguousarray(data), dtype=torch.complex64, device=device
    )
    head.torch(device)

    tend = time.time()
    if verbose == 1:
        print(f"done! Elapsed time: {round(tend-tstart, 2)} s")
    elif verbose == 2:
        print(f"Total elapsed time: {round(tend-tstart, 2)} s")

    return data, head


# %% sub routines
def _select_readout(data, head):
    if head._adc is not None:
        if head._adc[-1] == data.shape[-1]:
            data = data[..., head._adc[0] :]
        else:
            data = data[..., head._adc[0] : head._adc[1] + 1]
    return data


def _fov_centering(data, head):
    if head.traj is not None and np.allclose(head._shift, 0.0) is False:
        # ndimensions
        ndim = head.traj.shape[-1]

        # shift (mm)
        dr = np.asarray(head._shift)[:ndim]

        # convert in units of voxels
        dr /= head._resolution[::-1][:ndim]

        # apply
        data *= np.exp(1j * 2 * math.pi * (head.traj * dr).sum(axis=-1))

    return data


def _remove_oversampling(data, head):
    if data.shape[-1] != head.shape[-1]:  # oversampled
        center = int(data.shape[-1] // 2)
        hwidth = int(head.shape[-1] // 2)
        data = _fft(data, -1)
        data = data[..., center - hwidth : center + hwidth]
        data = _fft(data, -1)
        dt = np.diff(head.t)[0]
        head.t = np.linspace(0, head.t[-1], data.shape[-1])

    return data, head


def _fft(data, axis):
    tmp = torch.as_tensor(data)
    tmp = torch.fft.fftshift(
        torch.fft.fft(torch.fft.fftshift(tmp, dim=axis), dim=axis), dim=axis
    )
    return tmp.numpy()
