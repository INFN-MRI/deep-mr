"""Test image space I/O routines."""

import os
import tempfile

import numpy as np
import numpy.testing as npt

import deepmr

def test_dicom_read():
    path = deepmr.testdata("dicom")
    image, head = deepmr.io.read_image(path)
    
    # image shape
    npt.assert_allclose(image.shape, [3,2, 192, 192])
    
    # contrast info
    npt.assert_allclose(head.FA, 180.0)
    npt.assert_allclose(head.TE, [20.0, 40.0, 60.0])
    npt.assert_allclose(head.TR, 3000.0)
    assert np.isinf(head.TI)
    
    # geometry
    npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 10.5)
    npt.assert_allclose(head.ref_dicom.SliceThickness, 7.0)
    npt.assert_allclose(head.ref_dicom.PixelSpacing, [0.67, 0.67])
    npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, -0.0, 0.0, 0.0, 1.0, 0.0])
    
    # patient info
    assert head.ref_dicom.PatientName == "phantom"
    assert head.ref_dicom.PatientBirthDate == "20000125"
    assert head.ref_dicom.PatientAge == "010Y"

def test_dicom_write():
    ipath = deepmr.testdata("dicom")
    iimage, ihead = deepmr.io.read_image(ipath)
    
    with tempfile.TemporaryDirectory() as tempdir:
        opath = os.path.join(tempdir, 'dicomtest')
        deepmr.io.write_image(opath, iimage, ihead, dataformat="dicom")
        
        image, head = deepmr.io.read_image(opath)
        
        # image shape
        npt.assert_allclose(image.shape, [3,2, 192, 192])
        
        # contrast info
        npt.assert_allclose(head.FA, 180.0)
        npt.assert_allclose(head.TE, [20.0, 40.0, 60.0])
        npt.assert_allclose(head.TR, 3000.0)
        assert np.isinf(head.TI)
        
        # geometry
        npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 10.5)
        npt.assert_allclose(head.ref_dicom.SliceThickness, 7.0)
        npt.assert_allclose(head.ref_dicom.PixelSpacing, [0.67, 0.67])
        npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, -0.0, 0.0, 0.0, 1.0, 0.0])
        
        # patient info
        assert head.ref_dicom.PatientName == "phantom"
        assert head.ref_dicom.PatientBirthDate == "20000125"
        assert head.ref_dicom.PatientAge == "010Y"
        
def test_dicom_write_nohead():
    ipath = deepmr.testdata("dicom")
    iimage, _ = deepmr.io.read_image(ipath)
    
    with tempfile.TemporaryDirectory() as tempdir:
        opath = os.path.join(tempdir, 'dicomtest')
        deepmr.io.write_image(opath, iimage, dataformat="dicom")
        
        image, head = deepmr.io.read_image(opath)
        
        # image shape
        npt.assert_allclose(image.shape, [3,2, 192, 192])
        
        # contrast info
        npt.assert_allclose(head.FA, 90.0)
        npt.assert_allclose(head.TE, 0.0)
        npt.assert_allclose(head.TR, 1000.0)
        assert np.isinf(head.TI)
        
        # geometry
        npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 1.0)
        npt.assert_allclose(head.ref_dicom.SliceThickness, 1.0)
        npt.assert_allclose(head.ref_dicom.PixelSpacing, [1.0, 1.0])
        npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        
        # patient info
        assert head.ref_dicom.PatientName == "Lastname^Firstname"
        assert head.ref_dicom.PatientBirthDate == ""
        assert head.ref_dicom.PatientAge == ""

def test_dicom_write_anon():
    ipath = deepmr.testdata("dicom")
    iimage, ihead = deepmr.io.read_image(ipath)
    
    with tempfile.TemporaryDirectory() as tempdir:
        opath = os.path.join(tempdir, 'dicomtest')
        deepmr.io.write_image(opath, iimage, ihead, dataformat="dicom", anonymize=True)
        
        image, head = deepmr.io.read_image(opath)
        
        # image shape
        npt.assert_allclose(image.shape, [3,2, 192, 192])
        
        # contrast info
        npt.assert_allclose(head.FA, 180.0)
        npt.assert_allclose(head.TE, [20.0, 40.0, 60.0])
        npt.assert_allclose(head.TR, 3000.0)
        assert np.isinf(head.TI)
        
        # geometry
        npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 10.5)
        npt.assert_allclose(head.ref_dicom.SliceThickness, 7.0)
        npt.assert_allclose(head.ref_dicom.PixelSpacing, [0.67, 0.67])
        npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, -0.0, 0.0, 0.0, 1.0, 0.0])
        
        # patient info
        assert head.ref_dicom.PatientName == ""
        assert head.ref_dicom.PatientBirthDate == ""
        assert head.ref_dicom.PatientAge == "010Y"

def test_nifti_read():
    path = deepmr.testdata("nifti")
    image, head = deepmr.io.read_image(path)
    
    # image shape
    npt.assert_allclose(image.shape, [3,2, 192, 192])
    
    # contrast info
    npt.assert_allclose(head.FA, 180.0)
    npt.assert_allclose(head.TE, [20.0, 40.0, 60.0])
    npt.assert_allclose(head.TR, 3000.0)
    assert np.isinf(head.TI)
    
    # geometry
    npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 10.5)
    npt.assert_allclose(head.ref_dicom.SliceThickness, 7.0)
    npt.assert_allclose(head.ref_dicom.PixelSpacing, [0.67, 0.67])
    npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, -0.0, 0.0, 0.0, 1.0, 0.0])
    
    # patient info
    assert head.ref_dicom.PatientName == "Lastname^Firstname"
    assert head.ref_dicom.PatientBirthDate == ""
    assert head.ref_dicom.PatientAge == ""

def test_nifti_write():
    ipath = deepmr.testdata("dicom")
    iimage, ihead = deepmr.io.read_image(ipath)
    
    with tempfile.TemporaryDirectory() as tempdir:
        opath = os.path.join(tempdir, 'niftitest.nii')
        deepmr.io.write_image(opath, iimage, ihead, dataformat="nifti")
        
        image, head = deepmr.io.read_image(opath)
        
        # image shape
        npt.assert_allclose(image.shape, [3,2, 192, 192])
        
        # contrast info
        npt.assert_allclose(head.FA, 180.0)
        npt.assert_allclose(head.TE, [20.0, 40.0, 60.0])
        npt.assert_allclose(head.TR, 3000.0)
        assert np.isinf(head.TI)
        
        # geometry
        npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 10.5)
        npt.assert_allclose(head.ref_dicom.SliceThickness, 7.0)
        npt.assert_allclose(head.ref_dicom.PixelSpacing, [0.67, 0.67])
        npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, -0.0, 0.0, 0.0, 1.0, 0.0])
        
        # patient info
        assert head.ref_dicom.PatientName == "phantom"
        assert head.ref_dicom.PatientBirthDate == "20000125"
        assert head.ref_dicom.PatientAge == "010Y"
        
def test_dicom_nifti_nohead():
    ipath = deepmr.testdata("dicom")
    iimage, _ = deepmr.io.read_image(ipath)
    
    with tempfile.TemporaryDirectory() as tempdir:
        opath = os.path.join(tempdir, 'niftitest.nii')
        deepmr.io.write_image(opath, iimage, dataformat="nifti")
        
        image, head = deepmr.io.read_image(opath)
        
        # image shape
        npt.assert_allclose(image.shape, [3,2, 192, 192])
        
        # contrast info
        npt.assert_allclose(head.FA, 90.0)
        npt.assert_allclose(head.TE, 0.0)
        npt.assert_allclose(head.TR, 1000.0)
        assert np.isinf(head.TI)
        
        # geometry
        npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 1.0)
        npt.assert_allclose(head.ref_dicom.SliceThickness, 1.0)
        npt.assert_allclose(head.ref_dicom.PixelSpacing, [1.0, 1.0])
        npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        
        # patient info
        assert head.ref_dicom.PatientName == "Lastname^Firstname"
        assert head.ref_dicom.PatientBirthDate == ""
        assert head.ref_dicom.PatientAge == ""

def test_dicom_nifti_anon():
    ipath = deepmr.testdata("dicom")
    iimage, ihead = deepmr.io.read_image(ipath)
    
    with tempfile.TemporaryDirectory() as tempdir:
        opath = os.path.join(tempdir, 'niftitest.nii')
        deepmr.io.write_image(opath, iimage, ihead, dataformat="nifti", anonymize=True)
        
        image, head = deepmr.io.read_image(opath)
        
        # image shape
        npt.assert_allclose(image.shape, [3,2, 192, 192])
        
        # contrast info
        npt.assert_allclose(head.FA, 180.0)
        npt.assert_allclose(head.TE, [20.0, 40.0, 60.0])
        npt.assert_allclose(head.TR, 3000.0)
        assert np.isinf(head.TI)
        
        # geometry
        npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 10.5)
        npt.assert_allclose(head.ref_dicom.SliceThickness, 7.0)
        npt.assert_allclose(head.ref_dicom.PixelSpacing, [0.67, 0.67])
        npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, -0.0, 0.0, 0.0, 1.0, 0.0])
        
        # patient info
        assert head.ref_dicom.PatientName == ""
        assert head.ref_dicom.PatientBirthDate == ""
        assert head.ref_dicom.PatientAge == "010Y"
