"""Test k-space I/O routines."""

import numpy.testing as npt

import deepmr


def test_mrd_read():
    path = deepmr.testdata("mrd")
    data, head = deepmr.io.read_rawdata(path)

    # image shape
    npt.assert_allclose(data.shape, [1, 36, 1, 32, 1284])
    npt.assert_allclose(head.traj.shape, [1, 32, 1284, 2])
    npt.assert_allclose(head.dcf.shape, [1, 32, 1284])

    # contrast info
    npt.assert_allclose(head.FA, 10.0)
    npt.assert_allclose(head.TE, 0.86)
    npt.assert_allclose(head.TR, 4.96)

    # geometry
    npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 5.0)
    npt.assert_allclose(head.ref_dicom.SliceThickness, 5.0)
    npt.assert_allclose(head.ref_dicom.PixelSpacing, [1.56, 1.56])
    npt.assert_allclose(
        head.ref_dicom.ImageOrientationPatient, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    )

    # patient info
    assert head.ref_dicom.PatientName == "Lastname^Firstname"
    assert head.ref_dicom.PatientBirthDate == ""
    assert head.ref_dicom.PatientAge == ""


# def test_gehc_pfile():
#     path = deepmr.testdata("gehc::pfile")
#     data, head = deepmr.io.read_rawdata(path)

#     # image shape
#     npt.assert_allclose(data.shape, [7, 8, 1, 128, 128])

#     # contrast info
#     npt.assert_allclose(head.FA, 10.0)
#     npt.assert_allclose(head.TE, 5.1)
#     npt.assert_allclose(head.TR, 250.0)

#     # geometry
#     npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 18.0)
#     npt.assert_allclose(head.ref_dicom.SliceThickness, 3.0)
#     npt.assert_allclose(head.ref_dicom.PixelSpacing, [1.0, 1.0])
#     npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, 0.0, 0.0, 0.0, 0.0, -1.0])

#     # patient info
#     assert head.ref_dicom.PatientName == "Phantom^FBIRN+Tangerine"
#     assert head.ref_dicom.PatientBirthDate == "19800909"
#     assert head.ref_dicom.PatientAge == "038Y"

# def test_gehc_archive():
#     path = deepmr.testdata("gehc::pfile")
#     data, head = deepmr.io.read_rawdata(path)

#     # image shape
#     npt.assert_allclose(data.shape, [7, 8, 1, 128, 128])

#     # contrast info
#     npt.assert_allclose(head.FA, 10.0)
#     npt.assert_allclose(head.TE, 5.1)
#     npt.assert_allclose(head.TR, 250.0)

#     # geometry
#     npt.assert_allclose(head.ref_dicom.SpacingBetweenSlices, 18.0)
#     npt.assert_allclose(head.ref_dicom.SliceThickness, 3.0)
#     npt.assert_allclose(head.ref_dicom.PixelSpacing, [1.0, 1.0])
#     npt.assert_allclose(head.ref_dicom.ImageOrientationPatient, [1.0, 0.0, 0.0, 0.0, 0.0, -1.0])

#     # patient info
#     assert head.ref_dicom.PatientName == "Phantom^FBIRN+Tangerine"
#     assert head.ref_dicom.PatientBirthDate == "19800909"
#     assert head.ref_dicom.PatientAge == "038Y"
