"""Module containing phantom generation routines."""

__all__ = ["create_shepp_logan", "create_brainweb", "ArbitraryPhantomBuilder"]

from dataclasses import asdict as _asdict
from dataclasses import dataclass as _dataclass
from dataclasses import fields as _fields
from typing import Union as _Union

import numpy as _np
import numpy.typing as _npt

from .brainweb import create_brainweb
from .ct_shepp_logan import ct_shepp_logan
from .mr_shepp_logan import mr_shepp_logan

def create_shepp_logan(
    npix: int,
    nslices: int = 1,
    mr: bool = False,
    B0: float = 3.0,
    model: str = "single",
):
    """
    Initialize numerical phantom for MR simulations.

    This function generates a numerical phantom for MR or CT simulations based on the provided parameters.

    Args:
        npix (int): Number of voxels in the phantom's in-plane dimension.
        nslices (int, optional): Number of slices in the phantom (default is 1).
        mr (bool, optional): Flag indicating whether the phantom is for qMRI (True) or CT (False) simulations (default is False).
        B0 (float, optional): Static field strength in units of [T]; ignored if `mr` is False (default is 3.0 T).
        model (str, optional): MR signal model (default is "single"). Options are:
            - "single": Single pool T1 / T2.
            - "bm": 2-pool model with exchange (iew pool + mw pool).
            - "mt": 2-pool model with exchange (iew/mw pool + Zeeman semisolid pool).
            - "bm-mt": 3-pool model (iew pool, mw pool, Zeeman semisolid pool; mw exchange with iew and ss).

    Returns:
        array or tuple: if mr = False, this is the numerical phantom. For mr = True, this is a tuple with the following:
            - segmentation (array): phantom segmentation (e.g., 1 := GM, 2 := WM, 3 := CSF...)
            - mrtp (list): list of dictionaries containing 1) free water T1/T2/T2*/ADC/v, 2) bm/mt T1/T2/fraction, 3) exchange matrix
                     for each class (index along the list correspond to value in segmentation mask)
            - emtp (list): list of dictionaries containing electromagnetic tissue properties for each class.

    Example:
        >>> seg, mrtp, emtp = create_shepp_logan(npix=256, nslices=3, mr=True, B0=1.5, model="mt")
        >>> phantom = create_shepp_logan(npix=512, mr=False)
    """
    if mr:
        return mr_shepp_logan(npix, nslices, B0, model)
    else:
        return ct_shepp_logan(npix, nslices)


@_dataclass
class ArbitraryPhantomBuilder:
    """Helper class to build qMRI phantoms from externally provided maps."""

    # relaxation properties
    T1: _Union[float, _npt.NDArray]  # ms
    T2: _Union[float, _npt.NDArray]  # ms
    segmentation: _npt.NDArray = None
    M0: float = 1.0

    # other properties
    T2star: _Union[float, _npt.NDArray] = 0.0  # ms
    chemshift: _Union[float, _npt.NDArray] = 0.0  # Hz / T

    # motion properties
    D: _Union[float, _npt.NDArray] = 0.0  # um**2 / ms
    v: _Union[float, _npt.NDArray] = 0.0  # cm / s

    # multi-component related properties
    exchange_rate: _Union[float, _npt.NDArray] = 0.0  # 1 / s

    # smaller pools
    bm: dict = None
    mt: dict = None

    # electromagnetic properties
    chi: float = 0.0
    sigma: float = 0.0  # S / m
    epsilon: float = 0.0

    # size and shape
    n_atoms: int = 1
    shape: tuple = None

    def __post_init__(self):
        # convert scalar to array and gather sizes and shapes
        sizes = []
        shapes = []
        for field in _fields(self):
            value = getattr(self, field.name)
            if (
                field.name != "bm"
                and field.name != "mt"
                and field.name != "segmentation"
                and field.name != "n_atoms"
                and field.name != "shape"
                and field.name != "exchange_rate"
            ):
                val = _np.asarray(value)
                sizes.append(val.size)
                shapes.append(val.shape)
                setattr(self, field.name, val)

        # get number of atoms
        self.n_atoms = _np.max(sizes)
        self.shape = shapes[_np.argmax(sizes)]

        # check shapes
        shapes = [shape for shape in shapes if shape != ()]
        assert (
            len(set(shapes)) <= 1
        ), "Error! All input valus must be either scalars or arrays of the same shape!"

        # check segmentation consistence
        if self.segmentation is not None:
            seg = self.segmentation
            if issubclass(seg.dtype.type, _np.integer):  # discrete segmentation case
                assert seg.max() == self.n_atoms - 1, (
                    f"Error! Number of atoms = {self.n_atoms} must match number of"
                    f" classes = {seg.max()}"
                )
            else:
                assert seg.shape[0] == self.n_atoms - 1, (
                    f"Error! Number of atoms = {self.n_atoms} must match number of"
                    f" classes = {seg.shape[0]}"
                )

        # expand scalars
        for field in _fields(self):
            value = getattr(self, field.name)
            if (
                field.name != "bm"
                and field.name != "mt"
                and field.name != "segmentation"
                and field.name != "n_atoms"
                and field.name != "shape"
                and field.name != "exchange_rate"
            ):
                if value.size == 1:
                    value = value * _np.ones(self.shape, dtype=_np.float32)
                value = _np.atleast_1d(value)
                setattr(self, field.name, value)

        # initialize exchange_rate
        self.exchange_rate = _np.zeros(list(self.shape) + [1], dtype=_np.float32)

        # initialize BM and MT pools
        self.bm = {}
        self.mt = {}

    def add_cest_pool(
        self,
        T1: _Union[float, _npt.NDArray],
        T2: _Union[float, _npt.NDArray],
        weight: _Union[float, _npt.NDArray],
        chemshift: _Union[float, _npt.NDArray] = 0.0,
    ):
        """
        Add a new Chemical Exchanging pool to the model.

        Args:
            T1 (Union[float, npt.NDArray]): New pool T1.
            T2 (Union[float, npt.NDArray]): New pool T2.
            weight (Union[float, npt.NDArray]): New pool relative fraction.
            chemshift (Union[float, npt.NDArray], optional): New pool chemical shift. Defaults to 0.0.

        """
        # check pool
        if _np.isscalar(T1):
            T1 *= _np.ones((self.n_atoms, 1), dtype=_np.float32)
        elif len(T1.shape) == 1:
            assert _np.array_equal(
                T1.shape, self.shape
            ), "Input T1 must be either a scalar or match the existing shape."
            T1 = T1[..., None]
        else:
            assert _np.array_equal(
                T1.squeeze().shape, self.shape
            ), "Input T1 must be either a scalar or match the existing shape."
            assert T1.shape[-1] == 1, "Pool dimension size must be 1!"
        if _np.isscalar(T2):
            T2 *= _np.ones((self.n_atoms, 1), dtype=_np.float32)
        elif len(T2.shape) == 1:
            assert _np.array_equal(
                T2.shape, self.shape
            ), "Input T2 must be either a scalar or match the existing shape."
            T2 = T2[..., None]
        else:
            assert _np.array_equal(
                T2.squeeze().shape, self.shape
            ), "Input T2 must be either a scalar or match the existing shape."
            assert T2.shape[-1] == 1, "Pool dimension size must be 1!"
        if _np.isscalar(weight):
            weight *= _np.ones((self.n_atoms, 1), dtype=_np.float32)
        elif len(weight.shape) == 1:
            assert _np.array_equal(
                weight.shape, self.shape
            ), "Input weight must be either a scalar or match the existing shape."
            weight = weight[..., None]
        else:
            assert _np.array_equal(
                weight.squeeze().shape, self.shape
            ), "Input weight must be either a scalar or match the existing shape."
            assert weight.shape[-1] == 1, "Pool dimension size must be 1!"
        if _np.isscalar(chemshift):
            chemshift *= _np.ones((self.n_atoms, 1), dtype=_np.float32)
        elif len(chemshift.shape) == 1:
            assert _np.array_equal(chemshift.shape, self.shape), (
                "Input chemical_shift must be either a scalar or match the existing"
                " shape."
            )
            chemshift = chemshift[..., None]
        else:
            assert _np.array_equal(chemshift.squeeze().shape, self.shape), (
                "Input chemical_shift must be either a scalar or match the existing"
                " shape."
            )
            assert chemshift.shape[-1] == 1, "Pool dimension size must be 1!"

        # BM pool already existing; add a new component
        if self.bm:
            self.bm["T1"] = _np.concatenate((self.bm["T1"], T1), axis=-1)
            self.bm["T2"] = _np.concatenate((self.bm["T2"], T2), axis=-1)
            self.bm["weight"] = _np.concatenate((self.bm["weight"], weight), axis=-1)
            self.bm["chemshift"] = _np.concatenate(
                (self.bm["chemshift"], chemshift), axis=-1
            )
        else:
            self.bm["T1"] = T1
            self.bm["T2"] = T2
            self.bm["weight"] = weight
            self.bm["chemshift"] = chemshift

    def add_mt_pool(self, weight: _Union[float, _npt.NDArray]):
        """
        Set macromolecolar pool.

        Args:
            weight (Union[float, npt.NDArray]): Semisolid pool relative fraction.
        """
        # check pool
        if _np.isscalar(weight):
            weight *= _np.ones((self.n_atoms, 1), dtype=_np.float32)
        elif len(weight.shape) == 1:
            assert _np.array_equal(
                weight.shape, self.shape
            ), "Input weight must be either a scalar or match the existing shape."
            weight = weight[..., None]
        else:
            assert _np.array_equal(
                weight.squeeze().shape, self.shape
            ), "Input weight must be either a scalar or match the existing shape."
            assert weight.shape[-1] == 1, "Pool dimension size must be 1!"

        self.mt["weight"] = weight

    def set_exchange_rate(self, *exchange_rate_matrix_rows: _Union[list, tuple]):
        """
        Build system exchange matrix.

        Args:
            *exchange_rate_matrix_rows (list or tuple): list or tuple of exchange constant.
                Each argument represent a row of the exchange matrix in s**-1.
                Each element of each argument represent a single element of the row; these can
                be either scalar or array-like objects of shape (n_atoms,)

        """
        # check that every row has enough exchange rates for each pool
        npools = 1
        if self.bm:
            npools += self.bm["T1"].shape[-1]
        if self.mt:
            npools += self.mt["T1"].shape[-1]

        # count rows
        assert (
            len(exchange_rate_matrix_rows) == npools
        ), "Error! Incorrect number of exchange constant"
        for row in exchange_rate_matrix_rows:
            row = _np.asarray(row).T
            assert (
                row.shape[0] == npools
            ), "Error! Incorrect number of exchange constant per row"
            for el in row:
                if _np.isscalar(el):
                    el *= _np.ones(self.n_atoms, dtype=_np.float32)
                else:
                    assert _np.array_equal(el.shape, self.shape), (
                        "Input exchange constant must be either a scalar or match the"
                        " existing shape."
                    )
            # stack element in row
            row = _np.stack(row, axis=-1)

        # stack row
        self.exchange_rate = _np.stack(exchange_rate_matrix_rows, axis=-1)

        # check it is symmetric
        assert _np.allclose(
            self.exchange_rate, self.exchange_rate.swapaxes(-1, -2)
        ), "Error! Non-directional exchange matrix must be symmetric."

    def build(self):
        """
        Return structures for MR simulation.
        """
        # check that exchange matrix is big enough
        npools = 1
        if self.bm:
            npools += self.bm["T1"].shape[-1]
        if self.mt:
            npools += self.mt["T1"].shape[-1]

        # actual check
        assert (
            self.exchange_rate.shape[-1] == npools
        ), "Error! Incorrect exchange matrix size."
        if npools > 1:
            assert (
                self.exchange_rate.shape[-2] == npools
            ), "Error! Incorrect exchange matrix size."

        # prepare output
        mrtp = _asdict(self)

        # erase unused stuff
        mrtp.pop("n_atoms")
        mrtp.pop("shape")

        # get segmentation
        seg = mrtp.pop("segmentation")

        # electromagnetic tissue properties
        emtp = {}
        emtp["chi"] = mrtp.pop("chi")
        emtp["sigma"] = mrtp.pop("sigma")
        emtp["epsilon"] = mrtp.pop("epsilon")

        return seg, mrtp, emtp
