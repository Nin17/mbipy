"""Utilities for normal integration methods that solve sparse least squares problems."""

from __future__ import annotations

__all__ = ("BaseSparseNormalIntegration", "csr_matrix", "factorized")

import importlib
from typing import TYPE_CHECKING

from array_api_compat import (
    array_namespace,
    is_cupy_namespace,
    is_jax_namespace,
)
from numpy import broadcast_shapes

if TYPE_CHECKING:  # pragma: no cover

    from types import ModuleType
    from typing import Callable

    from numpy import floating, integer
    from numpy.typing import DTypeLike, NDArray

    from mbipy.src.config import config as cfg

    if cfg.have_scipy:
        from scipy.sparse import spmatrix


def csr_matrix(
    data: NDArray[floating],
    rows: NDArray[integer],
    cols: NDArray[integer],
    shape: tuple[int, int],
) -> spmatrix:
    """Compressed Sparse Row Matrix.

    data, rows and cols satisfy the relationship a[rows[k], cols[k]] = data[k]

    Parameters
    ----------
    data : NDArray[floating]
        data values
    rows : NDArray[integer]
        row indices
    cols : NDArray[integer]
        column indices
    shape : tuple[int, int]
        shape of the sparse matrix.

    Returns
    -------
    spmatrix
        sparse matrix in Compressed Sparse Row format.
    """
    xp = array_namespace(data, rows, cols)
    if is_cupy_namespace(xp):
        sparse = importlib.import_module("cupyx.scipy.sparse")
    else:
        sparse = importlib.import_module("scipy.sparse")
    return sparse.csr_matrix((data, (rows, cols)), shape=shape)


def factorized(a: spmatrix) -> Callable[[NDArray[floating]], NDArray[floating]]:
    """Return a function for solving a sparse linear system, with A pre-factorized.

    Parameters
    ----------
    a : spmatrix
        sparse matrix in Compressed Sparse Column format.

    Returns
    -------
    Callable[[NDArray[floating]], NDArray[floating]]
        function that takes a vector and returns the solution to the linear system.
    """
    xp = array_namespace(a.data, a.indices, a.indptr)
    if is_cupy_namespace(xp):
        splinalg = importlib.import_module("cupyx.scipy.sparse.linalg")
    else:
        splinalg = importlib.import_module("scipy.sparse.linalg")
    return splinalg.factorized(a)


class BaseSparseNormalIntegration:
    """Base class for sparse least squares normal integration."""

    def __init__(
        self,
        shape: tuple[int, int],
        xp: ModuleType | None = None,
        idtype: DTypeLike | None = None,
        fdtype: DTypeLike | None = None,
    ) -> None:
        """Based class for sparse least squares normal integration."""
        self.shape = shape
        xp = xp or importlib.import_module("numpy")
        self.xp = xp
        idtype = idtype or xp.int64
        fdtype = fdtype or xp.float64
        self.idtype = idtype
        self.fdtype = fdtype

        f, mt = self._factorized_mt_func(shape, xp, idtype, fdtype)
        self._f = f
        self._mt = mt

    def __repr__(self) -> str:
        """Representation of sparse normal integration classes."""
        return (
            f"{self.__class__.__name__}({self.shape}, xp={self.xp.__name__}, "
            f"idtype={self.idtype.__name__}, fdtype={self.fdtype.__name__})"
        )

    def __call__(
        self,
        gy: NDArray[floating],
        gx: NDArray[floating],
    ) -> NDArray[floating]:
        """Perform normal integration.

        Parameters
        ----------
        gy : (M, N) NDArray[floating]
            Vertical gradient.
        gx : (M, N) NDArray[floating]
            Horizontal gradient.

        Returns
        -------
        (M, N) NDArray[floating]
            Normal field.

        """
        shape = broadcast_shapes(gy.shape, gx.shape)
        if shape != self.shape:
            msg = "Input arrays must have the same shape as the object."
            raise ValueError(msg)
        vector = self._vec_func(gy, gx)

        return self.xp.reshape(self._f(self._mt @ vector), shape)

    @staticmethod
    def _factorized_mt_func(
        shape: tuple[int, int],
        xp: ModuleType,
        idtype: DTypeLike,
        fdtype: DTypeLike,
    ) -> tuple[Callable[[NDArray], NDArray], spmatrix]:
        """Factorization and transpose of matrix: to be implemented by subclasses."""
        msg = "Subclasses must implement _factorize_func."
        raise NotImplementedError(msg)

    @staticmethod
    def _vec_func(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
        """Vector function to be implemented by subclasses."""
        msg = "Subclasses must implement _vec_func."
        raise NotImplementedError(msg)
