"""Utilities for normal integration methods that solve sparse least squares problems."""

from __future__ import annotations

__all__ = ["BaseSparseNormalIntegration", "csr_matrix", "factorized"]

import importlib
from typing import TYPE_CHECKING

from array_api_compat import array_namespace, is_cupy_namespace
from numpy import broadcast_shapes

if TYPE_CHECKING:

    from collections.abc import Callable
    from types import ModuleType

    from numpy import dtype, floating, integer
    from numpy.typing import NDArray

    from mbipy.src.config import config as cfg

    if cfg._have_scipy:
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
    def __init__(
        self,
        shape: tuple[int, int],
        xp: ModuleType | None = None,
        idtype: dtype[integer] | None = None,
        fdtype: dtype[floating] | None = None,
    ) -> None:
        """Sparse least squares normal integration.

        !!! warning "Not part of the public API - use at your own risk!"

        Parameters
        ----------
        shape : tuple[int, int]
            Shape of the gradient fields: (M, N).
        xp : ModuleType | None, optional
            Array library (only [numpy][] & [cupy][] supported), by default `None`
        idtype : dtype[integer] | None, optional
            integer dtype, by default `None`
        fdtype : dtype[floating] | None, optional
            floating dtype, by default `None`
        """
        self.shape = shape
        self.xp = xp or importlib.import_module("numpy")
        self.idtype = idtype or self.xp.int64
        self.fdtype = fdtype or self.xp.float64

        self._f, self._mt = self._factorized_mt_func(shape, self.xp, idtype, fdtype)

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
        gy : NDArray[floating] (M, N)
            Vertical gradient.
        gx : NDArray[floating] (M, N)
            Horizontal gradient.

        Returns
        -------
        NDArray[floating] (M, N)
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
        idtype: dtype[integer],
        fdtype: dtype[floating],
    ) -> tuple[Callable[[NDArray[floating]], NDArray[floating]], spmatrix]:
        """Factorization and transpose of matrix: to be implemented by subclasses."""
        msg = "Subclasses must implement _factorize_func."
        raise NotImplementedError(msg)

    @staticmethod
    def _vec_func(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
        """Vector function to be implemented by subclasses."""
        msg = "Subclasses must implement _vec_func."
        raise NotImplementedError(msg)

    def integrate(
        self,
        gy: NDArray[floating],
        gx: NDArray[floating],
    ) -> NDArray[floating]:
        """Perform normal integration.

        Parameters
        ----------
        gy : NDArray[floating] (M, N)
            Vertical gradient.
        gx : NDArray[floating] (M, N)
            Horizontal gradient.

        Returns
        -------
        NDArray[floating] (M, N)
            Normal field.

        """
        return self.__call__(self, gy, gx)
