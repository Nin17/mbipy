"""
Array based implementations of: XST, XSVT & XST-XSVT.
Compatible with numpy, cupy & jax.
"""

from __future__ import annotations

__all__ = "create_xst", "create_xsvt", "create_xst_xsvt"

import types
import typing

from numpy.typing import NDArray

PAD_MODE = "reflect"


def create_xst(
    xp: types.ModuleType,
    vectors_st: typing.Callable,
    similarity_st: typing.Callable,
    find_displacement: typing.Callable,
) -> typing.Callable:
    def xst(
        img1: NDArray,
        img2: NDArray,
        ss: tuple[int, int],
        ts: tuple[int, int],
        pcc: bool = False,
        transmission: bool = False,
        darkfield: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """_summary_

        Parameters
        ----------
        img1 : NDArray (..., M, N)
            The image to be searched.
        img2 : NDArray (..., M, N)
            _description_
        ss : tuple[int, int]
            Search window size - must be odd.
        ts : tuple[int, int]
            Template window size - must be odd.
        pcc : bool, optional
            Whether to calculate the Pearson Correlation Coefficient instead of the
            Cosine Similarity, by default False
        transmission : bool, optional
            _description_, by default False
        darkfield : bool, optional
            _description_, by default False

        Returns
        -------
        tuple[NDArray, NDArray]
            Vertical and horizontal displacements in pixels.
            Both arrays have shape (..., M - ss[0] - ts[0] + 2, N - ss[1] - ts[1] + 2)

        Raises
        ------
        NotImplementedError
            _description_
        """
        vectors = vectors_st(img1, img2, ss, ts)
        similarity = similarity_st(*vectors, ss, pcc)
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        displacement = find_displacement(similarity_padded)
        if transmission or darkfield:
            # TODO nin17: implement attenuation & darkfield
            raise NotImplementedError
        return displacement

    return xst


def create_xst_xsvt(xp, vectors_st_svt, similarity_st, find_displacement):
    def xst_xsvt(img1, img2, ss, ts, pcc=False):
        # TODO nin17: add docstring
        vectors = vectors_st_svt(img1, img2, ss, ts)
        similarity = similarity_st(*vectors, ss, pcc)
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        return find_displacement(similarity_padded)

    return xst_xsvt


def create_xsvt(
    xp: types.ModuleType,
    similarity_svt: typing.Callable,
    find_displacement: typing.Callable,
) -> typing.Callable:
    def xsvt(img1, img2, m: int, n: int, pcc: bool = False):
        # TODO nin17: add docstring
        similarity = similarity_svt(img1, img2[..., m:-m, n:-n], m, n, pcc)
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        return find_displacement(similarity_padded)

    return xsvt
