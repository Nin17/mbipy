"""
Array based implementations of: CST, CSVT & CST-CSVT.
Compatible with numpy, cupy & jax.
"""
from __future__ import annotations

__all__ = "create_cst", "create_cst_csvt", "create_csvt"

import types
import typing

from .utils import cutoff_warning

# !!! nin17: Don't change DCT_NORM - only norm="ortho" gives correct results
DCT_NORM = "ortho"
PAD_MODE = "reflect"


def create_cst(
    xp: types.ModuleType,
    vectors_st: typing.Callable,
    similarity_st: typing.Callable,
    find_displacement: typing.Callable,
    dct: typing.Callable,
) -> typing.Callable:
    """_summary_

    Parameters
    ----------
    xp : types.ModuleType
        _description_
    vectors_st : typing.Callable
        _description_
    similarity_st : typing.Callable
        _description_
    find_displacement : typing.Callable
        _description_
    dct : typing.Callable
        _description_

    Returns
    -------
    typing.Callable
        _description_
    """
    # TODO nin17: add docstring

    def cst(
        img1,
        img2,
        ss: tuple[int, int],
        ts: tuple[int, int],
        cutoff: int | None = None,
        dct_kwargs: dict | None = None,
    ):
        """_summary_

        Parameters
        ----------
        img1 : array
            _description_
        img2 : array
            _description_
        ss : tuple[int, int]
            _description_
        ts : tuple[int, int]
            _description_
        cutoff : int | None, optional
            _description_, by default None
        pcc : bool, optional
            _description_, by default False

        Returns
        -------
        array
            _description_
        """
        pcc = False  # nin17: pcc not preserved in dct
        if dct_kwargs is None:
            dct_kwargs = {}
        vectors = vectors_st(img1, img2, ss, ts)
        vectors = [
            dct(i, norm=DCT_NORM, axis=-1, **dct_kwargs)[..., :cutoff] for i in vectors
        ]
        cutoff_warning(*vectors, cutoff=cutoff, axis=-1)
        similarity = similarity_st(*vectors, ss, pcc)
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE,
        )
        return find_displacement(similarity_padded)

    return cst


def create_csvt(xp, cosine_similarity_svt, find_displacement, dct):
    def csvt(img1, img2, m, n, cutoff=None, dct_kwargs=None):
        pcc = False  # nin17: pcc not preserved in dct
        if dct_kwargs is None:
            dct_kwargs = {}

        cutoff_warning(img1, img2, cutoff=cutoff, axis=-3)

        img1_dct = dct(img1, norm=DCT_NORM, axis=-3, **dct_kwargs)[..., :cutoff, :, :]
        img2_dct = dct(img2[..., m:-m, n:-n], norm=DCT_NORM, axis=-3, **dct_kwargs)[
            ..., :cutoff, :, :,
        ]

        cs = cosine_similarity_svt(img1_dct, img2_dct, m, n, pcc)

        cs_padded = xp.pad(cs, ((0, 0),) * (cs.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE)

        return find_displacement(cs_padded)

    return csvt


def create_cst_csvt(xp, vectors_st_svt, similarity_st, find_displacement, dct):
    def cst_csvt(img1, img2, ss, ts, cutoff=None, dct_kwargs=None):
        pcc = False  # nin17: pcc not preserved in dct
        if dct_kwargs is None:
            dct_kwargs = {}
        vectors = vectors_st_svt(img1, img2, ss, ts)
        vectors = [
            dct(i, norm=DCT_NORM, axis=-1, **dct_kwargs)[..., :cutoff] for i in vectors
        ]

        cutoff_warning(*vectors, cutoff=cutoff, axis=-1)

        similarity = similarity_st(*vectors, ss, pcc)
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE,
        )
        return find_displacement(similarity_padded)

    return cst_csvt
