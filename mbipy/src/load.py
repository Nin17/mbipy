"""Convenience functions for reading images."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import fabio
import numpy as np

if TYPE_CHECKING:

    from os import PathLike

    from numpy.typing import DTypeLike, NDArray

    StrPath = str | PathLike[str]

__all__ = ("load_stack",)


def load_data(path: StrPath, *, frame: None = None) -> NDArray:
    """Load an image from a file.

    Parameters
    ----------
    path : StrPath
        Path to the image.
    frame : None, optional
        Passed to fabio.open, by default None

    Returns
    -------
    NDArray
        Image data.

    """
    with fabio.open(path, frame) as f:
        return f.data


def load_paths(
    *paths: StrPath,
    frame: None = None,
    axis: int | None = None,
    out: NDArray | None = None,
    dtype: None = None,
) -> list[NDArray] | NDArray:
    """Load images from paths.

    Parameters
    ----------
    *paths : StrPath
        Paths to the images.
    frame : None, optional
        Passed to fabio.open, by default None
    axis : int | None, optional
        If None, returns a list of NDArrays, otherwise passed to np.stack, by default
        None
    out : NDArray | None, optional
        Passed to np.stack if axis is not None, by default None
    dtype : DTypeLike | None, optional
        Passed to np.stack if axis is not None, by default None

    Returns
    -------
    list[NDArray] | NDArray
        If axis is None, returns a list of NDArrays, otherwise returns a single NDArray.
    """
    images = [load_data(i, frame=frame) for i in paths]
    if axis is None:
        return images
    return np.stack(images, axis=axis, out=out, dtype=dtype)


def load_stack(
    path: StrPath,
    *patterns: str,
    frame: None = None,
    axis: int | None = None,
    out: NDArray | None = None,
    dtype: DTypeLike | None = None,
    **kwargs: bool,
) -> list[NDArray] | NDArray:
    """Load all images matching the patterns, sorted by path, from a directory.

    Parameters
    ----------
    path : StrPath
        Path to the directory containing the images.
    *patterns : str
        Patterns to match the files.
    frame : None, optional
        Passed to fabio.open, by default None
    axis : int | None, optional
        If None, returns a list of NDArrays, otherwise passed to np.stack, by default
        None
    out : NDArray | None, optional
        Passed to np.stack if axis is not None, by default None
    dtype : DTypeLike | None, optional
        Passed to np.stack if axis is not None, by default None
    **kwargs : bool
        Additional keyword arguments passed to Path.glob.
        case_sensitive: bool | None (python >= 3.12)
        recurse_symlinks: bool (python >= 3.13)

    Returns
    -------
    list[NDArray] | NDArray
        If axis is None, returns a list of NDArrays, otherwise returns a single NDArray.

    """
    _path = Path(path)
    file_paths = sorted(
        chain(*[_path.glob(p, **kwargs) for p in patterns]),
    )
    return load_paths(*file_paths, frame=frame, axis=axis, out=out, dtype=dtype)
