"""Convenience functions for reading images."""

from __future__ import annotations

__all__ = ["load_data", "load_paths", "load_stack"]

from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING

import fabio
from numpy import stack

if TYPE_CHECKING:

    from os import PathLike

    from numpy.typing import DTypeLike, NDArray


def load_data(path: str | PathLike, *, frame: int | None = None) -> NDArray:
    """Load an image from a file.

    Uses [fabio.open][].

    !!! example

        ```python
        from mbipy.load import load_data

        load_data("img00.tiff")
        ```

    Parameters
    ----------
    path : str | PathLike
        Path to the image.
    frame : int | None, optional
        Passed to [fabio.open][], by default `None`

    Returns
    -------
    NDArray
        Image data.

    """  # FIXME(nin17): [fabio.open][]
    with fabio.open(path, frame) as f:
        return f.data


def load_paths(
    *paths: str | PathLike,
    frame: int | None = None,
    axis: int | None = None,
    out: NDArray | None = None,
    dtype: None = None,
) -> list[NDArray] | NDArray:
    """Load images from paths.

    Uses [fabio.open][].

    !!! example

        ```python
        from mbipy.load import load_paths

        load_paths("img00.tiff", "img02.tiff", "img01.tiff")
        ```

    Parameters
    ----------
    *paths : str | PathLike
        Paths to the images.
    frame : int | None, optional
        Passed to [fabio.open][], by default `None`
    axis : int | None, optional
        If `None`, returns a [list][] of [NDArray][numpy.typing.NDArray]s, otherwise
        images passed to [numpy.stack][], by default `None`
    out : NDArray | None, optional
        Passed to [numpy.stack][] if `axis` is not `None`, by default `None`
    dtype : DTypeLike | None, optional
        Passed to [numpy.stack][] if `axis` is not `None`, by default `None`

    Returns
    -------
    list[NDArray] | NDArray
        If `axis` is `None`, returns a [list][] of [NDArray][numpy.typing.NDArray]s,
        otherwise returns a single [NDArray][numpy.typing.NDArray].
    """  # FIXME(nin17): [fabio.open][]
    images = [load_data(i, frame=frame) for i in paths]
    if axis is None:
        return images
    return stack(images, axis=axis, out=out, dtype=dtype)


def load_stack(
    path: str | PathLike,
    *patterns: str,
    frame: int | None = None,
    axis: int | None = None,
    out: NDArray | None = None,
    dtype: DTypeLike | None = None,
    **kwargs: bool,
) -> list[NDArray] | NDArray:
    # TODO(nin17): check example - do with path to data
    r"""Load all images matching the patterns, sorted by path, from a directory.

    Uses [fabio.open][].

    !!! example

        ```python
        from mbipy.load import load_stack

        load_stack(".", "*[0-9][0-9].tiff")
        ```

    Parameters
    ----------
    path : str | PathLike
        Path to the directory containing the images.
    *patterns : str
        Patterns to match the files.
    frame : int | None, optional
        Passed to [fabio.open][], by default `None`
    axis : int | None, optional
        If `None`, returns a [list][] of [NDArray][numpy.typing.NDArray]s, otherwise
        images passed to [numpy.stack][], by default `None`
    out : NDArray | None, optional
        Passed to [numpy.stack][] if `axis` is not `None`, by default `None`
    dtype : DTypeLike | None, optional
        Passed to [numpy.stack][] if `axis` is not `None`, by default `None`
    **kwargs : bool
        Additional keyword arguments passed to [Path.glob][pathlib.Path.glob].

        * `case_sensitive`: `bool | None`
        [(python $`\ge`$ 3.12)](https://docs.python.org/3.12/whatsnew/3.12.html#pathlib)
        * `recurse_symlinks`: `bool`
        [(python $`\ge`$ 3.13)](https://docs.python.org/3.13/whatsnew/3.13.html#pathlib)

    Returns
    -------
    list[NDArray] | NDArray
        If `axis` is `None`, returns a [list][] of [NDArray][numpy.typing.NDArray]s,
        otherwise returns a single [NDArray][numpy.typing.NDArray].

    """  # FIXME(nin17): [fabio.open][]
    _path = Path(path)
    file_paths = sorted(
        chain(*[_path.glob(p, **kwargs) for p in patterns]),
    )
    return load_paths(*file_paths, frame=frame, axis=axis, out=out, dtype=dtype)
