"""Utility functions for normal integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import broadcast_shapes

if TYPE_CHECKING:  # pragma: no cover

    from numpy.typing import NDArray


def check_shapes(gy: NDArray, gx: NDArray) -> tuple[int, int]:
    """Check the shapes of gy and gx broadcast, returning the last two dimensions.

    Returns
    -------
    tuple[int, int]
        Shape of the last two dimensions of the broadcasted arrays.

    """
    return broadcast_shapes(gy.shape, gx.shape)[-2:]
