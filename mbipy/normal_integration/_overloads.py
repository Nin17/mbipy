"""Numba overloads for normal integration functions."""

from __future__ import annotations

from numba import extending

from ._utils import check_shapes

extending.register_jitable(check_shapes)
