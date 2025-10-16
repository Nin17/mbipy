"""Normal integration."""

from __future__ import annotations

__all__ = (
    "FFTMethod",
    "Li",
    "Southwell",
    "arnison",
    "dct_poisson",
    "dst_poisson",
    "frankot",
    "kottler",
    "li",
    "southwell",
)

from .fourier import arnison, dct_poisson, dst_poisson, frankot, kottler
from .fourier.utils import FFTMethod
from .least_squares import Li, Southwell, li, southwell
