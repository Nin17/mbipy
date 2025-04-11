"""Normal integration methods for phase integration from gradients."""

__all__ = (
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

from .src.config import __have_scipy__
from .src.normal_integration import arnison, dct_poisson, dst_poisson, frankot, kottler

if __have_scipy__:  # ??? do i need this? probably not
    from .src.normal_integration import Li, Southwell, li, southwell
