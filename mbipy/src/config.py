"""_summary_"""

__all__ = (
    "_have_cupy",
    "_have_jax",
    "_have_jaxwt",
    "_have_numba",
    "_have_ptwt",
    "_have_pyvkfft",
    "_have_pywt",
    "_have_rocketfft",
    "_have_scipy",
    "_have_torch",
)

import importlib
import os


class Config:
    """Configuration class for mbipy."""

    def __init__(self) -> None:
        # ----------------------------- cupy dependencies ---------------------------- #
        self._have_cupy = bool(importlib.util.find_spec("cupy"))
        self._have_pyvkfft = bool(importlib.util.find_spec("pyvkfft"))
        self._use_pyvkfft = os.getenv("MBIPY_USE_PYVKFFT", "False").lower() == "true"
        # ----------------------------- jax dependencies ----------------------------- #
        self._have_jax = bool(importlib.util.find_spec("jax"))
        self._have_jaxwt = bool(importlib.util.find_spec("jaxwt"))
        # ---------------------------- numba dependencies ---------------------------- #
        self._have_numba = bool(importlib.util.find_spec("numba"))
        self._have_rocketfft = bool(importlib.util.find_spec("rocket_fft"))
        # ---------------------------- numpy dependencies ---------------------------- #
        self._have_pywt = bool(importlib.util.find_spec("pywt"))
        self._have_scipy = bool(importlib.util.find_spec("scipy"))
        # --------------------------- pytorch dependencies --------------------------- #
        self._have_ptwt = bool(importlib.util.find_spec("ptwt"))
        self._have_torch = bool(importlib.util.find_spec("torch"))
        self._have_torch_dct = bool(importlib.util.find_spec("torch_dct"))
        # ---------------------------------------------------------------------------- #

    @property
    def have_cupy(self) -> bool:
        """Check if cupy is installed."""
        return self._have_cupy

    @property
    def have_pyvkfft(self) -> bool:
        """Check if pyvkfft is installed."""
        return self._have_pyvkfft

    @property
    def use_pyvkfft(self) -> bool:
        return self._use_pyvkfft

    @use_pyvkfft.setter
    def use_pyvkfft(self, value: bool) -> None:
        """Set the use_pyvkfft flag."""
        if not self._have_cupy:
            msg = "CuPy is not installed."
            raise ImportError(msg)
        if not self._have_pyvkfft:
            msg = "pyvkfft is not installed."
            raise ImportError(msg)
        self._use_pyvkfft = bool(value)

    @property
    def have_jax(self) -> bool:
        """Check if jax is installed."""
        return self._have_jax

    @property
    def have_jaxwt(self) -> bool:
        """Check if jaxwt is installed."""
        return self._have_jaxwt

    @property
    def have_numba(self) -> bool:
        """Check if numba is installed."""
        return self._have_numba

    @property
    def have_rocketfft(self) -> bool:
        """Check if rocketfft is installed."""
        return self._have_rocketfft

    @property
    def have_pywt(self) -> bool:
        """Check if pywt is installed."""
        return self._have_pywt

    @property
    def have_scipy(self) -> bool:
        """Check if scipy is installed."""
        return self._have_scipy

    @property
    def have_torch(self) -> bool:
        """Check if torch is installed."""
        return self._have_torch

    @property
    def have_torch_dct(self) -> bool:
        """Check if torch_dct is installed."""
        return self._have_torch_dct


_have_cupy = bool(importlib.util.find_spec("cupy"))
_have_pyvkfft = bool(importlib.util.find_spec("pyvkfft"))
# TODO(nin17): use_pyvkfft flag to use pyvkfft instead of cufft for all fft calls with
# cupy rather than just the dst

_have_jax = bool(importlib.util.find_spec("jax"))
_have_jaxwt = bool(importlib.util.find_spec("jaxwt"))


_have_numba = bool(importlib.util.find_spec("numba"))
_have_rocketfft = bool(importlib.util.find_spec("rocket_fft"))


_have_pywt = bool(importlib.util.find_spec("pywavelets"))
_have_scipy = bool(importlib.util.find_spec("scipy"))


_have_torch = bool(importlib.util.find_spec("torch"))
_have_ptwt = bool(importlib.util.find_spec("ptwt"))

config = Config()
