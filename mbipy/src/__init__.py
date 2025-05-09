"""Modulation based x-ray phase contrast imaging."""

from __future__ import annotations

def _init_numba() -> None:
    """Import numba overloads."""
    from mbipy.src.config import config

    if not config.have_rocketfft:
        import warnings

        warnings.warn(
            "Rocket-FFT is not installed. Some functions may not be available.",
            ImportWarning,
            stacklevel=2,
        )

    from mbipy.src.normal_integration.fourier.overloads import (
        _dct2_2d_overload,
        _dst1_2d_overload,
        _fft_2d_overload,
        _flip_overload,
        _idct2_2d_overload,
        _idst1_2d_overload,
        _ifft_2d_overload,
        _irfft_2d_overload,
        _rfft_2d_overload,
    )
    from mbipy.src.utils_overloads import (
        _overload_array_namespace,
        _overload_cast_scalar,
        _overload_get_dtypes,
        _overload_idiv,
        _overload_imul,
        _overload_isub,
        _overload_setitem,
        _overload_astype,
    )
    from mbipy.src.normal_integration.overloads import (check_shapes)
