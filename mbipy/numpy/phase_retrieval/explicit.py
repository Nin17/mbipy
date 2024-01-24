"""_summary_
"""

__all__ = "xst", "xsvt", "xst_xsvt"


import warnings

import numpy as np

try:
    from scipy import fft

    SCIPY_FFT = True
except ImportError:
    SCIPY_FFT = False
    warnings.warn("Cosine and Sine transforms unavailable. Need to install scipy.")
    from numpy import fft

try:
    import pywt

    PYWT = True
except ImportError:
    PYWT = False
    warnings.warn("Wavelet transforms unavailable. Need to install pywavelets.")


from ...src.phase_retrieval.explicit import (  # create_cst,; create_cst_csvt,; create_csvt,
    create_find_displacement,
    create_similarity_st,
    create_similarity_svt,
    create_vectors_st,
    create_vectors_st_svt,
    create_xst,
    create_xst_xsvt,
    create_xsvt,
)

find_displacement = create_find_displacement(np)

similarity_st = create_similarity_st(np, np.lib.stride_tricks.sliding_window_view)
similarity_svt = create_similarity_svt(np, np.lib.stride_tricks.sliding_window_view)

vectors_st = create_vectors_st(np.lib.stride_tricks.sliding_window_view)
vectors_st_svt = create_vectors_st_svt(np, np.lib.stride_tricks.sliding_window_view)


def _hartley(*args, **kwargs):
    """
    Hartley transform for real inputs
    """
    # TODO nin17: dot product of hartley transform with rfft
    result = fft.fft(*args, **kwargs)
    return result.real - result.imag


transforms = {
    "fourier": fft.fft,  # TODO nin17: fft.rfft would be faster dot(:1, :1) + 2*dot(1:, 1:)
    "hartley": _hartley,
}
if SCIPY_FFT:
    transforms |= {"cosine": fft.dct, "sine": fft.dst}
if PYWT:
    transforms |= {"wavelet": pywt.wavedec}


xst = create_xst(np, vectors_st, similarity_st, find_displacement, transforms)
xst_xsvt = create_xst_xsvt(
    np, vectors_st_svt, similarity_st, find_displacement, transforms
)
xsvt = create_xsvt(np, similarity_svt, find_displacement, transforms)
