"""_summary_
"""

__all__ = "wst", "wst_wsvt", "wsvt"

import numpy as np
import pywt

from ...src.phase_retrieval.explicit import (
    create_find_displacement,
    create_similarity_st,
    create_similarity_svt,
    create_vectors_st,
    create_vectors_st_svt,
    create_wst,
    create_wst_wsvt,
    create_wsvt,
)

find_displacement = create_find_displacement(np)

similarity_st = create_similarity_st(np, np.lib.stride_tricks.sliding_window_view)
similarity_svt = create_similarity_svt(np, np.lib.stride_tricks.sliding_window_view)

vectors_st = create_vectors_st(np.lib.stride_tricks.sliding_window_view)
vectors_st_svt = create_vectors_st_svt(np, np.lib.stride_tricks.sliding_window_view)

wst = create_wst(np, vectors_st, similarity_st, find_displacement, pywt.wavedec)
wst_wsvt = create_wst_wsvt(
    np, vectors_st_svt, similarity_st, find_displacement, pywt.wavedec
)
wsvt = create_wsvt(np, similarity_svt, find_displacement, pywt.wavedec)
