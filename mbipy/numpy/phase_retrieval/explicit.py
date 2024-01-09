"""_summary_
"""

__all__ = "cst", "cst_csvt", "csvt", "xst", "xst_xsvt", "xsvt"


import numpy as np
from scipy import fft

from ...src.phase_retrieval.explicit import (
    create_cst,
    create_cst_csvt,
    create_csvt,
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


cst = create_cst(np, vectors_st, similarity_st, find_displacement, fft.dct)
cst_csvt = create_cst_csvt(
    np, vectors_st_svt, similarity_st, find_displacement, fft.dct
)
csvt = create_csvt(np, similarity_svt, find_displacement, fft.dct)



xst = create_xst(np, vectors_st, similarity_st, find_displacement)
xst_xsvt = create_xst_xsvt(np, vectors_st_svt, similarity_st, find_displacement)
xsvt = create_xsvt(np, similarity_svt, find_displacement)
