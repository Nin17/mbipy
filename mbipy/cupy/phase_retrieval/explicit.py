"""_summary_
"""

__all__ = "cst", "cst_csvt", "csvt", "xst", "xst_xsvt", "xsvt"

import cupy as cp
import numpy as np
from cupyx.scipy import fft

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
from ..utils.swv import sliding_window_view

find_displacement = create_find_displacement(cp)

similarity_st = create_similarity_st(cp, sliding_window_view)
similarity_svt = create_similarity_svt(cp, sliding_window_view)

vectors_st = create_vectors_st(sliding_window_view)
vectors_st_svt = create_vectors_st_svt(cp, sliding_window_view)

cst = create_cst(cp, vectors_st, similarity_st, find_displacement, fft.dct)
cst_csvt = create_cst_csvt(
    cp, vectors_st_svt, similarity_st, find_displacement, fft.dct
)
csvt = create_csvt(cp, similarity_svt, find_displacement, fft.dct)

xst = create_xst(cp, vectors_st, similarity_st, find_displacement)
xst_xsvt = create_xst_xsvt(cp, vectors_st_svt, similarity_st, find_displacement)
xsvt = create_xsvt(cp, similarity_svt, find_displacement)
