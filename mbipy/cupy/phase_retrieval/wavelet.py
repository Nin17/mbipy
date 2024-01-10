"""_summary_
"""

__all__ = "wst", "wst_wsvt", "wsvt"


import cupy as cp
import ptwt
import torch

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
from ..utils.swv import sliding_window_view


# TODO nin17: use pycudwt or pycudaDWT or similar if they eventually support 'axis'
def wavedec(data, wavelet, mode="symmetric", level=None, axis=-1):
    return tuple(
        cp.asarray(i)
        for i in ptwt.wavedec(
            torch.as_tensor(data, device="cuda"), wavelet, mode, level, axis
        )
    )


find_displacement = create_find_displacement(cp)

similarity_st = create_similarity_st(cp, sliding_window_view)
similarity_svt = create_similarity_svt(cp, sliding_window_view)

vectors_st = create_vectors_st(sliding_window_view)
vectors_st_svt = create_vectors_st_svt(cp, sliding_window_view)

wst = create_wst(cp, vectors_st, similarity_st, find_displacement, wavedec)
wst_wsvt = create_wst_wsvt(
    cp, vectors_st_svt, similarity_st, find_displacement, wavedec
)
wsvt = create_wsvt(cp, similarity_svt, find_displacement, wavedec)
