"""_summary_
"""
__all__ = (
    "csvt",
    "wsvt",
    "xst",
    "xsvt",
)

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy import fft
from jaxwt import wavedec

from ...src.phase_retrieval.explicit import (
    create_csvt,
    create_find_displacement,
    create_similarity_st,
    create_similarity_svt,
    create_vectors_st,
    create_wsvt,
    create_xst,
    create_xsvt,
)

# TODO nin17: experiment with different jax sliding window view methods
# from ..utils.swv import moving_window2d as swv
from ..utils.swv import swv

wavedec = partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))(wavedec)

find_displacement = jax.jit(create_find_displacement(jnp))

similarity_st = partial(jax.jit, static_argnums=(2, 3))(create_similarity_st(jnp, swv))
similarity_svt = partial(jax.jit, static_argnums=(2, 3, 4))(
    create_similarity_svt(jnp, swv)
)


vectors_st = partial(jax.jit, static_argnums=(2, 3))(create_vectors_st(swv))


csvt = partial(jax.jit, static_argnums=(2, 3, 4))(
    create_csvt(jnp, similarity_svt, find_displacement, fft.dct)
)


wsvt = partial(jax.jit, static_argnums=(2, 3, 4))(
    create_wsvt(jnp, similarity_svt, find_displacement, wavedec)
)

xst = create_xst(jnp, vectors_st, similarity_st, find_displacement)
xsvt = partial(jax.jit, static_argnums=(2, 3, 4))(
    create_xsvt(jnp, similarity_svt, find_displacement)
)
