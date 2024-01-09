"""_summary_
"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(1,))
def _swv_2_1(array, window, axis=(-2, -1)):
    warnings.warn("This function is very memory inefficient during compilation.")
    assert axis == (-2, -1)
    y, x = window
    _y, _x = array.shape[-2:]
    indices = jnp.arange(x)[None, None, None]
    indices = jnp.broadcast_to(indices, (_y - y + 1, _x - x + 1, y, x))
    indices = (
        indices
        + jnp.arange(_y * _x).reshape(_y, _x)[: _y - y + 1, : _x - x + 1][
            ..., None, None
        ]
    )
    indices = indices + jnp.arange(0, y * _x, _x)[None, None, :, None]
    return array.reshape(array.shape[:-2] + (-1,)).take(indices, axis=-1)


@partial(jax.jit, static_argnums=(1,))
def _swv_3_2(array, window, axis=(-3, -2)):
    warnings.warn("This function is very memory inefficient during compilation.")
    assert axis == (-3, -2)
    y, x = window
    _y, _x = array.shape[-3:-1]
    indices = jnp.arange(x)[None, None, None]
    indices = jnp.broadcast_to(indices, (_y - y + 1, _x - x + 1, y, x))
    indices = (
        indices
        + jnp.arange(_y * _x).reshape(_y, _x)[: _y - y + 1, : _x - x + 1][
            ..., None, None
        ]
    )
    indices = indices + jnp.arange(0, y * _x, _x)[None, None, :, None]
    return jnp.moveaxis(
        array.reshape(array.shape[:-3] + (-1,) + array.shape[-1:]).take(
            indices, axis=-2
        ),
        -1,
        -3,
    )


@partial(jax.jit, static_argnums=(1, 2))
def swv(array, window, axis=(-2, -1)):
    # TODO nin17: sliding window view over other axes - particularly (-3, -2)
    assert axis in {(-2, -1), (-3, -2)}
    if axis == (-2, -1):
        return _swv_2_1(array, window)
    elif axis == (-3, -2):
        return _swv_3_2(array, window)


@partial(jax.jit, static_argnums=(1,))
def _moving_window2d(matrix, window_shape):
    matrix_width = matrix.shape[-1]
    matrix_height = matrix.shape[-2]

    window_width = window_shape[0]
    window_height = window_shape[1]

    startsx = jnp.arange(matrix_width - window_width + 1)
    startsy = jnp.arange(matrix_height - window_height + 1)
    starts_xy = jnp.dstack(jnp.meshgrid(startsx, startsy)).reshape(
        -1, 2
    )  # cartesian product => [[x,y], [x,y], ...]

    def _slice_window(start):
        return jax.lax.dynamic_slice(
            matrix, (start[1], start[0]), (window_height, window_width)
        )

    return jax.vmap(_slice_window)(starts_xy).reshape(
        matrix_height - window_height + 1,
        matrix_width - window_width + 1,
        window_height,
        window_width,
    )


@partial(jax.jit, static_argnums=(1, 2))
def moving_window2d(matrix, window_shape, axis):
    assert axis == (-2, -1)
    func = _moving_window2d
    for i in range(matrix.ndim - 2):
        func = jax.vmap(func, in_axes=(i, (None, None)), out_axes=i)
    return func(matrix, window_shape)
