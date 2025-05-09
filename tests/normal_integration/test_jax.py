"""Tests for normal integration functions with JAX."""

from __future__ import annotations

from functools import partial

try:
    import jax
    import jax.numpy as jnp

    _have_jax = True
except ImportError:
    _have_jax = False
    jnp = None

import pytest

from .utils import (
    _Test_arnison,
    _Test_dct_poisson,
    _Test_dst_poisson,
    _Test_frankot,
    _Test_kottler,
)


@property
def _func(self):
    if hasattr(self, "static_argnames"):
        return partial(jax.jit, static_argnames=self.static_argnames)(
            super(self.__class__, self)._func,
        )
    return jax.jit(super(self.__class__, self)._func)


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
class Test_arnison(_Test_arnison):
    xp = jnp
    static_argnames = ("pad", "use_rfft")
    _func = _func


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
class Test_kottler(_Test_kottler):
    xp = jnp
    static_argnames = ("pad", "use_rfft")
    _func = _func


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
class Test_frankot(_Test_frankot):
    xp = jnp
    static_argnames = ("pad", "use_rfft")
    _func = _func


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
class Test_dct_poisson(_Test_dct_poisson):
    xp = jnp
    _func = _func


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
class Test_dst_poisson(_Test_dst_poisson):
    xp = jnp
    _func = _func
