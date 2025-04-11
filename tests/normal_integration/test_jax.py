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
from typing import TYPE_CHECKING

import pytest

from .utils import (
    _Test_arnison,
    _Test_dct_poisson,
    _Test_dst_poisson,
    _Test_frankot,
    _Test_kottler,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Callable


def jit(cls=None, *, static_argnames: str | Iterable[str] | None = None) -> Callable:
    """Wrap cls._func with JAX's jit decorator."""

    def _jit(cls):
        """Wrap cls._func with JAX's jit decorator."""

        @property
        def _func(self) -> Callable:
            return partial(jax.jit, static_argnames=static_argnames)(
                super(cls, self)._func,
            )

        cls._func = _func

        return cls

    if cls is None:
        return _jit
    return _jit(cls)


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit(static_argnames=("pad", "use_rfft"))
class Test_arnison(_Test_arnison):
    xp = jnp


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit(static_argnames=("pad", "use_rfft"))
class Test_kottler(_Test_kottler):
    xp = jnp


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit(static_argnames=("pad", "use_rfft"))
class Test_frankot(_Test_frankot):
    xp = jnp


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit
class Test_dct_poisson(_Test_dct_poisson):
    xp = jnp


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit
class Test_dst_poisson(_Test_dst_poisson):
    xp = jnp
