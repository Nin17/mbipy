"""_summary_
"""

from functools import partial

try:
    import jax
    import jax.numpy as jnp
    _have_jax = True
except ImportError:
    _have_jax = False
    jnp = None
import pytest

from .utils import ArnisonTest, DctPoissonTest, DstPoissonTest, FrankotTest, KottlerTest


def jit(cls):
    @property
    def _func(self):
        static_argnames = None
        return partial(jax.jit, static_argnames=static_argnames)(
            super(cls, self).func,
        )

    cls.func = _func

    return cls


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit
class TestArnison(ArnisonTest):
    xp = jnp


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit
class TestKottler(KottlerTest):
    xp = jnp


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit
class TestFrankot(FrankotTest):
    xp = jnp


@pytest.mark.skipif(not _have_jax, reason="Jax is not installed")
@jit
class TestDctPoisson(DctPoissonTest):
    xp = jnp


@pytest.mark.skip(reason="DST Type 1 only works with scipy")
@jit
class TestDstPoisson(DstPoissonTest):
    xp = jnp
