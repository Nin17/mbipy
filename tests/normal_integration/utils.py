"""Tests for normal integration functions."""

from functools import cached_property
from typing import Callable

import pytest
from numpy.random import random

from mbipy import normal_integration

# M, N = 11, 13
M, N =10, 12
K = M * N

GY64 = random(K).reshape(M, N)
GX64 = random(K).reshape(M, N)
UB64 = random(K).reshape(M, N)
GY32 = GY64.astype("float32", copy=True)
GX32 = GX64.astype("float32", copy=True)


class BaseTestNormalIntegration:
    @cached_property
    def _func(self) -> Callable:
        name = self.__class__.__name__.removeprefix("Test_")
        return getattr(normal_integration, name)

    def test_broadcast(self, **kwargs) -> None:
        """Test that the integration functions can handle stacked broadcasted images."""
        gy = self.xp.asarray(random(5 * K), dtype=self.xp.float32).reshape(5, 1, M, N)
        gx = self.xp.asarray(random(3 * K), dtype=self.xp.float32).reshape(1, 3, M, N)
        assert self._func(gy, gx).shape == (5, 3, M, N)

    def test_f32(self, **kwargs) -> None:
        """Test that the integration functions return f4 if both inputs are f4."""
        gy = self.xp.asarray(GY32)
        gx = self.xp.asarray(GX32)
        assert gy.dtype == gx.dtype
        assert self._func(gy, gx).dtype == gy.dtype

    def test_f64(self, **kwargs) -> None:
        """Test that the integration functions return f8 if both inputs are f8."""
        gy = self.xp.asarray(GY64)
        gx = self.xp.asarray(GX64)
        assert gy.dtype == gx.dtype
        assert self._func(gy, gx, **kwargs).dtype == gy.dtype


class FFTTest(BaseTestNormalIntegration):
    def test_broadcast(self, **kwargs) -> None:
        """Test that the integration functions can handle stacked broadcasted images."""
        super().test_broadcast(use_rfft=True)
        super().test_broadcast(use_rfft=False)

    def test_f32(self) -> None:
        """Test that the FFT integration functions return f4 if both inputs are f4."""
        super().test_f32(use_rfft=True)
        super().test_f32(use_rfft=False)

    def test_f64(self) -> None:
        """Test that the FFT integration functions return f8 if both inputs are f8."""
        super().test_f64(use_rfft=True)
        super().test_f64(use_rfft=False)

    def test_padding(self) -> None:
        """Test that the FFT integration functions support anti-symmetric padding."""
        gy = self.xp.asarray(GY32)
        gx = self.xp.asarray(GX32)
        assert self._func(gy, gx, pad="antisym").shape == (M, N)

    # def test_use_rfft(self) -> None:
    #     """Test that the FFT integration functions support use_rfft."""
    #     gy = self.xp.asarray(GY64)
    #     gx = self.xp.asarray(GX64)
    #     idk = self._func(gy, gx, use_rfft=True)
    #     idk2 = self._func(gy, gx, use_rfft=False)
    #     print(self.__class__.__name__, self.xp.abs(idk - idk2).max())
    #     print((idk - idk2).std())
    #     print((idk - idk2).mean())
    #     assert self.xp.allclose(self._func(gy, gx, use_rfft=True), self._func(gy, gx, use_rfft=False))


class LstsqTest(BaseTestNormalIntegration):
    @pytest.mark.xfail(reason="Not implemented")
    def test_broadcast(self) -> None:
        """Sparse least squares based integration doesn't support stacked inputs."""
        return super().test_broadcast()


class _Test_arnison(FFTTest):
    __slots__ = ("xp",)


class _Test_kottler(FFTTest):
    __slots__ = ("xp",)


class _Test_frankot(FFTTest):
    __slots__ = ("xp",)


class _Test_dct_poisson(BaseTestNormalIntegration):
    __slots__ = ("xp",)


class _Test_dst_poisson(BaseTestNormalIntegration):
    __slots__ = ("xp",)

    def test_ub(self) -> None:
        """Test that the dst_poisson function supports boundary conditions with ub."""
        gy = self.xp.asarray(GY64)
        gx = self.xp.asarray(GX64)
        ub = self.xp.asarray(UB64)
        assert self._func(gy, gx, ub).shape == (M, N)


class _Test_li(LstsqTest):
    __slots__ = ("xp",)


class _Test_southwell(LstsqTest):
    __slots__ = ("xp",)
