"""Tests for normal integration functions."""

from functools import cached_property
from typing import Callable

import pytest
from numpy import float32, float64
from numpy.random import random
from numpy.typing import NDArray

from mbipy import normal_integration

# M, N = 11, 13
M, N = 10, 12
K = M * N

GY64 = random(K).reshape(M, N)
GX64 = random(K).reshape(M, N)
UB64 = random(K).reshape(M, N)
GY32 = GY64.astype("float32", copy=True)
GX32 = GX64.astype("float32", copy=True)


class BaseTestNormalIntegration:

    @cached_property
    def gf64(self) -> NDArray[float64]:
        return self.xp.asarray(GY64), self.xp.asarray(GX64)

    @cached_property
    def gf32(self) -> NDArray[float32]:
        return self.xp.asarray(GY32), self.xp.asarray(GX32)

    # @cached_property
    @property
    def _func(self) -> Callable:
        name = self.__class__.__name__.removeprefix("Test_")
        return getattr(normal_integration, name)

    def _test_broadcast(self, **kwargs) -> None:
        xp = self.xp
        gy = xp.reshape(xp.asarray(random(5 * K), dtype=xp.float32), (5, 1, M, N))
        gx = xp.reshape(xp.asarray(random(3 * K), dtype=self.xp.float32), (1, 3, M, N))
        assert self._func(gy, gx, **kwargs).shape == (5, 3, M, N)

    def test_broadcast(self, **kwargs) -> None:
        """Test that the integration functions can handle stacked broadcasted images."""
        self._test_broadcast(**kwargs)

    def test_f32(self, **kwargs) -> None:
        """Test that the integration functions return f4 if both inputs are f4."""
        gy, gx = self.gf32
        assert self._func(gy, gx, **kwargs).dtype == gy.dtype

    def test_f64(self, **kwargs) -> None:
        """Test that the integration functions return f8 if both inputs are f8."""
        gy, gx = self.gf64
        assert self._func(gy, gx, **kwargs).dtype == gy.dtype

    def test_wrong_dtype(self) -> None:
        """Test that the FFT integration functions raise an error for non-real inputs."""
        gy = self.xp.asarray(GY32, dtype=self.xp.complex64)
        gx = self.xp.asarray(GX32, dtype=self.xp.complex64)
        with pytest.raises(ValueError, match="Input arrays must be real-valued."):
            self._func(gy, gx)


class FFTTest(BaseTestNormalIntegration):
    def test_broadcast(self) -> None:
        """Test that the integration functions can handle stacked broadcasted images."""
        super().test_broadcast(use_rfft=False)
        super().test_broadcast(use_rfft=False)

    def test_f32(self) -> None:
        """Test that the FFT integration functions return f4 if both inputs are f4."""
        super().test_f32(use_rfft=False)
        super().test_f32(use_rfft=True)

    def test_f64(self) -> None:
        """Test that the FFT integration functions return f8 if both inputs are f8."""
        super().test_f64(use_rfft=False)
        super().test_f64(use_rfft=True)

    def test_padding(self) -> None:
        """Test that the FFT integration functions support anti-symmetric padding."""
        gy = self.xp.asarray(GY32)
        gx = self.xp.asarray(GX32)
        assert self._func(gy, gx, pad="antisymmetric").shape == (M, N)
        with pytest.raises(ValueError, match="Invalid value for pad: invalid"):
            self._func(gy, gx, pad="invalid")
        with pytest.raises(ValueError, match="Invalid value for pad: invalid"):
            self._func(gy, gx, pad="invalid", use_rfft=True)

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


class DXTPoissonTest(BaseTestNormalIntegration): ...


class _Test_arnison(FFTTest):
    __slots__ = ("xp",)


class _Test_kottler(FFTTest):
    __slots__ = ("xp",)


class _Test_frankot(FFTTest):
    __slots__ = ("xp",)


class _Test_dct_poisson(DXTPoissonTest):
    __slots__ = ("xp",)

    def test_f32(self) -> None:
        super().test_f32()

    def test_f64(self) -> None:
        super().test_f64()

    def test_wrong_dtype(self) -> None:
        super().test_wrong_dtype()

    def test_broadcast(self) -> None:
        super().test_broadcast()


class _Test_dst_poisson(DXTPoissonTest):
    __slots__ = ("xp",)

    def test_f32(self) -> None:
        super().test_f32()

    def test_f64(self) -> None:
        super().test_f64()

    def test_wrong_dtype(self) -> None:
        super().test_wrong_dtype()

    def test_broadcast(self) -> None:
        super().test_broadcast()

    def test_ub(self) -> None:
        """Test that the dst_poisson function supports boundary conditions with ub."""
        gy = self.xp.asarray(GY64)
        gx = self.xp.asarray(GX64)
        ub = self.xp.asarray(UB64)
        assert self._func(gy, gx, ub).shape == (M, N)


class LstsqTest(BaseTestNormalIntegration):
    @pytest.mark.xfail(reason="Not implemented")
    def test_broadcast(self) -> None:
        """Sparse least squares based integration doesn't support stacked inputs."""
        return super().test_broadcast()


class _Test_li(LstsqTest):
    __slots__ = ("xp",)

    def test_f32(self) -> None:
        super().test_f32()

    def test_f64(self) -> None:
        super().test_f64()

    def test_wrong_dtype(self) -> None:
        super().test_wrong_dtype()

    def test_broadcast(self) -> None:
        super().test_broadcast()


class _Test_southwell(LstsqTest):
    __slots__ = ("xp",)

    def test_f32(self) -> None:
        super().test_f32()


class _Test_Li(LstsqTest): ...


class _Test_Southwell(LstsqTest): ...
