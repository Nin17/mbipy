"""Tests for normal integration functions with numpy."""

from __future__ import annotations

import array_api_strict as xp
import pytest

from .utils import (
    _Test_arnison,
    _Test_dct_poisson,
    _Test_dst_poisson,
    _Test_frankot,
    _Test_kottler,
    _Test_li,
    _Test_southwell,
)

_integer_index = "Integer index arrays are only allowed with integer indices; got"
_unpack = "too many values to unpack (expected 2)"
_out = "create_binary_func.<locals>.inner() got an unexpected keyword argument 'out'"


class Test_arnison(_Test_arnison):
    xp = xp


class Test_kottler(_Test_kottler):
    xp = xp


class Test_frankot(_Test_frankot):
    xp = xp


class Test_dct_poisson(_Test_dct_poisson):
    xp = xp

    @pytest.mark.xfail(reason=_integer_index, raises=IndexError)
    def test_f64(self) -> None:
        super().test_f64()

    @pytest.mark.xfail(reason=_integer_index, raises=IndexError)
    def test_f32(self) -> None:
        super().test_f32()

    @pytest.mark.xfail(reason=_integer_index, raises=IndexError)
    def test_broadcast(self) -> None:
        super().test_broadcast()


class Test_dst_poisson(_Test_dst_poisson):
    xp = xp

    @pytest.mark.xfail(reason=_integer_index, raises=IndexError)
    def test_f64(self) -> None:
        super().test_f64()

    @pytest.mark.xfail(reason=_integer_index, raises=IndexError)
    def test_f32(self) -> None:
        super().test_f32()

    @pytest.mark.xfail(reason=_integer_index, raises=IndexError)
    def test_broadcast(self) -> None:
        super().test_broadcast()

    @pytest.mark.xfail(reason=_integer_index, raises=IndexError)
    def test_ub(self) -> None:
        super().test_ub()


class Test_li(_Test_li):
    xp = xp

    @pytest.mark.xfail(reason=_out, raises=TypeError)
    def test_f32(self):
        return super().test_f32()

    @pytest.mark.xfail(reason=_out, raises=TypeError)
    def test_f64(self):
        return super().test_f64()

    @pytest.mark.xfail(reason=_unpack, raises=ValueError)
    def test_broadcast(self) -> None:
        super().test_broadcast()


class Test_southwell(_Test_southwell):
    xp = xp

    @pytest.mark.xfail(reason=_out, raises=TypeError)
    def test_f32(self):
        return super().test_f32()

    @pytest.mark.xfail(reason=_out, raises=TypeError)
    def test_f64(self):
        return super().test_f64()
