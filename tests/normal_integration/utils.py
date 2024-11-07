"""_summary_
"""

from functools import cached_property

from mbipy import normal_integration

from ..utils import camel_to_snake

# def camel_to_snake(s):
#     return "".join(["_" + c.lower() if c.isupper() else c for c in s]).lstrip("_")


class BaseTest:
    @cached_property
    def func(self):
        name = camel_to_snake(self.__class__.__name__.removeprefix("Test"))
        return getattr(normal_integration, name)

    def test_broadcast(self):
        gy = self.xp.arange(120, dtype=self.xp.float32).reshape(3, 1, 4, 10)
        gx = self.xp.arange(120, dtype=self.xp.float32).reshape(1, 3, 4, 10)
        assert self.func(gy, gx).shape == (3, 3, 4, 10)

    def test_f32(self):
        gy = self.xp.arange(12, dtype=self.xp.float32).reshape(3, 4)
        gx = self.xp.arange(12, dtype=self.xp.float32).reshape(3, 4)
        assert gy.dtype == gx.dtype
        assert self.func(gy, gx).dtype == gy.dtype

    def test_f64(self):
        gy = self.xp.arange(12, dtype=self.xp.float64).reshape(3, 4)
        gx = self.xp.arange(12, dtype=self.xp.float64).reshape(3, 4)
        assert gy.dtype == gx.dtype
        assert self.func(gy, gx).dtype == gy.dtype


class ArnisonTest(BaseTest):
    __slots__ = ("xp",)


class KottlerTest(BaseTest):
    __slots__ = ("xp",)


class FrankotTest(BaseTest):
    __slots__ = ("xp",)


class DctPoissonTest(BaseTest):
    __slots__ = ("xp",)


class DstPoissonTest(BaseTest):
    __slots__ = ("xp",)
