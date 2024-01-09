"""_summary_
"""

import types
import typing


def create_antisym(xp: types.ModuleType) -> typing.Callable:
    docstring = """
        Decorator to pad the vertical and horizonatl components of the normal
        vector field antisymmetrically.

        Parameters
        ----------
        func : typing.Callable
            Normal integration function that requires the inputs to be
            antisymmetrically padded

        Returns
        -------
        typing.Callable
            Wrapped function with the input padded and the output sliced to the
            original shape
        """

    if hasattr(xp, "block"):

        def antisym(*, gy, gx):
            assert gx.shape[-2:] == gy.shape[-2:]
            assert gx.ndim >= 2
            antisym_gx = xp.block(
                [
                    [gx, -gx[..., ::-1]],
                    [gx[..., ::-1, :], -gx[..., ::-1, ::-1]],
                ]
            )
            antisym_gy = xp.block(
                [
                    [gy, gy[..., ::-1]],
                    [-gy[..., ::-1, :], -gy[..., ::-1, ::-1]],
                ]
            )
            return antisym_gy, antisym_gx

    else:

        def antisym(*, gy, gx):
            assert gx.shape[-2:] == gy.shape[-2:]
            assert gx.ndim >= 2
            y, x = gx.shape[-2:]
            antisym_gx = xp.empty(gx.shape[:-2] + (y * 2, x * 2))
            antisym_gx[..., :y, :x] = gx
            antisym_gx[..., y:, :x] = gx[..., ::-1, :]
            antisym_gx[..., :y, x:] = -gx[..., ::-1]
            antisym_gx[..., y:, x:] = -gx[..., ::-1, ::-1]

            antisym_gy = xp.empty(gx.shape[:-2] + (y * 2, x * 2))
            antisym_gy[..., :y, :x] = gy
            antisym_gy[..., y:, :x] = -gy[..., ::-1, :]
            antisym_gy[..., :y, x:] = gy[..., ::-1]
            antisym_gy[..., y:, x:] = -gy[..., ::-1, ::-1]

            return antisym_gy, antisym_gx

    return antisym
