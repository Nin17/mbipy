"""_summary_
"""

import importlib

import numpy as np

from mbipy.src.config import __have_numba__, __have_scipy__
from mbipy.src.utils import array_namespace


def r_(s0, s1, xp):
    _s0 = slice(s0[0], s0[1]) if isinstance(s0, tuple) else s0
    _s1 = slice(s1[0], s1[1]) if isinstance(s1, tuple) else s1
    if "torch" in xp.__name__:
        # TODO(nin17): Implement r_ for PyTorch
        msg = "r_ is not implemented for PyTorch."
        raise NotImplementedError(msg)
    return xp.r_[_s0, _s1]


def check_shapes(*arrays: np.ndarray) -> tuple[int, int]:
    """
    Checks that the shapes of the input arrays are compatible for broadcasting and returns the shape of the last two dimensions.

    Returns
    -------
    tuple[int, int]
        The shape of the last two dimensions of the broadcasted arrays.
    """
    return np.broadcast_shapes(*(i.shape for i in arrays))[-2:]


def get_fft_module(xp):
    if "jax" in xp.__name__:
        return importlib.import_module("jax.numpy.fft")

    if "numpy" in xp.__name__:
        if __have_scipy__:
            return importlib.import_module("scipy.fft")
        return np.fft

    if "torch" in xp.__name__:
        return xp.fft

    # TODO(nin17): cupy


def _remove_workers(func):
    return lambda *args, workers, **kwargs: func(*args, **kwargs)


def _add_axes(func):
    return lambda *args, axes, **kwargs: func(*args, **kwargs)


def get_dfts(xp):

    if "jax" in xp.__name__:
        return _remove_workers(xp.fft.fft2), _remove_workers(xp.fft.ifft2)

    if "numpy" in xp.__name__:
        if __have_scipy__:
            fft = importlib.import_module("scipy.fft")
            return fft.fft2, fft.ifft2
        return np.fft.fft2, np.fft.ifft2

    if "torch" in xp.__name__:
        return _remove_workers(xp.fft.fftn), _remove_workers(xp.fft.ifftn)


def get_dcts(xp):
    if "jax" in xp.__name__:
        fft = importlib.import_module("jax.scipy.fft")
        return _remove_workers(fft.dctn), _remove_workers(fft.idctn)

    if "numpy" in xp.__name__:
        if __have_scipy__:
            fft = importlib.import_module("scipy.fft")
            return fft.dctn, fft.idctn
        raise ValueError("Scipy is required for the DCT.")

    if "torch" in xp.__name__:
        try:
            import torch_dct as tdct
        except ImportError as error:
            raise ImportError("torch_dct is required for the DCT.") from error
        return _add_axes(_remove_workers(tdct.dct_2d)), _add_axes(
            _remove_workers(tdct.idct_2d)
        )


# def isjax(xp):
#     return "jax" in xp.__name__


# def setitem(a, i, v, xp):
#     if isjax(xp):
#         a = a.at[i].set(v)
#     else:
#         a[i] = v
#     return a


if __have_numba__:
    import numba as nb
    from numba import types
    from numpy import arange

    # @nb.extending.overload(array_namespace)
    # def overload_array_namespace(*arrays):
    #     def impl(*arrays):
    #         return np
    #     return impl

    @nb.extending.overload(r_)
    def overload_r_(s0, s1, xp):
        if isinstance(s0, types.Integer) and isinstance(s1, types.UniTuple):
            if s1.count != 2:
                raise ValueError("Tuple argument must have length 2.")

            def impl(s0, s1, xp):
                start = s1[0]
                stop = s1[1]
                out = np.empty(stop - start + 1, dtype=np.int64)
                out[1:] = arange(start, stop)
                out[0] = s0
                return out

            return impl
        if isinstance(s0, types.UniTuple) and isinstance(s1, types.Integer):
            if s0.count != 2:
                raise ValueError("Tuple argument must have length 2.")

            def impl(s0, s1, xp):
                start = s0[0]
                stop = s0[1]
                out = np.empty(stop - start + 1, dtype=np.int64)
                out[:-1] = arange(start, stop)
                out[-1] = s1
                return out

            return impl

    @nb.extending.overload(get_dfts)
    def overload_get_dfts(xp):

        from rocket_fft import scipy_like

        scipy_like()

        if __have_scipy__:
            from scipy import fft as sp_fft

            def impl(xp):
                return sp_fft.fft2, sp_fft.ifft2

        else:
            from numpy import fft

            #!!! RocketFFT allows for workers to be passed as an argument
            def impl(xp):
                return fft.fft2, fft.ifft2

        return impl

    @nb.extending.overload(get_dcts)
    def overload_get_dcts(xp):
        try:
            from scipy import fft as sp_fft
        except ImportError as error:
            raise ImportError("Scipy is required for the DCT.") from error

        def impl(xp):
            return sp_fft.dctn, sp_fft.idctn

        return impl

    @nb.extending.overload(get_fft_module)
    def overload_get_fft_module(xp):
        from scipy import fft

        def impl(xp):
            return fft

        return impl

    @nb.extending.overload(check_shapes)
    def overload_check_shapes(gy, gx):
        def impl(gy, gx):
            return np.broadcast_shapes(gy.shape, gx.shape)[-2:]
            # with nb.objmode(x="int64", y="int64"):
            #     y, x = np.broadcast_shapes(gy.shape, gx.shape)[-2:]
            # return y, x

        return impl

    @nb.extending.overload(np.result_type)
    def overload_result_type(*arrays):
        dtypes = [i.dtype for i in arrays]
        if all(isinstance(i, types.Float) for i in dtypes):
            dtype = max(dtypes)
            
            def impl(*arrays):
                return dtype
            return impl

        elif all(isinstance(i, types.Complex) for i in dtypes):
            dtype = max(dtypes)
            
            def impl(*arrays):
                return dtype
            return impl

        elif all(isinstance(i, types.Integer) for i in dtypes):
            dtype = max(dtypes)
            
            def impl(*arrays):
                return dtype
            return impl

        # else:
        #     ...
        #     # raise ValueError("All arrays must have the same dtype.")

        # def impl(*arrays):
        #     return dtype

        # return impl

    if not hasattr(np, "astype"):
        setattr(np, "astype", lambda x, dtype, copy: x.astype(dtype))

    @nb.extending.overload(np.astype)
    def overload_as_type(x, dtype, copy):
        def impl(x, dtype, copy):
            return x.astype(dtype)

        return impl

    if not hasattr(np, "isdtype"):

        def isdtype(dtype, kind):
            if kind != "real floating":
                msg = "Invalid kind."
                raise ValueError(msg)
            return dtype.kind == "f"

        setattr(np, "isdtype", isdtype)

    @nb.extending.overload(np.isdtype)
    def overload_isdtype(dtype, kind):

        if not isinstance(dtype.dtype, types.Float):
            raise ValueError("Invalid dtype.")

        def impl(dtype, kind):

            return True

        return impl

    # @nb.extending.overload(setitem)
    # def overload_setitem(a, i, v, xp):
    #     def impl(a, i, v, xp):
    #         a[i] = v
    #         return a

    #     return impl
