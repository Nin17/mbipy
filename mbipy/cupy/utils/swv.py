"""_summary_
"""

import cupy as cp
import numpy as np

if hasattr(cp.lib.stride_tricks, "sliding_window_view"):
    sliding_window_view = cp.lib.stride_tricks.sliding_window_view
else:
    def sliding_window_view(
        x, window_shape, axis=None, *, subok=False, writeable=False
    ):
        # Copied from cupy/lib/stride_tricks.py

        window_shape = (
            tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
        )

        # writeable is not supported:
        if writeable:
            raise NotImplementedError("Writeable views are not supported.")

        # first convert input to array, possibly keeping subclass
        x = cp.array(x, copy=False, subok=subok)

        window_shape_array = cp.array(window_shape)
        for dim in window_shape_array:
            if dim < 0:
                raise ValueError("`window_shape` cannot contain negative values")

        if axis is None:
            axis = tuple(range(x.ndim))
            if len(window_shape) != len(axis):
                raise ValueError(
                    f"Since axis is `None`, must provide "
                    f"window_shape for all dimensions of `x`; "
                    f"got {len(window_shape)} window_shape elements "
                    f"and `x.ndim` is {x.ndim}."
                )
        else:
            axis = cp._core.internal._normalize_axis_indices(axis, x.ndim)
            if len(window_shape) != len(axis):
                raise ValueError(
                    f"Must provide matching length window_shape and "
                    f"axis; got {len(window_shape)} window_shape "
                    f"elements and {len(axis)} axes elements."
                )

        out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

        # note: same axis can be windowed repeatedly
        x_shape_trimmed = list(x.shape)
        for ax, dim in zip(axis, window_shape):
            if x_shape_trimmed[ax] < dim:
                raise ValueError("window shape cannot be larger than input array shape")
            x_shape_trimmed[ax] -= dim - 1
        out_shape = tuple(x_shape_trimmed) + window_shape
        return cp.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)