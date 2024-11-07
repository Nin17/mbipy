"""_summary_
"""

__all__ = (
    "create_lcs_matrices",
    "create_lcs_vectors",
    "create_lcs",
    "create_lcs_df",
    "create_lcs_df_matrices",
)

# __all__ = ("lcs", "lcs_df", "lcs_ddf", "Lcs", "LcsDf", "LcsDDf")

import importlib


def create_lcs_matrices(xp):
    def lcs_matrices(reference):
        assert reference.ndim >= 3
        gradient = xp.gradient(reference, axis=(-2, -1))
        matrices = xp.stack((reference, -gradient[0], -gradient[1]), axis=-1)
        order = tuple(range(0, matrices.ndim - 4)) + (-3, -2, -4, -1)
        return matrices.transpose(order)
        # return xp.moveaxis(matrices, -4, -2) # TODO nin17: see if transpose is faster

    return lcs_matrices


def create_lcs_vectors():
    def lcs_vectors(sample):
        assert sample.ndim >= 3
        return sample.transpose(tuple(range(0, sample.ndim - 3)) + (-2, -1, -3))

    return lcs_vectors


def create_lcs(
    lcs_matrices, lcs_vectors, solver, implicit_tracking, jax=False, numba=False
):
    if sum((jax, numba)) > 1:
        raise ValueError("Only one of jax or numba can be True")
    if not jax and not numba:

        def lcs(sample, reference, weak_absorption=True, m=None, n=None, **kwargs):
            matrices = lcs_matrices(reference)
            vectors = lcs_vectors(sample)

            kwargs = {"a": 1, "b": 2} | kwargs

            if all(i is not None for i in (m, n)):
                result = implicit_tracking(matrices, vectors, m, n, **kwargs)
            else:
                kwargs.pop("a", None)
                kwargs.pop("b", None)
                result = solver(matrices, vectors, **kwargs)

            if weak_absorption:
                return result
            result[..., 1:] /= result[..., :1]
            return result

        return lcs
    if jax:

        def lcs_jax(reference, sample, weak_absorption=True, **kwargs):
            matrices = lcs_matrices(reference)
            vectors = lcs_vectors(sample)
            result = solver(matrices, vectors, **kwargs)
            if weak_absorption:
                return result
            result = result.at[..., 1:].divide(result[..., :1])
            return result

    if numba:
        nb = importlib.import_module("numba")
        np = importlib.import_module("numpy")

        # TODO refactor to use guvectorize when it is supported in a jit function

        @nb.extending.register_jitable
        def lcs_numba(reference, sample, weak_absorption=True, alpha=0.0, **kwargs):
            assert reference.shape == sample.shape
            assert reference.ndim == 3
            x, y, z = reference.shape
            matrices = np.empty((y, z, x, 3), dtype=np.float64)
            out = np.empty((y, z, 3), dtype=np.float64)
            # TODO check this alpha stuff
            # alpha = np.asarray(alpha, dtype=np.float64)
            # alpha = alpha.reshape(alpha.shape + [1 for _ in range(matrices.ndim - alpha.ndim)])
            alpha_identity = alpha * np.identity(3, dtype=np.float64)

            sample = sample.transpose(1, 2, 0).copy()
            # TODO nin17: edges
            for j in nb.prange(1, y - 1):
                for k in range(1, z - 1):
                    for i in range(x):
                        matrices[j, k, i, 0] = reference[i, j, k]
                        matrices[j, k, i, 1] = (
                            -reference[i, j + 1, k] + reference[i, j - 1, k]
                        ) / 2.0
                        matrices[j, k, i, 2] = (
                            -reference[i, j, k + 1] + reference[i, j, k - 1]
                        ) / 2.0

            for j in nb.prange(1, y - 1):
                for k in range(1, z - 1):
                    a = matrices[j, k]
                    ata = (a.T @ a) + alpha_identity
                    atb = a.T @ sample[j, k]
                    out[j, k] = np.linalg.solve(ata, atb)

            if weak_absorption:
                return out

            for i in nb.prange(1, y - 1):
                for j in range(1, z - 1):
                    out[i, j, 1:] /= out[i, j, 0]

            return out

        return lcs_numba

    return lcs_jax


def create_lcs_df_matrices(xp, laplace):
    def lcs_matrices(reference):
        assert reference.ndim >= 3
        gradient = xp.gradient(reference, axis=(-2, -1))
        laplacian = laplace(reference)
        matrices = xp.stack((reference, -gradient[0], -gradient[1], laplacian), axis=-1)
        order = tuple(range(0, matrices.ndim - 4)) + (-3, -2, -4, -1)
        return matrices.transpose(order)
        # return xp.moveaxis(matrices, -4, -2) # TODO nin17: see if transpose is faster

    return lcs_matrices


def create_lcs_df(lcs_df_matrices, lcs_df_vectors, solver):
    def lcs(reference, sample, weak_absorption=True, m=None, n=None, **kwargs):
        matrices = lcs_df_matrices(reference)
        vectors = lcs_df_vectors(sample)

        kwargs = {"a": 1, "b": 2} | kwargs

        if all(i is not None for i in (m, n)):
            # result = implicit_tracking(matrices, vectors, m, n, **kwargs)
            pass
            # TODO nin17: lcs_df tracking
        else:
            kwargs.pop("a", None)
            kwargs.pop("b", None)
            result = solver(matrices, vectors, **kwargs)
        # result = solver(matrices, vectors, **kwargs)

        if weak_absorption:
            return result
        result[..., 1:] /= result[..., :1]
        return result

    return lcs


from numpy.typing import ArrayLike, NDArray

from ...utils import Pytree, array_namespace, static_field
from .utils import is_invertible, laplace


def _lcs_matrices(reference: NDArray) -> NDArray:
    xp = array_namespace(reference)
    if not reference.ndim >= 3:
        msg = f"reference must have at least 3 dimensions. {reference.ndim=}."
        raise ValueError(msg)
    gradient = xp.gradient(reference, axis=(-2, -1))
    matrices = xp.stack((reference, -gradient[0], -gradient[1]), axis=-1)
    order = tuple(range(0, matrices.ndim - 4)) + (-3, -2, -4, -1)

    if "torch" in xp.__name__:
        return matrices.permute(*order)
    return matrices.transpose(*order)
    # return xp.moveaxis(matrices, -4, -2) # TODO nin17: see if transpose is faster


def _lcs_df_matrices(reference: NDArray) -> NDArray:
    xp = array_namespace(reference)
    if not reference.ndim >= 3:
        msg = f"reference must have at least 3 dimensions. {reference.ndim=}."
        raise ValueError(msg)
    gradient = xp.gradient(reference, axis=(-2, -1))
    laplacian = laplace(reference)
    matrices = xp.stack((reference, -gradient[0], -gradient[1], laplacian), axis=-1)
    order = tuple(range(0, matrices.ndim - 4)) + (-3, -2, -4, -1)
    if "torch" in xp.__name__:
        return matrices.permute(*order)
    return matrices.transpose(order)


def _lcs_ddf_matrices(reference: NDArray) -> NDArray:
    xp = array_namespace(reference)
    if not reference.ndim >= 3:
        msg = f"reference must have at least 3 dimensions. {reference.ndim=}."
        raise ValueError(msg)
    gy, gx = xp.gradient(reference, axis=(-2, -1))
    gyy, gyx = xp.gradient(gy, axis=(-2, -1))
    gxy, gxx = xp.gradient(gx, axis=(-2, -1))
    # Ignore gxy as np.allclose(gxy, gyx) == True
    matrices = xp.stack((reference, -gy, -gx, gyy, gxx, gyx), axis=-1)
    order = tuple(range(0, matrices.ndim - 4)) + (-3, -2, -4, -1)
    if "torch" in xp.__name__:
        return matrices.permute(*order)
    return matrices.transpose(order)


def _lcs_vectors(sample: NDArray) -> NDArray:
    xp = array_namespace(sample)
    if not sample.ndim >= 3:
        msg = f"sample must have at least 3 dimensions. {sample.ndim=}."
        raise ValueError(msg)
    if "torch" in xp.__name__:
        return sample.permute(*range(0, sample.ndim - 3), -2, -1, -3)
    return sample.transpose(tuple(range(0, sample.ndim - 3)) + (-2, -1, -3))


def _process_slice_arg(
    s: int | tuple[int, int] | None
) -> tuple[int, int] | tuple[None, None]:
    if s is None:
        return None, None
    if isinstance(s, int):
        return s, s
    s0, s1 = s
    return s0, s1


def _process_slice(
    start: int | tuple[int, int] | None,
    stop: int | tuple[int, int] | None,
    step: int | tuple[int, int] | None,
) -> tuple[slice, slice]:
    start0, start1 = _process_slice_arg(start)
    stop0, stop1 = _process_slice_arg(stop)
    step0, step1 = _process_slice_arg(step)
    return slice(start0, stop0, step0), slice(start1, stop1, step1)


def lcs(
    sample: NDArray,
    reference: NDArray,
    weak_absorption: bool = False,
    alpha: ArrayLike = 0.0,
    search_window: int | tuple[int, int] | None = None,
    start: int | tuple[int, int] | None = None,
    stop: int | tuple[int, int] | None = None,
    step: int | tuple[int, int] | None = None,
) -> NDArray:
    xp = array_namespace(reference, sample)
    matrices = _lcs_matrices(reference)
    vectors = _lcs_vectors(sample)

    # if search_window:
    #     result = implicit_tracking(
    #         matrices, vectors, alpha, search_window, start, stop, step
    #     )
    # else:
    #     result = solver(matrices, vectors, alpha)

    alpha = xp.asarray(alpha, dtype=reference.dtype)
    shape = alpha.shape + (1,) * (matrices.ndim - alpha.ndim - 1)
    alpha = alpha.reshape(shape)
    transpose = tuple(range(matrices.ndim - 4)) + (-4, -3, -1, -2)
    ata = (matrices.transpose(transpose) @ matrices) + alpha * xp.eye(3)
    atb = matrices.transpose(transpose) @ vectors[..., None]
    result = xp.linalg.solve(ata, atb).squeeze(-1)

    if weak_absorption:
        return result
    elif "jax" in xp.__name__:
        result = result.at[..., 1:].divide(result[..., :1])
    else:
        result[..., 1:] /= result[..., :1]
    return result


def lcs_df(
    sample: NDArray,
    reference: NDArray,
    weak_absorption: bool = False,
    alpha: ArrayLike = 0.0,
    search_window: int | tuple[int, int] | None = None,
    start: int | tuple[int, int] | None = None,
    stop: int | tuple[int, int] | None = None,
    step: int | tuple[int, int] | None = None,
) -> NDArray:
    xp = array_namespace(reference, sample)
    matrices = _lcs_df_matrices(reference)
    vectors = _lcs_vectors(sample)
    if search_window:
        result = implicit_tracking(
            matrices, vectors, alpha, search_window, start, stop, step
        )
    else:
        result = solver(matrices, vectors, alpha)

    if weak_absorption:
        return result
    elif "jax" in xp.__name__:
        result = result.at[..., 1:].divide(result[..., :1])
    else:
        result[..., 1:] /= result[..., :1]
    return result


def lcs_ddf(
    sample: NDArray,
    reference: NDArray,
    weak_absorption: bool = False,
    alpha: ArrayLike = 0.0,
    search_window: int | tuple[int, int] | None = None,
    start: int | tuple[int, int] | None = None,
    stop: int | tuple[int, int] | None = None,
    step: int | tuple[int, int] | None = None,
) -> NDArray:
    xp = array_namespace(reference, sample)
    matrices = _lcs_ddf_matrices(reference)
    vectors = _lcs_vectors(sample)
    if search_window:
        result = implicit_tracking(
            matrices, vectors, alpha, search_window, start, stop, step
        )
    else:
        result = solver(matrices, vectors, alpha)

    if weak_absorption:
        return result
    elif "jax" in xp.__name__:
        result = result.at[..., 1:].divide(result[..., :1])
    else:
        result[..., 1:] /= result[..., :1]
    return result


# try:
#     import numba as nb

#     __have_numba__ = True
# except ImportError:
#     __have_numba__ = False

# if __have_numba__:
#     import numpy as np

# @nb.extending.overload(lcs)
# def overload_lcs(
#     sample,
#     reference,
#     weak_absorption=False,
#     alpha=0.0,
#     search_window=None,
#     start=None,
#     stop=None,
#     step=None,
# ):
#     def lcs_numba(
#         sample,
#         reference,
#         weak_absorption=False,
#         alpha=0.0,
#         search_window=None,
#         start=None,
#         stop=None,
#         step=None,
#     ):
#         assert reference.shape == sample.shape
#         assert reference.ndim == 3
#         x, y, z = reference.shape
#         matrices = np.empty((y, z, x, 3), dtype=np.float64)
#         out = np.empty((y, z, 3), dtype=np.float64)
#         # TODO check this alpha stuff
#         # alpha = np.asarray(alpha, dtype=np.float64)
#         # alpha = alpha.reshape(alpha.shape + [1 for _ in range(matrices.ndim - alpha.ndim)])
#         alpha_identity = alpha * np.identity(3, dtype=np.float64)

#         sample = sample.transpose(1, 2, 0).copy()
#         # TODO nin17: edges
#         for j in nb.prange(1, y - 1):
#             for k in range(1, z - 1):
#                 for i in range(x):
#                     matrices[j, k, i, 0] = reference[i, j, k]
#                     matrices[j, k, i, 1] = (
#                         -reference[i, j + 1, k] + reference[i, j - 1, k]
#                     ) / 2.0
#                     matrices[j, k, i, 2] = (
#                         -reference[i, j, k + 1] + reference[i, j, k - 1]
#                     ) / 2.0

#         for j in nb.prange(1, y - 1):
#             for k in range(1, z - 1):
#                 a = matrices[j, k]
#                 ata = (a.T @ a) + alpha_identity
#                 atb = a.T @ sample[j, k]
#                 out[j, k] = np.linalg.solve(ata, atb)

#         if weak_absorption:
#             return out

#         for i in nb.prange(1, y - 1):
#             for j in range(1, z - 1):
#                 out[i, j, 1:] /= out[i, j, 0]

#         return out

#     return lcs_numba


from types import ModuleType
from typing import Callable


class _BaseLcs(Pytree, mutable=True):

    _xp = static_field()

    def __init__(
        self,
        reference: NDArray,
        alpha: ArrayLike,
        rcond: ArrayLike,
        xp: ModuleType | None,
        matrices_func: Callable[[NDArray], NDArray],
    ):
        _xp = array_namespace(reference)
        xp = xp or _xp
        self._xp = xp

        self.reference = reference
        matrices = matrices_func(reference)
        self._matrices = matrices

        _alpha = xp.asarray(alpha, dtype=reference.dtype)
        self._alpha = _alpha.reshape(
            _alpha.shape + (1,) * (matrices.ndim - _alpha.ndim - 1)
        )

        shape_max = float(max(matrices.shape[-2:]))
        rcond_min = xp.asarray(xp.finfo(reference.dtype).eps * shape_max)
        _rcond = xp.maximum(xp.asarray(rcond, dtype=reference.dtype), rcond_min)
        self._rcond = _rcond.reshape(
            _rcond.shape + (1,) * (matrices.ndim - _rcond.ndim - 1)
        )

        _u, _s, _vh = _xp.linalg.svd(matrices, full_matrices=False)
        u, s, vh = xp.asarray(_u), xp.asarray(_s), xp.asarray(_vh)
        self._u = u
        self._s = s
        self._s2 = s**2
        self._vh = vh
        s_max = xp.max(s, axis=-1, keepdims=True)
        self._s_max = s_max
        self._s_max_rcond = s_max * self._rcond

        transposed = tuple(range(matrices.ndim - 4)) + (-4, -3, -1, -2)
        if "torch" in xp.__name__:
            self._vht = vh.permute(*transposed)
            self._ut = u.permute(*transposed)
        else:
            self._vht = vh.transpose(*transposed)  # .mT
            self._ut = u.transpose(*transposed)  # .mT
        # self._vhtut = vh.transpose(transposed) @ u.transpose(transposed)

        self._compute_tikhonov_alpha()
        self._compute_tikhonov_rcond()
        self._compute_tikhonov()

        # self._vht_tikhonov = vh.transpose(transposed) * self._tikhonov

        self._pinv()

    def __call__(
        self,
        sample: NDArray,
        weak_absorption: bool = False,
        search_window: int | tuple[int, int] | None = None,
        start: int | tuple[int, int] | None = None,
        stop: int | tuple[int, int] | None = None,
        step: int | tuple[int, int] | None = None,
    ) -> NDArray:
        # TODO(nin17): implement implicit tracking
        xp = array_namespace(self._matrices, sample)
        vectors = _lcs_vectors(sample)

        result = (self.pinv @ vectors[..., None]).squeeze(-1)

        if weak_absorption:
            return result
        elif "jax" in xp.__name__:
            result = result.at[..., 1:].divide(result[..., :1])
        else:
            result[..., 1:] /= result[..., :1]
        return result

    @property
    def alpha(self) -> NDArray:
        return self._alpha

    @alpha.setter
    def alpha(self, value: ArrayLike) -> None:
        xp = self._xp
        value = xp.asarray(value, dtype=self.reference.dtype)
        shape = value.shape + (1,) * (self._matrices.ndim - value.ndim - 1)
        self._alpha = value.reshape(shape)
        self._compute_tikhonov_alpha()
        self._compute_tikhonov()
        self._pinv()

    @property
    def rcond(self) -> NDArray:
        return self._rcond

    @rcond.setter
    def rcond(self, value: ArrayLike) -> None:
        xp = self._xp
        value = xp.asarray(value, dtype=self.reference.dtype)
        shape = value.shape + (1,) * (self._matrices.ndim - value.ndim - 1)
        self._rcond = value.reshape(shape)
        self._s_max_rcond = self._s_max * self.rcond
        self._compute_tikhonov_rcond()
        self._compute_tikhonov()
        self._pinv()

    @property
    def xp(self) -> ModuleType:
        return self._xp

    @xp.setter
    def xp(self, value: ModuleType) -> None:
        # Convert all necessary arrays to new xp
        # ??? possibly only convert pinv as that is only one strictly required
        # ??? others can be converted as required
        self._xp = value
        self.pinv = value.asarray(self.pinv)
        self._rcond = value.asarray(self._rcond)
        self._alpha = value.asarray(self._alpha)
        self._s = value.asarray(self._s)
        self._s2 = value.asarray(self._s2)
        self._s_max = value.asarray(self._s_max)
        self._s_max_rcond = value.asarray(self._s_max_rcond)
        self._tikhonov = value.asarray(self._tikhonov)
        self._tikhonov_alpha = value.asarray(self._tikhonov_alpha)
        self._tikhonov_rcond = value.asarray(self._tikhonov_rcond)
        self._vht = value.asarray(self._vht)
        self._vht_tikhonov = value.asarray(self._vht_tikhonov)
        self._ut = value.asarray(self._ut)

    def _compute_tikhonov_alpha(self) -> None:
        # Call when alpha changes
        self._tikhonov_alpha = self._s / (self._s2 + self.alpha)

    def _compute_tikhonov_rcond(self) -> None:
        # Call when rcond changes
        self._tikhonov_rcond = self._s < self._s_max_rcond

    def _compute_tikhonov(self) -> None:
        # Call when alpha or rcond changes
        # After _compute_tikhonov_alpha or _compute_tikhonov_rcond
        self._tikhonov = self.xp.where(self._tikhonov_rcond, 0.0, self._tikhonov_alpha)
        self._vht_tikhonov = self._vht * self._tikhonov[..., None, :]

    def _pinv(self) -> None:
        # Pseudo-inverse with Tikhonov regularization
        self.pinv = self._vht_tikhonov @ self._ut


class Lcs(_BaseLcs, mutable=True):
    def __init__(
        self,
        reference: NDArray,
        alpha: ArrayLike = 0.0,
        rcond: ArrayLike = 0.0,
        *,
        xp: ModuleType | None = None,
    ):
        super().__init__(reference, alpha, rcond, xp, _lcs_matrices)


class LcsDf(_BaseLcs, mutable=True):
    def __init__(
        self,
        reference: NDArray,
        alpha: ArrayLike = 0.0,
        rcond: ArrayLike = 0.0,
        *,
        xp: ModuleType | None = None,
    ):
        super().__init__(reference, alpha, rcond, xp, _lcs_df_matrices)


class LcsDDf(_BaseLcs, mutable=True):
    def __init__(
        self,
        reference: NDArray,
        alpha: ArrayLike = 0.0,
        rcond: ArrayLike = 0.0,
        *,
        xp: ModuleType | None = None,
    ):
        super().__init__(reference, alpha, rcond, xp, _lcs_ddf_matrices)
