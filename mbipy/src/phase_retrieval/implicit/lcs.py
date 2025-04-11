"""_summary_"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

from array_api_compat import is_jax_namespace, is_torch_namespace

from mbipy.src.config import __have_numba__
from mbipy.src.phase_retrieval.explicit.utils import assert_odd, get_swv
from mbipy.src.phase_retrieval.implicit.utils import laplace32  # is_invertible,
from mbipy.src.utils import Pytree, array_namespace, static_field

if TYPE_CHECKING:
    from types import ModuleType

    from numpy import floating
    from numpy.typing import ArrayLike, DTypeLike, NDArray

# __all__ = ("lcs", "lcs_df", "lcs_ddf", "Lcs", "LcsDf", "LcsDDf")

MIN_NDIM = 3

# !!! gradient not in array api standard
# !!! .squeeze() not in array api standard


def _lcs_matrices(reference: NDArray[floating]) -> NDArray[floating]:
    xp = array_namespace(reference)
    if not reference.ndim >= MIN_NDIM:
        msg = f"reference must have at least 3 dimensions. {reference.ndim=}."
        raise ValueError(msg)
    gradient = xp.gradient(reference, axis=(-3, -2))
    return xp.stack((reference, -gradient[0], -gradient[1]), axis=-1)


def _lcs_df_matrices(reference: NDArray) -> NDArray:
    xp = array_namespace(reference)
    if not reference.ndim >= MIN_NDIM:
        msg = f"reference must have at least 3 dimensions. {reference.ndim=}."
        raise ValueError(msg)
    gradient = xp.gradient(reference, axis=(-3, -2))
    laplacian = laplace32(reference)
    return xp.stack((reference, -gradient[0], -gradient[1], laplacian), axis=-1)


def _lcs_ddf_matrices(reference: NDArray) -> NDArray:
    xp = array_namespace(reference)
    if not reference.ndim >= MIN_NDIM:
        msg = f"reference must have at least 3 dimensions. {reference.ndim=}."
        raise ValueError(msg)
    gy, gx = xp.gradient(reference, axis=(-3, -2))
    gyy, gyx = xp.gradient(gy, axis=(-3, -2))
    gxy, gxx = xp.gradient(gx, axis=(-3, -2))
    # Ignore gxy as np.allclose(gxy, gyx) == True
    return xp.stack((reference, -gy, -gx, gyy, gxx, gyx), axis=-1)


def _lcs_vectors(sample: NDArray) -> NDArray:
    if not sample.ndim >= MIN_NDIM:
        msg = f"sample must have at least 3 dimensions. {sample.ndim=}."
        raise ValueError(msg)
    return sample


def _process_alpha(
    alpha: ArrayLike,
    n: int,
    m_ndim: int,
    xp: ModuleType,
    dtype: DTypeLike,
) -> NDArray[floating]:
    alpha = xp.asarray(alpha, dtype=dtype) if isinstance(alpha, (int, float)) else alpha
    shape = alpha.shape + (1,) * (m_ndim - alpha.ndim - 1)
    return alpha.reshape(shape) * xp.eye(n, dtype=dtype)


def _solve(
    matrices: NDArray[floating],
    vectors: NDArray[floating],
    alpha: NDArray[floating],
) -> NDArray[floating]:
    xp = array_namespace(matrices, vectors, alpha)
    mtconj = xp.conj(matrices.mT)
    ata = (mtconj @ matrices) + alpha
    atb = mtconj @ vectors[..., None]
    return xp.linalg.solve(ata, atb)  # .squeeze(-1).real


def _process_slice_arg(
    s: int | tuple[int, int] | None,
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


def _solve_window(
    matrices: NDArray[floating],
    vectors: NDArray[floating],
    alpha: ArrayLike,
    search_window: tuple[int, int],
    start: int | tuple[int, int] | None,
    stop: int | tuple[int, int] | None,
    step: int | tuple[int, int] | None,
    xp: ModuleType,
):
    raise NotImplementedError
    assert_odd(search_window)
    m, n = search_window
    _m, _n = (m - 1) // 2, (n - 1) // 2
    matrices = matrices[..., _m:-_m, _n:-_n, :, :]
    swv = get_swv(xp)
    s0, s1 = _process_slice(start, stop, step)
    vectors = swv(vectors, search_window, axis=(-3, -2))[..., s0, s1]
    vectors = vectors.reshape(*vectors.shape[:-2], -1)
    result = _solve(matrices, vectors, alpha, xp)
    residuals = (
        xp.einsum("...ij, ...jk->...ik", matrices, result, optimize=True) - vectors
    )


def _lcs(
    sample: NDArray[floating],
    reference: NDArray,
    weak_absorption: bool | None,
    alpha: ArrayLike,
    search_window: int | tuple[int, int] | None,
    slices: tuple[slice, slice] | None,
):
    xp = array_namespace(reference, sample)
    dtype = xp.result_type(reference, sample)
    matrices = _lcs_matrices(reference)
    vectors = _lcs_vectors(sample)
    alpha_eye = _process_alpha(alpha, 3, matrices.ndim, xp, dtype)

    if search_window:
        result = _solve_window(
            matrices,
            vectors,
            alpha_eye,
            search_window,
            slices,
            xp,
        )
    else:
        result = _solve(matrices, vectors, alpha_eye).squeeze(-1)

    if weak_absorption:
        return result
    if is_jax_namespace(xp):
        result = result.at[..., 1:].divide(result[..., :1])
    else:
        result[..., 1:] /= result[..., :1]
    return result


def lcs(
    sample: NDArray,
    reference: NDArray,
    weak_absorption: bool | None = None,
    alpha: ArrayLike = 0.0,
    search_window: int | tuple[int, int] | None = None,
    slices: tuple[slice, slice] | None = None,
) -> NDArray:
    return _lcs(
        sample,
        reference,
        weak_absorption,
        alpha,
        search_window,
        slices,
    )


def _lcs_df(
    sample,
    reference,
    weak_absorption,
    alpha,
    search_window,
    slices,
):
    xp = array_namespace(reference, sample)
    dtype = xp.result_type(reference, sample)
    matrices = _lcs_df_matrices(reference)
    vectors = _lcs_vectors(sample)
    alpha_eye = _process_alpha(alpha, 4, matrices.ndim, xp, dtype)
    if search_window:
        result = _solve_window(
            matrices,
            vectors,
            alpha_eye,
            search_window,
            slices,
            xp,
        )
    else:
        result = _solve(matrices, vectors, alpha_eye).squeeze(-1)

    if weak_absorption:
        return result
    if is_jax_namespace(xp):
        result = result.at[..., 1:].divide(result[..., :1])
    else:
        result[..., 1:] /= result[..., :1]
    return result


def lcs_df(
    sample: NDArray[floating],
    reference: NDArray,
    weak_absorption: bool | None = None,
    alpha: ArrayLike = 0.0,
    search_window: int | tuple[int, int] | None = None,
    slices: tuple[slice, slice] | None = None,
) -> NDArray:
    return _lcs_df(
        sample,
        reference,
        weak_absorption,
        alpha,
        search_window,
        slices,
    )


def _lcs_ddf(
    sample,
    reference,
    weak_absorption,
    alpha,
    search_window,
    slices,
):
    xp = array_namespace(reference, sample)
    dtype = xp.result_type(reference, sample)
    matrices = _lcs_ddf_matrices(reference)
    vectors = _lcs_vectors(sample)
    alpha_eye = _process_alpha(alpha, 6, matrices.ndim, xp, dtype)
    if search_window:
        result = _solve_window(
            matrices,
            vectors,
            alpha_eye,
            search_window,
            slices,
            xp,
        )
    else:
        result = _solve(matrices, vectors, alpha_eye).squeeze(-1)

    if weak_absorption:
        return result
    if is_jax_namespace(xp):
        result = result.at[..., 1:].divide(result[..., :1])
    else:
        result[..., 1:] /= result[..., :1]
    return result


def lcs_ddf(
    sample: NDArray,
    reference: NDArray,
    weak_absorption: bool | None = None,
    alpha: ArrayLike = 0.0,
    search_window: int | tuple[int, int] | None = None,
    slices: tuple[slice, slice] | None = None,
) -> NDArray:
    return _lcs_ddf(
        sample,
        reference,
        weak_absorption,
        alpha,
        search_window,
        slices,
    )


if __have_numba__:
    from numba import extending, prange, types
    from numpy import empty, float64, identity
    from numpy.linalg import solve

    @extending.overload(_lcs, jit_options={"parallel": True, "fastmath": True})
    def _lcs_overload(
        sample,
        reference,
        weak_absorption,
        alpha,
        search_window,
        start,
        stop,
        step,
    ):
        ndim = 3
        if isinstance(alpha, types.Array):
            msg = "only scalar alpha is implemented."
            raise TypeError(msg)

        if sample.ndim != ndim or reference.ndim != ndim:
            msg = "only 3D sample and reference arrays are implemented."
            raise ValueError(msg)
        if not all(i is types.none for i in (search_window, start, stop, step)):
            msg = "search_window, start, stop, and step are not implemented."
            raise ValueError(msg)

        warnings.warn(
            "Numba implementation is not correct for the edge pixels.",
            stacklevel=2,
        )

        # TODO(nin17): complex conjugate
        def impl(
            sample,
            reference,
            weak_absorption,
            alpha,
            search_window,
            start,
            stop,
            step,
        ):
            if sample.shape != reference.shape:
                msg = "sample and reference must have the same shape."
                raise ValueError(msg)
            z, y, x = reference.shape
            # y, x, z = reference.shape
            matrices = empty((y, x, z, 3), dtype=float64)
            out = empty((y, x, 3), dtype=float64)
            # TODO check this alpha stuff
            # alpha = np.asarray(alpha, dtype=np.float64)
            # alpha = alpha.reshape(alpha.shape + [1 for _ in range(matrices.ndim - alpha.ndim)])
            alpha_identity = alpha * identity(3, dtype=float64)

            # sample = sample.transpose(1, 2, 0).copy()
            sample = sample.transpose(1, 2, 0)
            # TODO nin17: edges
            for j in prange(1, y - 1):
                for k in range(1, x - 1):
                    for i in range(z):
                        matrices[j, k, i, 0] = reference[i, j, k]
                        matrices[j, k, i, 1] = (
                            -reference[i, j + 1, k] + reference[i, j - 1, k]
                        ) / 2.0
                        matrices[j, k, i, 2] = (
                            -reference[i, j, k + 1] + reference[i, j, k - 1]
                        ) / 2.0
                        #!!! change to this eventually - ...yxk faster than ...kyx
                        # matrices[j, k, i, 0] = reference[j, k, i]
                        # matrices[j, k, i, 1] = (
                        #     -reference[j + 1, k, i] + reference[j - 1, k, i]
                        # ) / 2.0
                        # matrices[j, k, i, 2] = (
                        #     -reference[j, k + 1, i] + reference[j, k - 1, i]
                        # ) / 2.0
                    a = matrices[j, k]
                    ata = (a.T @ a) + alpha_identity
                    atb = a.T @ sample[j, k]
                    result = solve(ata, atb)
                    if weak_absorption:
                        out[j, k] = result
                    else:
                        out[j, k, 1:] = result[1:] / result[0]
                        out[j, k, 0] = result[0]

            # for j in prange(1, y - 1):
            #     for k in range(1, x - 1):
            #         a = matrices[j, k]
            #         ata = (a.T @ a) + alpha_identity
            #         atb = a.T @ sample[j, k]
            #         out[j, k] = solve(ata, atb)

            # if weak_absorption:
            #     return out

            # for i in prange(1, y - 1):
            #     for j in range(1, x - 1):
            #         out[i, j, 1:] /= out[i, j, 0]

            return out

        return impl


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
            _alpha.shape + (1,) * (matrices.ndim - _alpha.ndim - 1),
        )

        shape_max = float(max(matrices.shape[-2:]))
        rcond_min = xp.asarray(xp.finfo(reference.dtype).eps * shape_max)
        _rcond = xp.maximum(xp.asarray(rcond, dtype=reference.dtype), rcond_min)
        self._rcond = _rcond.reshape(
            _rcond.shape + (1,) * (matrices.ndim - _rcond.ndim - 1),
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

        self._vht = xp.conj(vh.mT)
        self._ut = xp.conj(u.mT)

        self._compute_tikhonov_alpha()
        self._compute_tikhonov_rcond()
        self._compute_tikhonov()

        self._pinv()

    def __call__(
        self,
        sample: NDArray,
        weak_absorption: bool = False,
        search_window: int | tuple[int, int] | None = None,
        slices: tuple[slice, slice] | None = None,
    ) -> NDArray:
        # TODO(nin17): implement implicit tracking
        xp = array_namespace(self._matrices, sample)
        vectors = _lcs_vectors(sample)

        result = (self.pinv @ vectors[..., None]).squeeze(-1)

        if weak_absorption:
            return result
        if is_jax_namespace(xp):
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
    ) -> None:
        super().__init__(reference, alpha, rcond, xp, _lcs_matrices)


class LcsDf(_BaseLcs, mutable=True):
    def __init__(
        self,
        reference: NDArray,
        alpha: ArrayLike = 0.0,
        rcond: ArrayLike = 0.0,
        *,
        xp: ModuleType | None = None,
    ) -> None:
        super().__init__(reference, alpha, rcond, xp, _lcs_df_matrices)


class LcsDDf(_BaseLcs, mutable=True):
    def __init__(
        self,
        reference: NDArray,
        alpha: ArrayLike = 0.0,
        rcond: ArrayLike = 0.0,
        *,
        xp: ModuleType | None = None,
    ) -> None:
        super().__init__(reference, alpha, rcond, xp, _lcs_ddf_matrices)


# ------------------------------------ old methods ----------------------------------- #


# def create_lcs_matrices(xp):
#     def lcs_matrices(reference):
#         assert reference.ndim >= 3
#         gradient = xp.gradient(reference, axis=(-2, -1))
#         matrices = xp.stack((reference, -gradient[0], -gradient[1]), axis=-1)
#         order = tuple(range(matrices.ndim - 4)) + (-3, -2, -4, -1)
#         return matrices.transpose(order)
#         # return xp.moveaxis(matrices, -4, -2) # TODO nin17: see if transpose is faster

#     return lcs_matrices


# def create_lcs_vectors():
#     def lcs_vectors(sample):
#         assert sample.ndim >= 3
#         return sample.transpose(tuple(range(sample.ndim - 3)) + (-2, -1, -3))

#     return lcs_vectors


# def create_lcs(
#     lcs_matrices,
#     lcs_vectors,
#     solver,
#     implicit_tracking,
#     jax=False,
#     numba=False,
# ):
#     if sum((jax, numba)) > 1:
#         raise ValueError("Only one of jax or numba can be True")
#     if not jax and not numba:

#         def lcs(sample, reference, weak_absorption=True, m=None, n=None, **kwargs):
#             matrices = lcs_matrices(reference)
#             vectors = lcs_vectors(sample)

#             kwargs = {"a": 1, "b": 2} | kwargs

#             if all(i is not None for i in (m, n)):
#                 result = implicit_tracking(matrices, vectors, m, n, **kwargs)
#             else:
#                 kwargs.pop("a", None)
#                 kwargs.pop("b", None)
#                 result = solver(matrices, vectors, **kwargs)

#             if weak_absorption:
#                 return result
#             result[..., 1:] /= result[..., :1]
#             return result

#         return lcs
#     if jax:

#         def lcs_jax(reference, sample, weak_absorption=True, **kwargs):
#             matrices = lcs_matrices(reference)
#             vectors = lcs_vectors(sample)
#             result = solver(matrices, vectors, **kwargs)
#             if weak_absorption:
#                 return result
#             result = result.at[..., 1:].divide(result[..., :1])
#             return result

#     if numba:
#         nb = importlib.import_module("numba")
#         np = importlib.import_module("numpy")

#         # TODO refactor to use guvectorize when it is supported in a jit function

#         @nb.extending.register_jitable
#         def lcs_numba(reference, sample, weak_absorption=True, alpha=0.0, **kwargs):
#             assert reference.shape == sample.shape
#             assert reference.ndim == 3
#             x, y, z = reference.shape
#             matrices = np.empty((y, z, x, 3), dtype=np.float64)
#             out = np.empty((y, z, 3), dtype=np.float64)
#             # TODO check this alpha stuff
#             # alpha = np.asarray(alpha, dtype=np.float64)
#             # alpha = alpha.reshape(alpha.shape + [1 for _ in range(matrices.ndim - alpha.ndim)])
#             alpha_identity = alpha * np.identity(3, dtype=np.float64)

#             sample = sample.transpose(1, 2, 0).copy()
#             # TODO nin17: edges
#             for j in nb.prange(1, y - 1):
#                 for k in range(1, z - 1):
#                     for i in range(x):
#                         matrices[j, k, i, 0] = reference[i, j, k]
#                         matrices[j, k, i, 1] = (
#                             -reference[i, j + 1, k] + reference[i, j - 1, k]
#                         ) / 2.0
#                         matrices[j, k, i, 2] = (
#                             -reference[i, j, k + 1] + reference[i, j, k - 1]
#                         ) / 2.0

#             for j in nb.prange(1, y - 1):
#                 for k in range(1, z - 1):
#                     a = matrices[j, k]
#                     ata = (a.T @ a) + alpha_identity
#                     atb = a.T @ sample[j, k]
#                     out[j, k] = np.linalg.solve(ata, atb)

#             if weak_absorption:
#                 return out

#             for i in nb.prange(1, y - 1):
#                 for j in range(1, z - 1):
#                     out[i, j, 1:] /= out[i, j, 0]

#             return out

#         return lcs_numba

#     return lcs_jax


# def create_lcs_df_matrices(xp, laplace):
#     def lcs_matrices(reference):
#         assert reference.ndim >= 3
#         gradient = xp.gradient(reference, axis=(-2, -1))
#         laplacian = laplace(reference)
#         matrices = xp.stack((reference, -gradient[0], -gradient[1], laplacian), axis=-1)
#         order = tuple(range(matrices.ndim - 4)) + (-3, -2, -4, -1)
#         return matrices.transpose(order)
#         # return xp.moveaxis(matrices, -4, -2) # TODO nin17: see if transpose is faster

#     return lcs_matrices


# def create_lcs_df(lcs_df_matrices, lcs_df_vectors, solver):
#     def lcs(reference, sample, weak_absorption=True, m=None, n=None, **kwargs):
#         matrices = lcs_df_matrices(reference)
#         vectors = lcs_df_vectors(sample)

#         kwargs = {"a": 1, "b": 2} | kwargs

#         if all(i is not None for i in (m, n)):
#             # result = implicit_tracking(matrices, vectors, m, n, **kwargs)
#             pass
#             # TODO nin17: lcs_df tracking
#         else:
#             kwargs.pop("a", None)
#             kwargs.pop("b", None)
#             result = solver(matrices, vectors, **kwargs)
#         # result = solver(matrices, vectors, **kwargs)

#         if weak_absorption:
#             return result
#         result[..., 1:] /= result[..., :1]
#         return result

#     return lcs
#     return lcs
