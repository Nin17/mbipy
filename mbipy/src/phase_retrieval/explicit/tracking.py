"""
Array based implementations of: XST, XSVT & XST-XSVT.
Compatible with numpy, cupy & jax.
"""

from __future__ import annotations

__all__ = "create_xst", "create_xsvt", "create_xst_xsvt", "create_umpa"

import itertools
import types
import typing
import warnings

from numpy.typing import NDArray

PAD_MODE = "reflect"


def _transform_kwargs(transform, kwargs, axis=-1):
    if transform == "cosine":
        _kwargs = {"type": 2, "norm": "ortho"}

    elif transform == "sine":
        _kwargs = {"type": 1, "norm": "ortho"}
    elif transform == "fourier":
        _kwargs = {"norm": "ortho"}

    elif transform == "hartley":
        _kwargs = {"norm": "ortho"}

    elif transform == "wavelet":
        _kwargs = {"mode": "zero", "wavelet": "db2"}

    _kwargs |= {"axis": axis}

    if kwargs.keys() & _kwargs.keys():
        warnings.warn(
            f"""transform_kwargs will be overwritten as the given values will give incorrect results.
                      The defaults for the {transform} transform are {_kwargs}.
                      """
        )

    return kwargs | _kwargs


def level_cutoff_warning(*tuples, level_cutoff=None):
    if level_cutoff is None:
        return
    for i in tuples:
        if len(i) < level_cutoff:
            warnings.warn(
                f"Level cutoff is too high to take effect, there are {len(i)} levels."
            )


def cutoff_warning(*arrays, cutoff=None, axis=-1):
    if cutoff is None:
        return
    for i in arrays:
        if i.shape[axis] < cutoff:
            warnings.warn(
                f"Cutoff is too high to take effect, there are {i.shape[axis]} points."
            )


def create_xst(
    xp: types.ModuleType,
    vectors_st: typing.Callable,
    similarity_st: typing.Callable,
    find_displacement: typing.Callable,
    transforms: dict[str, typing.Callable] = None,
) -> typing.Callable:
    if transforms is None:
        transforms = {}

    def xst(
        sample: NDArray,
        reference: NDArray,
        ss: tuple[int, int],
        ts: tuple[int, int],
        transform: str | None = None,
        transform_kwargs=None,
        cutoff=None,
        level_cutoff=None,
    ) -> tuple[NDArray, NDArray]:
        # TODO transmission and darkfield calculation
        sample_v, reference_v = vectors_st(sample, reference, ss, ts)

        sample_sum = sample_v.sum(axis=-1)
        reference_sum = reference_v.sum(axis=-1)
        sample_std = sample_v.std(axis=-1)
        reference_std = reference_v.std(axis=-1)

        if transform is not None:
            if transform_kwargs is None:
                transform_kwargs = {}
            _transform = transforms[transform]
            transform_kwargs = _transform_kwargs(transform, transform_kwargs, -1)

            sample_v = _transform(sample_v, **transform_kwargs)
            reference_v = _transform(reference_v, **transform_kwargs)

            level_cutoff_warning(sample_v, reference_v, level_cutoff=level_cutoff)
            if isinstance(sample_v, list) and isinstance(reference_v, list):
                sample_v = xp.concatenate(sample_v[:level_cutoff], -1)
                reference_v = xp.concatenate(reference_v[:level_cutoff], -1)

        cutoff_warning(sample_v, reference_v, cutoff=cutoff, axis=-1)
        sample_v = sample_v[..., :cutoff]

        # conj() only necessary for the fourier transform - but ns on real arrays
        reference_v = reference_v[..., :cutoff].conj()

        # .real only necessary for the fourier transform - but ns on real arrays
        similarity = similarity_st(sample_v, reference_v, ss).real

        # Pad before finding the sub-pixel displacement
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        displacement = find_displacement(similarity_padded)

        # TODO nin17: implement attenuation & darkfield
        # ??? Can possibly be int32
        indices_y, indices_x = [xp.rint(i).astype(xp.int64) for i in displacement]
        diff_y, diff_x = ss[0] + ts[0] - 2, ss[1] + ts[1] - 2
        assert not (diff_y % 2 or diff_x % 2)
        diff_y, diff_x = diff_y // 2, diff_x // 2

        disp_y = xp.rint(displacement[0]).astype(xp.int64)
        disp_x = xp.rint(displacement[1]).astype(xp.int64)

        indices_y = disp_y + xp.arange(
            diff_y, diff_y + disp_y.shape[-2], dtype=xp.int64
        ).reshape((1,) * (disp_y.ndim - 2) + (-1, 1))

        indices_x = disp_x + xp.arange(
            diff_x, diff_x + disp_x.shape[-1], dtype=xp.int64
        ).reshape((1,) * (disp_x.ndim - 1) + (-1,))

        # offset_y = xp.arange(diff_y, diff_y + indices_y.shape[-2], dtype=xp.int64)
        # offset_x = xp.arange(diff_x, diff_x + indices_x.shape[-1], dtype=xp.int64)

        # indices_y += offset_y.reshape((1,) * (indices_y.ndim - 2) + (-1, 1))
        # indices_x += offset_x.reshape((1,) * (indices_x.ndim - 1) + (-1,))

        indices_y = xp.clip(0, disp_y.shape[-2] - 1, indices_y)
        indices_x = xp.clip(0, disp_x.shape[-1] - 1, indices_x)
        ndim = sample.ndim - 1
        # preceding = tuple(
        #     xp.arange(j).reshape((1,) * i + (-1,) + (1,) * (ndim - i))
        #     for i, j in enumerate(indices_y.shape[:-2])
        # )
        preceding = ()
        # FIXME nin17: IndexError
        transmission = (
            sample_sum[preceding + (indices_y, indices_x)]
            / reference_sum[preceding + (indices_y, indices_x)]
        )
        dark_field = (
            sample_std[preceding + (indices_y, indices_x)]
            / reference_std[preceding + (indices_y, indices_x)]
        ) / transmission

        # transmission2 = (
        #     sample[preceding + (indices_y, indices_x)] / reference[..., 10:-10, 10:-10]
        # )

        return displacement + (transmission, dark_field)  # , transmission2)

    return xst


def create_xst_xsvt(
    xp: types.ModuleType,
    vectors_st_svt: typing.Callable,
    similarity_st: typing.Callable,
    find_displacement: typing.Callable,
    transforms: dict[str, typing.Callable] = None,
):
    if transforms is None:
        transforms = {}

    def xst_xsvt(
        sample,
        reference,
        ss,
        ts,
        transform: str | None = None,
        transform_kwargs=None,
        cutoff: int | None = None,
        level_cutoff: int | None = None,
    ):
        # TODO nin17: add docstring
        # TODO nin17: pcc option again
        sample_v, reference_v = vectors_st_svt(sample, reference, ss, ts)
        _sample_v = sample_v
        _reference_v = reference_v

        if transform is not None:
            if transform_kwargs is None:
                transform_kwargs = {}
            _transform = transforms[transform]
            transform_kwargs = _transform_kwargs(transform, transform_kwargs, -1)

            _sample_v = _transform(sample_v, **transform_kwargs)
            _reference_v = _transform(reference_v, **transform_kwargs)

            level_cutoff_warning(_sample_v, _reference_v, level_cutoff=level_cutoff)
            if isinstance(_sample_v, list) and isinstance(_reference_v, list):
                _sample_v = xp.concatenate(_sample_v[:level_cutoff], -1)
                _reference_v = xp.concatenate(_reference_v[:level_cutoff], -1)

        cutoff_warning(_sample_v, _reference_v, cutoff=cutoff, axis=-1)
        _sample_v = _sample_v[..., :cutoff]
        _reference_v = _reference_v[..., :cutoff].conj()

        similarity = similarity_st(_sample_v, _reference_v, ss).real
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        displacement = find_displacement(similarity_padded)

        sample_v_sum = sample_v.sum(axis=-1)
        reference_v_sum = reference_v.sum(axis=-1)
        sample_v_std = sample_v.std(axis=-1)
        reference_v_std = reference_v.std(axis=-1)

        diff_y = ss[0] + ts[0] - 2
        diff_x = ss[1] + ts[1] - 2

        assert not (diff_y % 2 or diff_x % 2)
        diff_y, diff_x = diff_y // 2, diff_x // 2

        disp_y = xp.rint(displacement[0]).astype(xp.int64)
        indices_y = disp_y + xp.arange(
            diff_y, sample.shape[-2] - diff_y, dtype=xp.int64
        ).reshape((1,) * (displacement[0].ndim - 2) + (-1, 1))

        disp_x = xp.rint(displacement[1]).astype(xp.int64)
        indices_x = disp_x + xp.arange(
            diff_x, sample.shape[-1] - diff_x, dtype=xp.int64
        ).reshape((1,) * (displacement[1].ndim - 1) + (-1,))

        indices_y = xp.clip(
            indices_y, 0, disp_y.shape[-2] - 1
        )  # !!! shouldn't need this
        indices_x = xp.clip(
            indices_x, 0, disp_x.shape[-1] - 1
        )  # !!! shouldn't need this

        preceding = ()  # TODO nin17: preceding arrays for broadcasting
        transmission = (
            sample_v_sum[preceding + (indices_y, indices_x)]
            / reference_v_sum[preceding + (indices_y, indices_x)]
        )
        dark_field = (
            sample_v_std[preceding + (indices_y, indices_x)]
            / reference_v_std[preceding + (indices_y, indices_x)]
        ) / transmission

        return displacement + (transmission, dark_field)

    return xst_xsvt


def create_xsvt(
    xp: types.ModuleType,
    similarity_svt: typing.Callable,
    find_displacement: typing.Callable,
    transforms: dict[str, typing.Callable] = None,
) -> typing.Callable:
    if transforms is None:
        transforms = {}

    def xsvt(
        sample,
        reference,
        m: int,
        n: int,
        transform: str | None = None,
        transform_kwargs=None,
        cutoff: int | None = None,
        level_cutoff: int | None = None,
    ):
        # TODO nin17: add docstring
        _sample = sample
        _reference = reference
        if transform is not None:
            if transform_kwargs is None:
                transform_kwargs = {}
            _transform = transforms[transform]
            transform_kwargs = _transform_kwargs(transform, transform_kwargs, -3)

            _sample = _transform(sample, **transform_kwargs)
            _reference = _transform(reference, **transform_kwargs)

            level_cutoff_warning(_sample, _reference, level_cutoff=level_cutoff)
            if isinstance(_sample, list) and isinstance(_reference, list):
                _sample = xp.concatenate(_sample[:level_cutoff], -3)
                _reference = xp.concatenate(_reference[:level_cutoff], -3)

        cutoff_warning(_sample, _reference, cutoff=cutoff, axis=-3)
        _sample = _sample[..., :cutoff, :, :]
        _reference = _reference[..., :cutoff, :, :].conj()

        similarity = similarity_svt(_sample, _reference[..., m:-m, n:-n], m, n).real
        similarity_padded = xp.pad(
            similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        displacement = find_displacement(similarity_padded)
        # displacement = find_displacement(similarity)

        sample_sum = sample.sum(axis=-3)
        reference_sum = reference.sum(axis=-3)
        sample_std = sample.std(axis=-3)
        reference_std = reference.std(axis=-3)

        # TODO nin17: do this inplace - numpy & jax versions separately
        disp_y = xp.rint(displacement[0]).astype(xp.int64)
        indices_y = disp_y + xp.arange(m, sample.shape[-2] - m, dtype=xp.int64).reshape(
            (1,) * (displacement[0].ndim - 2) + (-1, 1)
        )
        disp_x = xp.rint(displacement[1]).astype(xp.int64)
        indices_x = disp_x + xp.arange(n, sample.shape[-1] - n, dtype=xp.int64).reshape(
            (1,) * (displacement[1].ndim - 1) + (-1,)
        )

        # This is likely necessary due to problems with the sub-pixel fitting returning
        # values outside of the unpadded region - which it shouldn't
        indices_y = xp.clip(
            indices_y, 0, disp_y.shape[-2] - 1
        )  # !!! shouldn't need this
        indices_x = xp.clip(
            indices_x, 0, disp_x.shape[-1] - 1
        )  # !!! shouldn't need this

        # TODO nin17: need preceeding arrays for broadcasting
        preceding = ()
        transmission = (
            sample_sum[preceding + (indices_y, indices_x)]
            / reference_sum[preceding + (indices_y, indices_x)]
        )
        dark_field = (
            sample_std[preceding + (indices_y, indices_x)]
            / reference_std[preceding + (indices_y, indices_x)]
        ) / transmission
        return displacement + (transmission, dark_field)

    return xsvt


def create_umpa(xp, correlate1d, swv, find_displacement):
    def umpa(
        sample,
        reference,
        ss,
        ts,
        df: bool = False,
        weights: None | bool | tuple["array", "array"] = True,
    ):
        if not all(i % 2 == 1 for i in itertools.chain(ss, ts)):
            raise ValueError("All search and template dimensions must be odd.")

        if weights is None or isinstance(weights, bool) and not weights:
            hamming_2 = xp.ones(ts[0], dtype=float) / ts[0]
            hamming_1 = xp.ones(ts[1], dtype=float) / ts[1]

        elif isinstance(weights, bool) and weights:
            hamming_2 = xp.hamming(ts[0])
            hamming_2 = hamming_2 / hamming_2.sum()

            hamming_1 = xp.hamming(ts[1])
            hamming_1 = hamming_1 / hamming_1.sum()

        else:
            hamming_2 = xp.asarray(weights[0], dtype=float)
            assert hamming_2.size == ts[0] and hamming_2.ndim == 1
            hamming_2 = hamming_2 / hamming_2.sum()
            hamming_1 = xp.asarray(weights[1], dtype=float)
            assert hamming_1.size == ts[1] and hamming_1.ndim == 1
            hamming_1 = hamming_1 / hamming_1.sum()

        hamming2d = hamming_2[:, None] * hamming_1[None]

        n0, n1 = tuple(int(i // 2) for i in ss)
        m0, m1 = tuple(int(i // 2) for i in ts)

        start0 = n0 + m0
        start1 = n1 + m1

        s2 = xp.square(sample)
        r2 = xp.square(reference)

        # convolve1d just calls correlate1d and as hamming is symmetric it doesn't matter
        l1 = correlate1d(s2, hamming_2, mode="constant", axis=-2)
        l1 = correlate1d(l1, hamming_1, mode="constant", axis=-1).sum(axis=-3)

        l1_swv = swv(l1, ss, axis=(-2, -1))

        l3 = correlate1d(r2, hamming_2, mode="constant", axis=-2)
        l3 = correlate1d(l3, hamming_1, mode="constant", axis=-1).sum(axis=-3)

        l3 = l3[..., start0:-start0, start1:-start1]  # "mode='valid'"

        r_swv = swv(reference, ts, axis=(-2, -1))
        r_swv = r_swv * hamming2d
        r_swv = r_swv.transpose(tuple(range(r_swv.ndim - 5)) + (-4, -3, -5, -2, -1))
        r_swv = r_swv.reshape(r_swv.shape[:-3] + (-1,))

        s_swv = swv(sample, ts, axis=(-2, -1))
        s_swv = s_swv.transpose(tuple(range(s_swv.ndim - 5)) + (-4, -3, -5, -2, -1))
        s_swv = s_swv.reshape(s_swv.shape[:-3] + (-1,))
        s_swv_swv = swv(s_swv, ss, axis=(-3, -2))

        l5 = xp.einsum("...k,...klm ->...lm", r_swv[..., n0:-n0, n1:-n1, :], s_swv_swv)
        l5_2 = xp.square(l5)

        if df:
            if sum(ts) < 4:
                raise ValueError("Template size must be larger than 1x1, if df=True.")
            mean = correlate1d(reference, hamming_2, mode="constant", axis=-2)
            mean = correlate1d(mean, hamming_1, mode="constant", axis=-1)
            mean2 = xp.square(mean)

            mean_swv = swv(mean, ts, axis=(-2, -1))
            mean_swv = mean_swv * hamming2d
            mean_swv = mean_swv.transpose(
                tuple(range(mean_swv.ndim - 5)) + (-4, -3, -5, -2, -1)
            )
            mean_swv = mean_swv.reshape(mean_swv.shape[:-3] + (-1,))

            l2 = correlate1d(mean2, hamming_2, mode="constant", axis=-2)
            l2 = correlate1d(l2, hamming_1, mode="constant", axis=-1).sum(axis=-3)

            l4 = xp.einsum(
                "...k,...klm ->...lm", mean_swv[..., n0:-n0, n1:-n1, :], s_swv_swv
            )

            l6 = correlate1d(reference * mean, hamming_2, mode="constant", axis=-2)
            l6 = correlate1d(l6, hamming_1, mode="constant", axis=-1).sum(axis=-3)

            l2 = l2[..., start0:-start0, start1:-start1]  # "mode='valid'"
            l6 = l6[..., start0:-start0, start1:-start1]  # "mode='valid'"

            denominator = l3 * l2 - xp.square(l6)
            denominator = denominator[..., None, None]

            alpha1 = l2[..., None, None] * l5
            alpha2 = l4 * l6[..., None, None]
            alpha = (alpha1 - alpha2) / denominator

            beta1 = l3[..., None, None] * l4
            beta2 = l6[..., None, None] * l5
            beta = (beta1 - beta2) / denominator

            # Negative of the loss: to use the maximum finding find_displacement
            loss = (
                -l1_swv[..., m0 : -m0 or None, m1 : -m1 or None, :, :]
                - xp.square(beta) * l2[..., None, None]
                - xp.square(alpha) * l3[..., None, None]
                + 2.0 * beta * l4
                + 2.0 * alpha * l5
                - 2.0 * alpha * beta * l6[..., None, None]
            )
            _loss = loss.reshape(loss.shape[:-2] + (-1,))
            loss_max = _loss.argmax(axis=-1)
            _alpha = xp.take_along_axis(
                alpha.reshape(alpha.shape[:-2] + (-1,)), loss_max[..., None], axis=-1
            ).squeeze(-1)
            _beta = xp.take_along_axis(
                beta.reshape(beta.shape[:-2] + (-1,)), loss_max[..., None], axis=-1
            ).squeeze(-1)

            transmission = _alpha + _beta
            dark_field = _alpha / transmission
        else:
            # Negative of the loss: to use the maximum finding find_displacement
            loss = (
                l5_2 / l3[..., None, None]
                - l1_swv[..., m0 : -m0 or None, m1 : -m1 or None, :, :]
            )
            _loss = loss.reshape(loss.shape[:-2] + (-1,))
            loss_max = _loss.argmax(axis=-1)

            transmission = (
                xp.take_along_axis(
                    l5.reshape(l5.shape[:-2] + (-1,)), loss_max[..., None], axis=-1
                ).squeeze(-1)
                / l3
            )
            dark_field = None

        similarity_padded = xp.pad(
            loss, ((0, 0),) * (loss.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
        )
        displacement = find_displacement(similarity_padded)
        return displacement + (transmission, dark_field)

    return umpa


# def xst(): ... # TODO(nin17): Implement
# def xsvt(): ... # TODO(nin17): Implement
# def xst_xsvt(): ... # TODO(nin17): Implement
# def umpa(): ... # TODO(nin17): Implement

# class XST: ... # TODO(nin17): Implement
# class XSVT: ... # TODO(nin17): Implement
# class XST_XSVT: ... # TODO(nin17): Implement
# class UMPA: ... # TODO(nin17): Implement
