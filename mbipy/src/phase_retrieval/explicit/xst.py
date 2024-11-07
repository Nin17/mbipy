"""_summary_

"""

__all__ = ("xst", "xst_xsvt", "xsvt")

import warnings

import numpy as np
from numpy import floating
from numpy.typing import NDArray

from ...utils import array_namespace
from .config import PAD_MODE
from .utils import assert_odd, find_displacement, get_swv


def _level_cutoff_warning(*tuples, level_cutoff=None):
    if level_cutoff is None:
        return
    for i in tuples:
        if len(i) < level_cutoff:
            warnings.warn(
                f"Level cutoff is too high to take effect, there are {len(i)} levels."
            )


def _cutoff_warning(*arrays, cutoff=None, axis=-1):
    if cutoff is None:
        return
    for i in arrays:
        if i.shape[axis] < cutoff:
            warnings.warn(
                f"Cutoff is too high to take effect, there are {i.shape[axis]} points."
            )


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


def _similarity_st(img1_swv, img2_swv, ss):
    xp = array_namespace(img1_swv, img2_swv)
    swv = get_swv(xp)

    img1_swv_norm = xp.linalg.norm(img1_swv, axis=-1)

    img1_swv = img1_swv / img1_swv_norm[..., None]

    img1_swv_swv = swv(img1_swv, ss, axis=(-3, -2))
    return xp.einsum(
        "...ijklm, ...ijklm -> ...ijlm",
        img1_swv_swv,
        img2_swv[..., None, None],
        optimize="optimal",
    )


def _similarity_svt(img1, img2, m, n):
    # TODO nin17: docstring
    xp = array_namespace(img1, img2)
    swv = get_swv(xp)
    _m, _n = 2 * m + 1, 2 * n + 1
    norm1 = xp.linalg.norm(img1, axis=-3, keepdims=True)
    img1 = img1 / norm1
    img1_swv = swv(img1, (_m, _n), axis=(-2, -1))
    return xp.einsum(
        "...ijklm, ...ijklm -> ...jklm",
        img1_swv,
        img2[..., None, None],
        optimize="optimal",
    )


def _vectors_st(img1, img2, ss, ts):
    assert img1.ndim == img2.ndim
    assert img1.ndim >= 2
    xp = array_namespace(img1, img2)
    swv = get_swv(xp)

    m1, m2, n1, n2 = None, None, None, None
    if ss[0] > 1:
        m1 = ss[0] // 2
        m2 = -m1
    if ss[1] > 1:
        n1 = ss[1] // 2
        n2 = -n1

    img2 = img2[..., m1:m2, n1:n2]

    _ts = np.zeros(img1.ndim, dtype=np.int64)
    _ts[-2:] = ts
    shape1 = np.array(img1.shape) - _ts
    shape1[-2:] += 1
    shape2 = np.array(img2.shape) - _ts
    shape2[-2:] += 1

    img1_swv = swv(img1, ts, axis=(-2, -1)).reshape(*shape1, -1)
    img2_swv = swv(img2, ts, axis=(-2, -1)).reshape(*shape2, -1)
    return img1_swv, img2_swv


def _vectors_st_svt(img1, img2, ss, ts):
    assert img1.ndim == img2.ndim
    assert img1.ndim >= 3
    xp = array_namespace(img1, img2)
    swv = get_swv(xp)

    m1, m2, n1, n2 = None, None, None, None
    if ss[0] > 1:
        m1 = ss[0] // 2
        m2 = -m1
    if ss[1] > 1:
        n1 = ss[1] // 2
        n2 = -n1

    img2 = img2[..., m1:m2, n1:n2]

    # _ss = xp.zeros(img1.ndim - 1, dtype=np.int64)
    _ts = np.zeros(img1.ndim - 1, dtype=np.int64)
    # _ss[-2:] = ss
    _ts[-2:] = ts

    # idk = _ss[-2:] // 2
    # img2 = img2[..., idk[0] : -idk[0], idk[1] : -idk[1]]
    img1_shape = np.array(img1.shape[:-3] + img1.shape[-2:])
    img2_shape = np.array(img2.shape[:-3] + img2.shape[-2:])
    shape1 = np.array(img1_shape) - _ts
    shape1[-2:] += 1
    shape2 = np.array(img2_shape) - _ts
    shape2[-2:] += 1
    # TODO do it with transpose and reshape
    ts = tuple(ts)
    shape1 = tuple(shape1)
    shape2 = tuple(shape2)
    img1_swv = swv(img1, ts, axis=(-2, -1))
    img1_swv = xp.moveaxis(img1_swv, -5, -3).reshape(*shape1, -1)
    img2_swv = swv(img2, ts, axis=(-2, -1))
    img2_swv = xp.moveaxis(img2_swv, -5, -3).reshape(*shape2, -1)
    return img1_swv, img2_swv


class Xst:
    # TODO(nin17): Implement this
    def __init__(self, reference):
        raise NotImplementedError

    def __call__(self, sample, search_window, template_window):
        raise NotImplementedError


class XstXsvt:
    # TODO(nin17): Implement this
    def __init__(self, reference):
        raise NotImplementedError

    def __call__(self, sample, search_window, template_window):
        raise NotImplementedError


class Xsvt:
    # TODO(nin17): Implement this
    def __init__(self, reference):
        raise NotImplementedError

    def __call__(self, sample, search_window):
        raise NotImplementedError


def xst(
    sample: NDArray[floating],
    reference: NDArray[floating],
    search_window: tuple[int, int],
    template_window: tuple[int, int],
    transform: str | None = None,
    transform_kwargs=None,
    cutoff: int | None = None,
    level_cutoff: int | None = None,
) -> tuple[NDArray[floating], ...]:
    # TODO(nin17): docstring
    # TODO(nin17): annotations
    # TODO nin17: pcc option again

    xp = array_namespace(sample, reference)
    assert_odd(*search_window, *template_window)
    # TODO transmission and darkfield calculation
    sample_v, reference_v = _vectors_st(
        sample, reference, search_window, template_window, xp=xp
    )

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

        _level_cutoff_warning(sample_v, reference_v, level_cutoff=level_cutoff)
        if isinstance(sample_v, list) and isinstance(reference_v, list):
            sample_v = xp.concatenate(sample_v[:level_cutoff], -1)
            reference_v = xp.concatenate(reference_v[:level_cutoff], -1)

    _cutoff_warning(sample_v, reference_v, cutoff=cutoff, axis=-1)
    sample_v = sample_v[..., :cutoff]

    # conj() only necessary for the fourier transform - but ns on real arrays
    reference_v = reference_v[..., :cutoff].conj()

    # .real only necessary for the fourier transform - but ns on real arrays
    similarity = _similarity_st(sample_v, reference_v, search_window).real

    # Pad before finding the sub-pixel displacement
    similarity_padded = xp.pad(
        similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
    )
    displacement = find_displacement(similarity_padded)

    # TODO nin17: implement attenuation & darkfield
    # ??? Can possibly be int32
    indices_y, indices_x = [xp.rint(i).astype(xp.int64) for i in displacement]
    diff_y, diff_x = (
        search_window[0] + template_window[0] - 2,
        search_window[1] + template_window[1] - 2,
    )
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

    return displacement + (
        transmission,
        dark_field,
    )  # , transmission2)    return displacement + (transmission, dark_field)  # , transmission2)


def xst_xsvt(
    sample: NDArray[floating],
    reference: NDArray[floating],
    search_window: tuple[int, int],
    template_window: tuple[int, int],
    transform: str | None = None,
    transform_kwargs=None,
    cutoff: int | None = None,
    level_cutoff: int | None = None,
) -> tuple[NDArray[floating], ...]:
    # TODO nin17: add docstring
    # TODO nin17: pcc option again

    xp = array_namespace(sample, reference)
    assert_odd(*search_window, *template_window)

    sample_v, reference_v = _vectors_st_svt(
        sample, reference, search_window, template_window
    )
    _sample_v = sample_v
    _reference_v = reference_v

    if transform is not None:
        if transform_kwargs is None:
            transform_kwargs = {}
        _transform = transforms[transform]
        transform_kwargs = _transform_kwargs(transform, transform_kwargs, -1)

        _sample_v = _transform(sample_v, **transform_kwargs)
        _reference_v = _transform(reference_v, **transform_kwargs)

        _level_cutoff_warning(_sample_v, _reference_v, level_cutoff=level_cutoff)
        if isinstance(_sample_v, list) and isinstance(_reference_v, list):
            _sample_v = xp.concatenate(_sample_v[:level_cutoff], -1)
            _reference_v = xp.concatenate(_reference_v[:level_cutoff], -1)

    _cutoff_warning(_sample_v, _reference_v, cutoff=cutoff, axis=-1)
    _sample_v = _sample_v[..., :cutoff]
    _reference_v = _reference_v[..., :cutoff].conj()

    similarity = _similarity_st(_sample_v, _reference_v, search_window).real
    similarity_padded = xp.pad(
        similarity, ((0, 0),) * (similarity.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE
    )
    displacement = find_displacement(similarity_padded)

    sample_v_sum = sample_v.sum(axis=-1)
    reference_v_sum = reference_v.sum(axis=-1)
    sample_v_std = sample_v.std(axis=-1)
    reference_v_std = reference_v.std(axis=-1)

    diff_y = search_window[0] + template_window[0] - 2
    diff_x = search_window[1] + template_window[1] - 2

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

    indices_y = xp.clip(indices_y, 0, disp_y.shape[-2] - 1)  # !!! shouldn't need this
    indices_x = xp.clip(indices_x, 0, disp_x.shape[-1] - 1)  # !!! shouldn't need this

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


def xsvt(
    sample: NDArray[floating],
    reference: NDArray[floating],
    search_window: tuple[int, int],
    transform: str | None = None,
    transform_kwargs=None,
    cutoff: int | None = None,
    level_cutoff: int | None = None,
):
    # TODO(nin17): add docstring
    # TODO nin17: pcc option again
    # TODO nin17: annotations
    xp = array_namespace(sample, reference)
    assert_odd(*search_window)
    m, n = search_window
    _sample = sample
    _reference = reference
    if transform is not None:
        if transform_kwargs is None:
            transform_kwargs = {}
        _transform = transforms[transform]
        transform_kwargs = _transform_kwargs(transform, transform_kwargs, -3)

        _sample = _transform(sample, **transform_kwargs)
        _reference = _transform(reference, **transform_kwargs)

        _level_cutoff_warning(_sample, _reference, level_cutoff=level_cutoff)
        if isinstance(_sample, list) and isinstance(_reference, list):
            _sample = xp.concatenate(_sample[:level_cutoff], -3)
            _reference = xp.concatenate(_reference[:level_cutoff], -3)

    _cutoff_warning(_sample, _reference, cutoff=cutoff, axis=-3)
    _sample = _sample[..., :cutoff, :, :]
    _reference = _reference[..., :cutoff, :, :].conj()

    similarity = _similarity_svt(_sample, _reference[..., m:-m, n:-n], m, n).real
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
    indices_y = xp.clip(indices_y, 0, disp_y.shape[-2] - 1)  # !!! shouldn't need this
    indices_x = xp.clip(indices_x, 0, disp_x.shape[-1] - 1)  # !!! shouldn't need this

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
