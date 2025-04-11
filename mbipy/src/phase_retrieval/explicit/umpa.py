"""_summary_
"""

__all__ = ("Umpa", "umpa")


from numpy import floating
from numpy.typing import NDArray

from ...utils import array_namespace
from .config import PAD_MODE
from .utils import assert_odd, find_displacement, get_correlate1d, get_swv


class Umpa:
    def __init__(self, sample, weights):
        raise NotImplementedError

    def __call__(self, reference, sw, tw, df):
        pass


# def create_umpa(xp, correlate1d, swv, find_displacement):
def umpa(
    sample: NDArray[floating],
    reference: NDArray[floating],
    search_window: tuple[int, int],
    template_window: tuple[int, int],
    df: bool = False,
    weights: None | bool | tuple[NDArray[floating], NDArray[floating]] = True,
) -> tuple[NDArray[floating], ...]:

    xp = array_namespace(sample, reference)
    correlate1d = get_correlate1d(xp)
    swv = get_swv(xp)

    assert_odd(*search_window, *template_window)
    # if not all(i % 2 == 1 for i in itertools.chain(search_window, template_window)):
    #     raise ValueError("All search and template dimensions must be odd.")

    if weights is None or (isinstance(weights, bool) and not weights):
        hamming_2 = xp.ones(template_window[0], dtype=float) / template_window[0]
        hamming_1 = xp.ones(template_window[1], dtype=float) / template_window[1]

    elif isinstance(weights, bool) and weights:
        hamming_2 = xp.hamming(template_window[0])
        hamming_2 = hamming_2 / hamming_2.sum()

        hamming_1 = xp.hamming(template_window[1])
        hamming_1 = hamming_1 / hamming_1.sum()

    else:
        hamming_2 = xp.asarray(weights[0], dtype=float)
        assert hamming_2.size == template_window[0] and hamming_2.ndim == 1
        hamming_2 = hamming_2 / hamming_2.sum()
        hamming_1 = xp.asarray(weights[1], dtype=float)
        assert hamming_1.size == template_window[1] and hamming_1.ndim == 1
        hamming_1 = hamming_1 / hamming_1.sum()

    hamming2d = hamming_2[:, None] * hamming_1[None]

    n0, n1 = tuple(int(i // 2) for i in search_window)
    m0, m1 = tuple(int(i // 2) for i in template_window)

    start0 = n0 + m0
    start1 = n1 + m1

    # s_2 = xp.square(sample)
    # r_2 = xp.square(reference)
    s_2 = sample * sample.conj()
    r_2 = reference * reference.conj()

    # convolve1d just calls correlate1d and as hamming is symmetric it doesn't matter
    l1 = correlate1d(s_2, hamming_2, mode="constant", axis=-2)
    l1 = correlate1d(l1, hamming_1, mode="constant", axis=-1).sum(axis=-3)

    l1_sw = swv(l1, search_window, axis=(-2, -1))

    l3 = correlate1d(r_2, hamming_2, mode="constant", axis=-2)
    l3 = correlate1d(l3, hamming_1, mode="constant", axis=-1).sum(axis=-3)

    l3 = l3[..., start0:-start0, start1:-start1]  # "mode='valid'"

    r_tw = swv(reference, template_window, axis=(-2, -1))
    r_tw = r_tw * hamming2d
    r_tw = r_tw.transpose(tuple(range(r_tw.ndim - 5)) + (-4, -3, -5, -2, -1))
    r_tw = r_tw.reshape(r_tw.shape[:-3] + (-1,))

    sample = sample.conj()
    s_tw = swv(sample, template_window, axis=(-2, -1))
    s_tw = s_tw.transpose(tuple(range(s_tw.ndim - 5)) + (-4, -3, -5, -2, -1))
    s_tw = s_tw.reshape(s_tw.shape[:-3] + (-1,))
    s_tw_sw = swv(s_tw, search_window, axis=(-3, -2))

    l5 = xp.einsum("...k,...klm ->...lm", r_tw[..., n0:-n0, n1:-n1, :], s_tw_sw)
    l5_2 = xp.square(l5)

    if df:
        if sum(template_window) < 4:
            raise ValueError("Template size must be larger than 1x1, if df=True.")
        mean = correlate1d(reference, hamming_2, mode="constant", axis=-2)
        mean = correlate1d(mean, hamming_1, mode="constant", axis=-1)
        mean_2 = xp.square(mean)

        mean_tw = swv(mean, template_window, axis=(-2, -1))
        mean_tw = mean_tw * hamming2d
        mean_tw = mean_tw.transpose(
            tuple(range(mean_tw.ndim - 5)) + (-4, -3, -5, -2, -1),
        )
        mean_tw = mean_tw.reshape(mean_tw.shape[:-3] + (-1,))

        # TODO(nin17): check this is correct
        # l2 = sum(mean**2) so K*mean**2
        l2 = correlate1d(mean_2, hamming_2, mode="constant", axis=-2)
        # !!! ok i do sum here, check check check
        l2 = correlate1d(l2, hamming_1, mode="constant", axis=-1).sum(axis=-3)

        l4 = xp.einsum("...k,...klm ->...lm", mean_tw[..., n0:-n0, n1:-n1, :], s_tw_sw)

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
            -l1_sw[..., m0 : -m0 or None, m1 : -m1 or None, :, :]
            - xp.square(beta) * l2[..., None, None]
            - xp.square(alpha) * l3[..., None, None]
            + 2.0 * beta * l4
            + 2.0 * alpha * l5
            - 2.0 * alpha * beta * l6[..., None, None]
        )
        _loss = loss.reshape(loss.shape[:-2] + (-1,))
        loss_max = _loss.argmax(axis=-1)
        _alpha = xp.take_along_axis(
            alpha.reshape(alpha.shape[:-2] + (-1,)), loss_max[..., None], axis=-1,
        ).squeeze(-1)
        _beta = xp.take_along_axis(
            beta.reshape(beta.shape[:-2] + (-1,)), loss_max[..., None], axis=-1,
        ).squeeze(-1)

        transmission = _alpha + _beta
        dark_field = _alpha / transmission
    else:
        # Negative of the loss: to use the maximum finding find_displacement
        loss = (
            l5_2 / l3[..., None, None]
            - l1_sw[..., m0 : -m0 or None, m1 : -m1 or None, :, :]
        ).real
        _loss = loss.reshape(loss.shape[:-2] + (-1,))
        loss_max = _loss.argmax(axis=-1)

        transmission = (
            xp.take_along_axis(
                l5.reshape(l5.shape[:-2] + (-1,)), loss_max[..., None], axis=-1,
            ).squeeze(-1)
            / l3
        )
        dark_field = None

    similarity_padded = xp.pad(
        loss, ((0, 0),) * (loss.ndim - 2) + ((1, 1), (1, 1)), PAD_MODE,
    )
    displacement = find_displacement(similarity_padded)
    return displacement + (transmission, dark_field)
