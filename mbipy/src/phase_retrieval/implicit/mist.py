"""_summary_"""

__all__ = ("Mist20", "Mist23", "mist20", "mist23")

from array_api_compat import is_torch_namespace

from mbipy.src.utils import array_namespace

from .utils import laplace


def _mist20_matrices(reference):
    xp = array_namespace(reference)
    if not reference.ndim >= 3:
        msg = f"reference must have at least 3 dimensions. {reference.ndim}."
        raise ValueError(msg)
    laplacian = laplace(reference)
    matrices = xp.stack((reference, -laplacian), axis=-1)
    order = (*tuple(range(matrices.ndim - 4)), -3, -2, -4, -1)
    if is_torch_namespace(xp):
        return matrices.permute(*order)
    return matrices.transpose(order)


def _mist20_vectors(sample, reference):
    xp = array_namespace(sample, reference)
    if not sample.ndim >= 3:
        msg = f"sample must have at least 3 dimensions. {sample.ndim}."
        raise ValueError(msg)
    r_s = reference - sample
    if is_torch_namespace(xp):
        return r_s.permute(*range(r_s.ndim - 3), -2, -1, -3)
    return r_s.transpose(tuple(range(r_s.ndim - 3)) + (-2, -1, -3))


def mist20(
    sample,
    reference,
    alpha=0.0,
    search_window=None,
    start=None,
    stop=None,
    step=None,
):
    xp = array_namespace(sample, reference)
    matrices = _mist20_matrices(reference)
    vectors = _mist20_vectors(sample, reference)

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
    ata = (matrices.transpose(transpose) @ matrices) + alpha * xp.eye(2)
    atb = matrices.transpose(transpose) @ vectors[..., None]
    return xp.linalg.solve(ata, atb).squeeze(-1)


def mist23(sample, reference, alpha=0.0):
    raise NotImplementedError("Not implemented yet.")


class Mist20:
    def __init__(self, reference, alpha=0.0):
        raise NotImplementedError("Not implemented yet.")


class Mist23:
    def __init__(self, reference, alpha=0.0):
        raise NotImplementedError("Not implemented yet.")
