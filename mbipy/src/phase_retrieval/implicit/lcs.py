"""_summary_
"""

__all__ = (
    "create_lcs_matrices",
    "create_lcs_vectors",
    "create_lcs",
    "create_lcs_df",
    "create_lcs_df_matrices",
)

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

        def lcs(reference, sample, weak_absorption=True, m=None, n=None, **kwargs):
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
