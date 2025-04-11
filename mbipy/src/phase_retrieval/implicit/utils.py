""" """

from array_api_compat import is_jax_namespace

from mbipy.src.utils import array_namespace


def laplace32(image_stack):
    # TODO(nin17): use iadd & use work arrays to compute intermediate terms
    xp = array_namespace(image_stack)
    output = xp.zeros_like(image_stack)

    if is_jax_namespace(xp):
        output = output.at[..., 1:-1, :, :].add(
            image_stack[..., :-2, :, :]
            + image_stack[..., 2:, :, :]
            - (2 * image_stack[..., 1:-1, :, :]),
        )
        output = output.at[..., 1:-1, :].add(
            image_stack[..., :-2, :]
            + image_stack[..., 2:, :]
            - (2 * image_stack[..., 1:-1, :]),
        )
        output = output.at[..., 0, :, :].add(
            image_stack[..., 0, :, :]
            + image_stack[..., 1, :, :]
            - 2 * image_stack[..., 0, :, :],
        )
        output = output.at[..., -1, :, :].add(
            image_stack[..., -1, :, :]
            + image_stack[..., -2, :, :]
            - 2 * image_stack[..., -1, :, :],
        )
        output = output.at[..., 0, :].add(
            image_stack[..., 0, :]
            + image_stack[..., 1, :]
            - 2 * image_stack[..., 0, :],
        )
        output = output.at[..., -1, :].add(
            image_stack[..., -1, :]
            + image_stack[..., -2, :]
            - 2 * image_stack[..., -1, :],
        )
    else:
        output[..., 1:-1, :, :] += (
            image_stack[..., :-2, :, :]
            + image_stack[..., 2:, :, :]
            - (2 * image_stack[..., 1:-1, :, :])
        )
        output[..., 1:-1, :] += (
            image_stack[..., :-2, :]
            + image_stack[..., 2:, :]
            - 2 * image_stack[..., 1:-1, :]
        )
        output[..., 0, :, :] += (
            image_stack[..., 0, :, :]
            + image_stack[..., 1, :, :]
            - 2 * image_stack[..., 0, :, :]
        )
        output[..., -1, :, :] += (
            image_stack[..., -1, :, :]
            + image_stack[..., -2, :, :]
            - 2 * image_stack[..., -1, :, :]
        )
        output[..., 0, :] += (
            image_stack[..., 0, :] + image_stack[..., 1, :] - 2 * image_stack[..., 0, :]
        )
        output[..., -1, :] += (
            image_stack[..., -1, :]
            + image_stack[..., -2, :]
            - 2 * image_stack[..., -1, :]
        )
    return output


def is_invertible(matrix_stack):
    xp = array_namespace(matrix_stack)
    if matrix_stack.shape[-2] != matrix_stack.shape[-1]:
        msg = "Matrices must be square"
        raise ValueError(msg)
    return xp.linalg.matrix_rank(matrix_stack) == matrix_stack.shape[-1]

# ------------------------------------ not needed ------------------------------------ #

def create_tikhonov_stack(xp):
    def tikhonov_stack(matrices, vectors, *, alpha):
        """
        Concatenate the input matrices and vectors with the regularisation terms:
        α¹ᐟ²Iₙ and 0ₙ respectively. As outlined in:
        https://math.stackexchange.com/a/299508.

        min ||(b) - (  A )x||
            ||(0)   (α¹ᐟ²I) ||_2

        Parameters
        ----------
        matrices : ArrayLike
            (..., M, N)
        vectors : ArrayLike
            (..., M)
        alpha : ArrayLike
            (...)

        Returns
        -------
        tuple[Array, Array]
            ((..., M + N, N), (..., M + N))
            Matrices concatenated with the regularisation matrix and vectors
            concatenated with zeros respectively.
        """
        matrices = xp.asarray(matrices)
        vectors = xp.asarray(vectors)
        alpha = xp.asarray(alpha)

        assert matrices.shape[:-1] == vectors.shape
        assert matrices.ndim == vectors.ndim + 1
        n = matrices.shape[-1]
        assert alpha.ndim <= matrices.ndim - 2

        alpha_05 = xp.sqrt(alpha).reshape(
            alpha.shape + (1,) * (matrices.ndim - alpha.ndim),
        )
        identity = xp.identity(n).reshape((1,) * (matrices.ndim - 2) + (n, n))

        tikhonov = xp.broadcast_to(
            identity * alpha_05,
            matrices.shape[:-2] + (n, n),
        )
        _matrices = xp.concatenate((matrices, tikhonov), axis=-2)
        _vectors = xp.concatenate(
            (vectors, xp.zeros(vectors.shape[:-1] + (n,))),
            axis=-1,
        )
        return _matrices, _vectors

    return tikhonov_stack


def create_normal_stack(xp):
    def normal_stack(matrices, vectors, *, alpha, use_einsum=None):
        """
        Form the stack of normal equations given stacks of matrices, vectors and
        regularisation factors.

        Parameters
        ----------
        matrices : ArrayLike
            (..., M, N)
        vectors : ArrayLike
            (..., M)
        alpha : ArrayLike
            (...)

        Returns
        -------
        tuple[Array, Array]
            ((..., N, N), (..., N))
            Stacks of the LHS and RHS of the normal equations respectively.
        """
        # einsum is faster with cupy, but slower with numpy
        if use_einsum is None:
            use_einsum = xp.__name__ != "numpy"

        matrices = xp.asarray(matrices)
        vectors = xp.asarray(vectors)
        alpha = xp.asarray(alpha)

        alpha = alpha.reshape(alpha.shape + (1,) * (matrices.ndim - alpha.ndim))
        # if use_einsum:
        ata = xp.einsum("...ij, ...ik", matrices.conj(), matrices, optimize="optimal")
        # else:
        #     ata = (
        #         matrices.transpose((tuple(range(matrices.ndim - 2)) + (-1, -2)))
        #         @ matrices
        #     )
        atai = ata + xp.identity(matrices.shape[-1]) * alpha
        # ??? nin17 Is there a faster way to do this too?
        # (matrices.transpose((tuple(range(matrices.ndim - 2)) + (-1, -2))) @ vectors[..., None]).squeeze(-1)
        if matrices.ndim == vectors.ndim + 1:
            atb = xp.einsum("...ij, ...i", matrices.conj(), vectors)
        elif matrices.ndim == vectors.ndim:
            atb = xp.einsum("...ij, ...ik->...jk", matrices, vectors)
        return atai, atb

    return normal_stack


def create_normal(xp):
    def normal(matrices, vectors):
        """
        Solve a stack of matrix equations (Ax = b).

        Parameters
        ----------
        matrices : ArrayLike
            (..., M, M)
            Stack of square matrices: A
        vectors : ArrayLike
            (..., M)
            Stack of vectors: b

        Returns
        -------
        Array
            (..., M)
            Stack of solutions: x
        """
        print(matrices.shape, vectors.shape)
        import time

        start = time.time()
        result = xp.linalg.solve(matrices, vectors)
        print(f"{time.time() - start=}")
        return result
        return xp.linalg.solve(matrices, vectors)

    return normal


def create_lstsq_solver(solvers, normal_stack, tikhonov_stack):
    def lstsq_solver(matrices, vectors, *, alpha=0.0, rcond=None, method=None):
        if method is None:
            method = "normal"
        if method not in solvers:
            raise ValueError(
                f"Invalid method: {method}. Valid methods: {solvers.keys()}",
            )

        if method in {"cholesky", "inv", "normal"}:
            matrices, vectors = normal_stack(matrices, vectors, alpha=alpha)

        elif method in {"lstsq", "pinv", "qr"}:
            matrices, vectors = tikhonov_stack(matrices, vectors, alpha=alpha)
            if method == "lstsq":
                return solvers[method](matrices, vectors, rcond=rcond)

        elif method in {"svd"}:
            return solvers[method](matrices, vectors, alpha=alpha, rcond=rcond)

        return solvers[method](matrices, vectors)

    return lstsq_solver


def create_implicit_tracking(xp, swv, solver):
    # TODO jax version
    def implicit_tracking(matrices, vectors, m, n, a=0, b=1, **kwargs):
        _m = 2 * m + 1
        _n = 2 * n + 1
        NEW = False
        if NEW:

            # !!! NEW
            matrices = matrices[..., m:-m, n:-n, :, :]
            vectors = swv(vectors, (_m, _n), axis=(-3, -2))
            print(vectors.shape)
            vectors = vectors[..., a::b, a::b]
            # TODO a and b in the window
            vectors = vectors.reshape(vectors.shape[:-2] + (-1,))

            result = solver(matrices, vectors, **kwargs)
            residuals = xp.einsum("...ij, ...jk->...ik", matrices, result) - vectors
            print(residuals.shape)
            minimum = (residuals * residuals.conj()).sum(axis=-2).argmin(axis=-1)
        else:
            # !!! OLD
            matrices = matrices[..., m:-m, n:-n, None, None, :, :]
            vectors = swv(vectors, (_m, _n), axis=(-3, -2)).transpose(
                tuple(range(vectors.ndim - 3)) + (-5, -4, -2, -1, -3),
            )
            vectors = vectors[..., a::b, a::b, :]
            result = solver(matrices, vectors, **kwargs)

            residuals = xp.einsum("...ij, ...j", matrices, result) - vectors

            minimum = (
                (residuals * residuals.conj())
                .sum(axis=-1)
                .reshape(residuals.shape[:-3] + (-1,))
                .argmin(axis=-1)
            )

        print(f"{minimum.shape=}")
        minima = xp.unravel_index(minimum, residuals.shape[-3:-1])
        if NEW:
            # NEW
            ndim = result.ndim - 3
            preceding = tuple(
                xp.arange(j).reshape((1,) * i + (-1,) + (1,) * (ndim - i))
                for i, j in enumerate(result.shape[:-2])
            )
        else:
            # OLD
            ndim = result.ndim - 4
            preceding = tuple(
                xp.arange(j).reshape((1,) * i + (-1,) + (1,) * (ndim - i))
                for i, j in enumerate(result.shape[:-3])
            )
        print(f"{result.shape=}")
        print([i.shape for i in preceding])

        if NEW:
            result_minimum = result[preceding + (slice(None), minimum)]
        else:
            result_minimum = result[preceding + minima]

        # TODO nin17: check this
        result_minimum[..., 1] += b * minima[0] + (a - m)
        result_minimum[..., 2] += b * minima[1] + (a - n)

        return result_minimum

    return implicit_tracking


def create_laplace(xp):
    if xp.__name__ == "jax.numpy":

        def laplace(image_stack):
            output = xp.zeros_like(image_stack)
            output = output.at[..., 1:-1, :].add(
                image_stack[..., :-2, :]
                + image_stack[..., 2:, :]
                - (2 * image_stack[..., 1:-1, :]),
            )
            output = output.at[..., 1:-1].add(
                image_stack[..., :-2]
                + image_stack[..., 2:]
                - (2 * image_stack[..., 1:-1]),
            )
            output = output.at[..., 0, :].add(
                image_stack[..., 0, :]
                + image_stack[..., 1, :]
                - 2 * image_stack[..., 0, :],
            )
            output = output.at[..., -1, :].add(
                image_stack[..., -1, :]
                + image_stack[..., -2, :]
                - 2 * image_stack[..., -1, :],
            )
            output = output.at[..., 0].add(
                image_stack[..., 0] + image_stack[..., 1] - 2 * image_stack[..., 0],
            )
            output = output.at[..., -1].add(
                image_stack[..., -1] + image_stack[..., -2] - 2 * image_stack[..., -1],
            )
            return output

    else:

        def laplace(image_stack):
            output = xp.zeros_like(image_stack)
            output[..., 1:-1, :] += (
                image_stack[..., :-2, :]
                + image_stack[..., 2:, :]
                - (2 * image_stack[..., 1:-1, :])
            )
            output[..., 1:-1] += (
                image_stack[..., :-2]
                + image_stack[..., 2:]
                - 2 * image_stack[..., 1:-1]
            )
            output[..., 0, :] += (
                image_stack[..., 0, :]
                + image_stack[..., 1, :]
                - 2 * image_stack[..., 0, :]
            )
            output[..., -1, :] += (
                image_stack[..., -1, :]
                + image_stack[..., -2, :]
                - 2 * image_stack[..., -1, :]
            )
            output[..., 0] += (
                image_stack[..., 0] + image_stack[..., 1] - 2 * image_stack[..., 0]
            )
            output[..., -1] += (
                image_stack[..., -1] + image_stack[..., -2] - 2 * image_stack[..., -1]
            )
            return output

    return laplace


def create_isinvertible(xp):
    def is_invertible(matrix_stack):
        if matrix_stack.shape[-2] != matrix_stack.shape[-1]:
            raise ValueError("Matrices must be square")
        return xp.linalg.matrix_rank(matrix_stack) == matrix_stack.shape[-1]

def laplace(image_stack):
    # TODO(nin17): change to axes = (-3, -2)
    xp = array_namespace(image_stack)
    output = xp.zeros_like(image_stack)
    if is_jax_namespace(xp):
        output = output.at[..., 1:-1, :].add(
            image_stack[..., :-2, :]
            + image_stack[..., 2:, :]
            - (2 * image_stack[..., 1:-1, :]),
        )
        output = output.at[..., 1:-1].add(
            image_stack[..., :-2] + image_stack[..., 2:] - (2 * image_stack[..., 1:-1]),
        )
        output = output.at[..., 0, :].add(
            image_stack[..., 0, :]
            + image_stack[..., 1, :]
            - 2 * image_stack[..., 0, :],
        )
        output = output.at[..., -1, :].add(
            image_stack[..., -1, :]
            + image_stack[..., -2, :]
            - 2 * image_stack[..., -1, :],
        )
        output = output.at[..., 0].add(
            image_stack[..., 0] + image_stack[..., 1] - 2 * image_stack[..., 0],
        )
        output = output.at[..., -1].add(
            image_stack[..., -1] + image_stack[..., -2] - 2 * image_stack[..., -1],
        )
    else:
        output[..., 1:-1, :] += (
            image_stack[..., :-2, :]
            + image_stack[..., 2:, :]
            - (2 * image_stack[..., 1:-1, :])
        )
        output[..., 1:-1] += (
            image_stack[..., :-2] + image_stack[..., 2:] - 2 * image_stack[..., 1:-1]
        )
        output[..., 0, :] += (
            image_stack[..., 0, :] + image_stack[..., 1, :] - 2 * image_stack[..., 0, :]
        )
        output[..., -1, :] += (
            image_stack[..., -1, :]
            + image_stack[..., -2, :]
            - 2 * image_stack[..., -1, :]
        )
        output[..., 0] += (
            image_stack[..., 0] + image_stack[..., 1] - 2 * image_stack[..., 0]
        )
        output[..., -1] += (
            image_stack[..., -1] + image_stack[..., -2] - 2 * image_stack[..., -1]
        )
    return output
