"""
"""


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
            alpha.shape + (1,) * (matrices.ndim - alpha.ndim)
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
    def normal_stack(matrices, vectors, *, alpha):
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
        matrices = xp.asarray(matrices)
        vectors = xp.asarray(vectors)
        alpha = xp.asarray(alpha)

        alpha = alpha.reshape(alpha.shape + (1,) * (matrices.ndim - alpha.ndim))
        ata = xp.einsum("...ij, ...ik", matrices, matrices, optimize="optimal")
        atai = ata + xp.identity(matrices.shape[-1]) * alpha
        atb = xp.einsum("...ij, ...i", matrices, vectors, optimize="optimal")

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

        return xp.linalg.solve(matrices, vectors)

    return normal
