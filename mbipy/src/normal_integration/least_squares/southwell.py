"""
Southwell, W. Wave-front estimation from wave-front slope measurements. J. Opt. Soc. Am. 70, 998–1006. https://opg.optica.org/abstract.cfm?URI=josa-70-8-998 (Aug. 1980).
"""

import functools

import numpy as np


def create_southwell_vectors(xp):
    def southwell_vectors(*, gx, gy):

        assert gx.shape == gy.shape
        assert gx.ndim >= 2

        new_shape = gx.shape[:-2] + (-1,)

        sxf = (0.5 * (gx[..., 1:] + gx[..., :-1])).reshape(new_shape)
        syf = (0.5 * (gy[..., :-1, :] + gy[..., 1:, :])).reshape(new_shape)

        return xp.concatenate((sxf, syf), axis=-1)

    return southwell_vectors


def create_southwell_matrix(xp, sparse):
    @functools.lru_cache()
    def southwell_matrix(
        shape: tuple[int, int], normal=False, idtype=np.int64, fdtype=np.float64
    ):
        """_summary_

        Parameters
        ----------
        shape : tuple[int, int]
            _description_

        Returns
        -------
        Array
            _description_
        """
        # TODO documentation
        # Actually a sparse array - depending on xp

        i, j = shape
        n = i * j

        rows1 = xp.arange(i * (j - 1), dtype=idtype)
        columns1 = xp.arange(n, dtype=idtype)
        if hasattr(xp, "lib"):
            strides = columns1.strides[0]
            columns1 = xp.lib.stride_tricks.as_strided(
                columns1, shape=(i, j - 1), strides=(strides * j, strides)
            ).ravel()
        else:
            columns1 = xp.delete(columns1, xp.arange(j - 1, n, j))

        columns2 = xp.arange(j * (i - 1), dtype=idtype)
        rows2 = columns2 + i * (j - 1)

        data1 = xp.ones_like(rows1, dtype=fdtype)
        data2 = xp.ones_like(rows2, dtype=fdtype)

        # TODO variable names
        # TODO can definitely do some of this without the concatenation
        rowage = xp.concatenate((rows1, rows1, rows2, rows2))
        columnage = xp.concatenate((columns1, columns1 + 1, columns2, columns2 + j))
        dataage = xp.concatenate((-data1, data1, -data2, data2))
        matrix = sparse.coo_matrix((dataage, (rowage, columnage)))  # .tocsr()
        if normal:
            return matrix.T @ matrix, matrix.T
        return matrix

    return southwell_matrix


def create_southwell(southwell_matrix, southwell_vectors, sparse):
    def southwell(*, gx, gy, normal=False, **kwargs):
        """_summary_

        Parameters
        ----------
        gx : ArrayLike
            _description_
        gy : ArrayLike
            _description_

        Returns
        -------
        Array
            _description_
        """
        assert gx.shape == gy.shape
        assert gx.ndim == 2  # Don't think the solvers will do stacks of matrices

        vectors = southwell_vectors(gx=gx, gy=gy)

        if not normal:
            matrix = southwell_matrix(gx.shape)
        else:
            a_ta, a_t = southwell_matrix(gx.shape, normal=normal)

        if not normal:
            return sparse.linalg.spsolve(
                matrix.T @ matrix, matrix.T @ vectors, **kwargs
            ).reshape(gx.shape)
        return sparse.linalg.spsolve(a_ta, a_t @ vectors, **kwargs).reshape(gx.shape)

        return sparse.linalg.lsqr(matrix, vectors, **kwargs)
        # !!! really slow dunno why they suggest that
        # return xp.linalg.lstsq(matrices.toarray(), vectors, **kwargs)
        # !!! really slow dunno why they suggest that
        # return xp.linalg.lstsq(matrices.toarray(), vectors, **kwargs)

    return southwell
