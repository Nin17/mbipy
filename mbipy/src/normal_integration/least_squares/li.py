"""
Li, G., Li, Y., Liu, K., Ma, X. & Wang, H. Improving wavefront reconstruction
accuracy by using integration equations with higher-order truncation errors in the
Southwell geometry. J. Opt. Soc. Am. A 30, 1448–1459. https://opg.optica.org/josaa/abstract.cfm?URI=josaa-30-7-1448 (July 2013).
"""

from __future__ import annotations

import functools

import numpy as np


def create_li_vectors(xp):
    def li_vectors(*, gx, gy):

        assert gx.shape == gy.shape
        assert gx.ndim >= 2

        sy, sx = gx.shape[-2:]

        newshape = gx.shape[:-2] + (-1,)

        Sx = (
            13
            / 24
            * (
                gx[..., 1:-2]
                - 1 / 13 * gx[..., :-3]
                + gx[..., 2:-1]
                - 1 / 13 * gx[..., 3:]
            )
        ).reshape(newshape)

        Sy = (
            13
            / 24
            * (
                gy[..., 1:-2, :]
                - 1 / 13 * gy[..., :-3, :]
                + gy[..., 2:-1, :]
                - 1 / 13 * gy[..., 3:, :]
            )
        ).reshape(newshape)

        S_Sxf = (
            (gx[..., (0, sx - 3)] + 4 * gx[..., (1, sx - 2)] + gx[..., (2, sx - 1)]) / 3
        ).reshape(newshape, order="F")

        S_Syf = (
            (
                gy[..., (0, sy - 3), :]
                + 4 * gy[..., (1, sy - 2), :]
                + gy[..., (2, sy - 1), :]
            )
            / 3
        ).reshape(newshape)

        return xp.concatenate((Sx, Sy, S_Sxf, S_Syf), axis=-1)

    return li_vectors


def create_li_matrix(xp, sparse):
    @functools.lru_cache()
    def li_matrix(
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
        # TODO variable names
        # TODO speed this up
        I, J = shape
        N = I * J
        # For each row, the elements 0,j-1 and j-1 are not considered, because close to the boundary
        rows = xp.arange(I * (J - 3), dtype=idtype)
        columns = xp.arange(N, dtype=idtype)
        to_delete = xp.concatenate(
            (
                xp.arange(0, N, J),
                xp.arange(J - 1, N, J),
                xp.arange(J - 2, N, J),
            )
        )
        columns = xp.delete(columns, to_delete)

        data1 = xp.ones_like(rows, dtype=fdtype)

        rows2 = xp.arange(I * (J - 3), I * (J - 3) + J * (I - 3), dtype=idtype)
        columns2 = xp.arange(J, J * (I - 2), dtype=idtype)
        data2 = xp.ones_like(rows2, dtype=fdtype)

        # We implement additional boundary conditions: Simpson equations
        off1 = I * (J - 3) + J * (I - 3)
        rows4 = xp.arange(I, dtype=idtype)
        data4 = xp.ones_like(rows4, dtype=fdtype)

        # Simpson conditions y direction
        off2 = I * (J - 1) + J * (I - 3)
        columns3 = xp.arange(J, dtype=idtype)
        rows3 = columns3 + off2
        data3 = xp.ones_like(rows3, dtype=fdtype)

        # TODO this without concatenation
        dataage = xp.concatenate(
            (
                -data1,
                data1,
                -data2,
                data2,
                -data4,
                data4,
                -data4,
                data4,
                -data3,
                data3,
                -data3,
                data3,
            )
        )

        rowage = xp.concatenate(
            (
                rows,
                rows,
                rows2,
                rows2,
                off1 + rows4,
                off1 + rows4,
                off1 + I + rows4,
                off1 + I + rows4,
                rows3,
                rows3,
                rows3 + J,
                rows3 + J,
            )
        )
        columnage = xp.concatenate(
            (
                columns,
                columns + 1,
                columns2,
                columns2 + J,
                rows4 * J,
                rows4 * J + 2,
                (rows4 + 1) * J - 3,
                (rows4 + 1) * J - 1,
                columns3,
                columns3 + 2 * J,
                columns3 + J * (I - 3),
                columns3 + J * (I - 1),
            )
        )
        matrix = sparse.coo_matrix((dataage, (rowage, columnage)))
        if normal:
            return matrix.T @ matrix, matrix.T
        return matrix

    return li_matrix


def create_li(li_matrix, li_vectors, sparse):
    def li(*, gx, gy, normal=False, **kwargs):
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
        assert gx.ndim == 2
        assert gx.dtype == gy.dtype
        assert gx.dtype.kind == "f"

        vectors = li_vectors(gx=gx, gy=gy)
        if not normal:
            matrix = li_matrix(gx.shape) # TODO(nin17): idtype, fdtype, normal
        else:
            a_ta, a_t = li_matrix(gx.shape, normal=normal)
        # ??? is matrices symmetric
        # ??? sparse.linalg.lsqr
        if not normal:
            return sparse.linalg.spsolve(
                matrix.T @ matrix, matrix.T @ vectors, **kwargs
            ).reshape(gx.shape) # TODO(nin17): broadcasted result shape
        else:
            return sparse.linalg.spsolve(a_ta, a_t @ vectors, **kwargs).reshape(gx.shape)

    return li
