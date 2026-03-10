# Copyright (c) 2025-2026 Hongyi Guan
# This file is part of CuPyMag
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import numpy as np
from numba import njit

from cupymag.core.parameters import precision
from cupymag.utils.precision_select import get_float_type
float_cp = get_float_type(precision, backend="cupy")


def shape_functions(r, s, t):
    """
    Return the 8 standard shape function values at parametric coordinates
    (r, s, t) on GPU.

    Parameters
    ----------
    r : scalar
        First parametric coordinate in [-1, 1].
    s : array_like or scalar
        Second parametric coordinate in [-1, 1].
    t : array_like or scalar
        Third parametric coordinate in [-1, 1].

    Returns
    -------
    cupy.ndarray
        An (8, ...) array of shape function values N_i(r, s, t) where
        the shape depends on the input array dimensions.
    """
    r = cp.asarray(r)
    s = cp.asarray(s)
    t = cp.asarray(t)

    N = cp.empty((8,) + r.shape, dtype=float_cp)

    N[0] = (1 - r)*(1 - s)*(1 - t)/8.0
    N[1] = (1 + r)*(1 - s)*(1 - t)/8.0
    N[2] = (1 + r)*(1 + s)*(1 - t)/8.0
    N[3] = (1 - r)*(1 + s)*(1 - t)/8.0
    N[4] = (1 - r)*(1 - s)*(1 + t)/8.0
    N[5] = (1 + r)*(1 - s)*(1 + t)/8.0
    N[6] = (1 + r)*(1 + s)*(1 + t)/8.0
    N[7] = (1 - r)*(1 + s)*(1 + t)/8.0

    return N

@njit
def shape_functions_cpu(r, s, t):
    """
    Return the 8 standard shape function values at parametric coordinates
    (r, s, t) on CPU.

    Parameters
    ----------
    r : scalar
        First parametric coordinate in [-1, 1].
    s : array_like or scalar
        Second parametric coordinate in [-1, 1].
    t : array_like or scalar
        Third parametric coordinate in [-1, 1].

    Returns
    -------
    numpy.ndarray
        An (8, ...) array of shape function values N_i(r, s, t) where
        the shape depends on the input array dimensions.
    """
    r = np.asarray(r)
    s = np.asarray(s)
    t = np.asarray(t)

    N = np.empty((8,) + r.shape, dtype=np.float64)

    N[0] = (1 - r)*(1 - s)*(1 - t)/8.0
    N[1] = (1 + r)*(1 - s)*(1 - t)/8.0
    N[2] = (1 + r)*(1 + s)*(1 - t)/8.0
    N[3] = (1 - r)*(1 + s)*(1 - t)/8.0
    N[4] = (1 - r)*(1 - s)*(1 + t)/8.0
    N[5] = (1 + r)*(1 - s)*(1 + t)/8.0
    N[6] = (1 + r)*(1 + s)*(1 + t)/8.0
    N[7] = (1 - r)*(1 + s)*(1 + t)/8.0

    return N


def shape_function_gradients(r, s, t):
    """
    Return the gradient of the 8 shape functions with respect to parametric
    coordinates on GPU.

    Parameters
    ----------
    r : array_like or scalar
        First parametric coordinate in [-1, 1].
    s : array_like or scalar
        Second parametric coordinate in [-1, 1].
    t : array_like or scalar
        Third parametric coordinate in [-1, 1].

    Returns
    -------
    cupy.ndarray
        An (8, 3, ...) array of shape function derivatives.
    """
    r = cp.asarray(r)
    s = cp.asarray(s)
    t = cp.asarray(t)

    dN = cp.empty((8, 3) + r.shape, dtype=float_cp)

    dN[0, 0] = -(1 - s)*(1 - t)/8.0
    dN[0, 1] = -(1 - r)*(1 - t)/8.0
    dN[0, 2] = -(1 - r)*(1 - s)/8.0

    dN[1, 0] = +(1 - s)*(1 - t)/8.0
    dN[1, 1] = -(1 + r)*(1 - t)/8.0
    dN[1, 2] = -(1 + r)*(1 - s)/8.0

    dN[2, 0] = +(1 + s)*(1 - t)/8.0
    dN[2, 1] = +(1 + r)*(1 - t)/8.0
    dN[2, 2] = -(1 + r)*(1 + s)/8.0

    dN[3, 0] = -(1 + s)*(1 - t)/8.0
    dN[3, 1] = +(1 - r)*(1 - t)/8.0
    dN[3, 2] = -(1 - r)*(1 + s)/8.0

    dN[4, 0] = -(1 - s)*(1 + t)/8.0
    dN[4, 1] = -(1 - r)*(1 + t)/8.0
    dN[4, 2] = +(1 - r)*(1 - s)/8.0

    dN[5, 0] = +(1 - s)*(1 + t)/8.0
    dN[5, 1] = -(1 + r)*(1 + t)/8.0
    dN[5, 2] = +(1 + r)*(1 - s)/8.0

    dN[6, 0] = +(1 + s)*(1 + t)/8.0
    dN[6, 1] = +(1 + r)*(1 + t)/8.0
    dN[6, 2] = +(1 + r)*(1 + s)/8.0

    dN[7, 0] = -(1 + s)*(1 + t)/8.0
    dN[7, 1] = +(1 - r)*(1 + t)/8.0
    dN[7, 2] = +(1 - r)*(1 + s)/8.0

    return dN

@njit
def shape_function_gradients_cpu(r, s, t):
    """
    Return the gradient of the 8 shape functions with respect to parametric
    coordinates on CPU.

    Parameters
    ----------
    r : array_like or scalar
        First parametric coordinate in [-1, 1].
    s : array_like or scalar
        Second parametric coordinate in [-1, 1].
    t : array_like or scalar
        Third parametric coordinate in [-1, 1].

    Returns
    -------
    numpy.ndarray
        An (8, 3, ...) array of shape function derivatives.
    """
    dN = np.empty((8, 3), dtype=np.float64)

    dN[0, 0] = -(1 - s)*(1 - t)/8.0
    dN[0, 1] = -(1 - r)*(1 - t)/8.0
    dN[0, 2] = -(1 - r)*(1 - s)/8.0

    dN[1, 0] = +(1 - s)*(1 - t)/8.0
    dN[1, 1] = -(1 + r)*(1 - t)/8.0
    dN[1, 2] = -(1 + r)*(1 - s)/8.0

    dN[2, 0] = +(1 + s)*(1 - t)/8.0
    dN[2, 1] = +(1 + r)*(1 - t)/8.0
    dN[2, 2] = -(1 + r)*(1 + s)/8.0

    dN[3, 0] = -(1 + s)*(1 - t)/8.0
    dN[3, 1] = +(1 - r)*(1 - t)/8.0
    dN[3, 2] = -(1 - r)*(1 + s)/8.0

    dN[4, 0] = -(1 - s)*(1 + t)/8.0
    dN[4, 1] = -(1 - r)*(1 + t)/8.0
    dN[4, 2] = +(1 - r)*(1 - s)/8.0

    dN[5, 0] = +(1 - s)*(1 + t)/8.0
    dN[5, 1] = -(1 + r)*(1 + t)/8.0
    dN[5, 2] = +(1 + r)*(1 - s)/8.0

    dN[6, 0] = +(1 + s)*(1 + t)/8.0
    dN[6, 1] = +(1 + r)*(1 + t)/8.0
    dN[6, 2] = +(1 + r)*(1 + s)/8.0

    dN[7, 0] = -(1 + s)*(1 + t)/8.0
    dN[7, 1] = +(1 - r)*(1 + t)/8.0
    dN[7, 2] = +(1 - r)*(1 + s)/8.0

    return dN

@njit
def element_jacobian(xc, yc, zc, n):
    """
    Compute the Jacobian matrix for the transformation from reference to physical space.

    Parameters
    ----------
    xc : array_like
        Array of length 8 containing x-coordinates of element nodes.
    yc : array_like
        Array of length 8 containing y-coordinates of element nodes.
    zc : array_like
        Array of length 8 containing z-coordinates of element nodes.
    n : int
        Gauss point index.

    Returns
    -------
    numpy.ndarray
        A (3, 3) array representing the Jacobian matrix.
    """
    dN = get_dN(n)
    J = np.zeros((3, 3), dtype=np.float64)

    for i in range(8):
        J[0, 0] += xc[i] * dN[i, 0]
        J[0, 1] += xc[i] * dN[i, 1]
        J[0, 2] += xc[i] * dN[i, 2]
        J[1, 0] += yc[i] * dN[i, 0]
        J[1, 1] += yc[i] * dN[i, 1]
        J[1, 2] += yc[i] * dN[i, 2]
        J[2, 0] += zc[i] * dN[i, 0]
        J[2, 1] += zc[i] * dN[i, 1]
        J[2, 2] += zc[i] * dN[i, 2]

    return J

@njit
def gauss_quadrature():
    """
    Return standard 2×2×2 Gauss-Legendre quadrature points and weights for hexahedral elements.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:
        - points: (8, 3) array of integration points in reference space
        - weights: (8,) array of corresponding quadrature weights

    Notes
    -----
    For linear 8-node hexahedral elements, this quadrature rule provides exact integration
    of polynomial integrands up to degree 1 in each parametric direction.
    """

    _gauss_pts_1d = [-0.57735026919, +0.57735026919]
    _gauss_wts_1d = [1.0, 1.0]

    GAUSS_POINTS = []
    GAUSS_WEIGHTS = []

    for rx, wx in zip(_gauss_pts_1d, _gauss_wts_1d):
        for ry, wy in zip(_gauss_pts_1d, _gauss_wts_1d):
            for rz, wz in zip(_gauss_pts_1d, _gauss_wts_1d):
                GAUSS_POINTS.append((rx, ry, rz))
                GAUSS_WEIGHTS.append(wx * wy * wz)

    GAUSS_POINTS = np.array(GAUSS_POINTS, dtype=np.float64)  # shape (8, 3)
    GAUSS_WEIGHTS = np.array(GAUSS_WEIGHTS, dtype=np.float64)

    return GAUSS_POINTS, GAUSS_WEIGHTS


@njit
def corners_local():
    return np.array([
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, +1.0],
            [-1.0, +1.0, +1.0],
        ], dtype=np.float64)

_dN_data = np.array([
    [[-0.311004233964, -0.311004233964, -0.311004233964],
     [ 0.311004233964, -0.083333333333, -0.083333333333],
     [ 0.083333333333,  0.083333333333, -0.022329099369],
     [-0.083333333333,  0.311004233964, -0.083333333333],
     [-0.083333333333, -0.083333333333,  0.311004233964],
     [ 0.083333333333, -0.022329099369,  0.083333333333],
     [ 0.022329099369,  0.022329099369,  0.022329099369],
     [-0.022329099369,  0.083333333333,  0.083333333333]],

    [[-0.083333333333, -0.083333333333, -0.311004233964],
     [ 0.083333333333, -0.022329099369, -0.083333333333],
     [ 0.022329099369,  0.022329099369, -0.022329099369],
     [-0.022329099369,  0.083333333333, -0.083333333333],
     [-0.311004233964, -0.311004233964,  0.311004233964],
     [ 0.311004233964, -0.083333333333,  0.083333333333],
     [ 0.083333333333,  0.083333333333,  0.022329099369],
     [-0.083333333333,  0.311004233964,  0.083333333333]],

    [[-0.083333333333, -0.311004233964, -0.083333333333],
     [ 0.083333333333, -0.083333333333, -0.022329099369],
     [ 0.311004233964,  0.083333333333, -0.083333333333],
     [-0.311004233964,  0.311004233964, -0.311004233964],
     [-0.022329099369, -0.083333333333,  0.083333333333],
     [ 0.022329099369, -0.022329099369,  0.022329099369],
     [ 0.083333333333,  0.022329099369,  0.083333333333],
     [-0.083333333333,  0.083333333333,  0.311004233964]],

    [[-0.022329099369, -0.083333333333, -0.083333333333],
     [ 0.022329099369, -0.022329099369, -0.022329099369],
     [ 0.083333333333,  0.022329099369, -0.083333333333],
     [-0.083333333333,  0.083333333333, -0.311004233964],
     [-0.083333333333, -0.311004233964,  0.083333333333],
     [ 0.083333333333, -0.083333333333,  0.022329099369],
     [ 0.311004233964,  0.083333333333,  0.083333333333],
     [-0.311004233964,  0.311004233964,  0.311004233964]],

    [[-0.311004233964, -0.083333333333, -0.083333333333],
     [ 0.311004233964, -0.311004233964, -0.311004233964],
     [ 0.083333333333,  0.311004233964, -0.083333333333],
     [-0.083333333333,  0.083333333333, -0.022329099369],
     [-0.083333333333, -0.022329099369,  0.083333333333],
     [ 0.083333333333, -0.083333333333,  0.311004233964],
     [ 0.022329099369,  0.083333333333,  0.083333333333],
     [-0.022329099369,  0.022329099369,  0.022329099369]],

    [[-0.083333333333, -0.022329099369, -0.083333333333],
     [ 0.083333333333, -0.083333333333, -0.311004233964],
     [ 0.022329099369,  0.083333333333, -0.083333333333],
     [-0.022329099369,  0.022329099369, -0.022329099369],
     [-0.311004233964, -0.083333333333,  0.083333333333],
     [ 0.311004233964, -0.311004233964,  0.311004233964],
     [ 0.083333333333,  0.311004233964,  0.083333333333],
     [-0.083333333333,  0.083333333333,  0.022329099369]],

    [[-0.083333333333, -0.083333333333, -0.022329099369],
     [ 0.083333333333, -0.311004233964, -0.083333333333],
     [ 0.311004233964,  0.311004233964, -0.311004233964],
     [-0.311004233964,  0.083333333333, -0.083333333333],
     [-0.022329099369, -0.022329099369,  0.022329099369],
     [ 0.022329099369, -0.083333333333,  0.083333333333],
     [ 0.083333333333,  0.083333333333,  0.311004233964],
     [-0.083333333333,  0.022329099369,  0.083333333333]],

    [[-0.022329099369, -0.022329099369, -0.022329099369],
     [ 0.022329099369, -0.083333333333, -0.083333333333],
     [ 0.083333333333,  0.083333333333, -0.311004233964],
     [-0.083333333333,  0.022329099369, -0.083333333333],
     [-0.083333333333, -0.083333333333,  0.022329099369],
     [ 0.083333333333, -0.311004233964,  0.083333333333],
     [ 0.311004233964,  0.311004233964,  0.311004233964],
     [-0.311004233964,  0.083333333333,  0.083333333333]]
])

_N_data = np.array([
    [0.490562612163, 0.131445855766, 0.035220810901, 0.131445855766,
     0.131445855766, 0.035220810901, 0.009437387838, 0.035220810901],
    [0.131445855766, 0.035220810901, 0.009437387838, 0.035220810901,
     0.490562612163, 0.131445855766, 0.035220810901, 0.131445855766],
    [0.131445855766, 0.035220810901, 0.131445855766, 0.490562612163,
     0.035220810901, 0.009437387838, 0.035220810901, 0.131445855766],
    [0.035220810901, 0.009437387838, 0.035220810901, 0.131445855766,
     0.131445855766, 0.035220810901, 0.131445855766, 0.490562612163],
    [0.131445855766, 0.490562612163, 0.131445855766, 0.035220810901,
     0.035220810901, 0.131445855766, 0.035220810901, 0.009437387838],
    [0.035220810901, 0.131445855766, 0.035220810901, 0.009437387838,
     0.131445855766, 0.490562612163, 0.131445855766, 0.035220810901],
    [0.035220810901, 0.131445855766, 0.490562612163, 0.131445855766,
     0.009437387838, 0.035220810901, 0.131445855766, 0.035220810901],
    [0.009437387838, 0.035220810901, 0.131445855766, 0.035220810901,
     0.035220810901, 0.131445855766, 0.490562612163, 0.131445855766]
])

@njit
def get_dN(n):
    """
    Return the 8×3 array of shape‐function derivatives at the n-th Gauss point.

    Parameters
    ----------
    n : int
        Gauss‐point index (0 ≤ n ≤ 7).

    Returns
    -------
    numpy.ndarray
        An (8,3) array of ∂N_i/∂ξ_j at Gauss point n.
    """
    return _dN_data[n]

@njit
def get_N(n):
    """
    Return the size 8 array of shape‐function at the n-th Gauss point.

    Parameters
    ----------
    n : int
        Gauss‐point index (0 ≤ n ≤ 7).

    Returns
    -------
    numpy.ndarray
        An (8,) array of N_i at Gauss point n.
    """
    return _N_data[n]



