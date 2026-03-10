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
    Return the 4 standard shape function values at parametric coordinates
    (r, s, t) on GPU.

    Parameters
    ----------
    r : scalar
        First parametric coordinate in [0, 1].
    s : array_like or scalar
        Second parametric coordinate in [0, 1].
    t : array_like or scalar
        Third parametric coordinate in [0, 1].

    Returns
    -------
    cupy.ndarray
        An (4, ...) array of shape function values N_i(r, s, t) where
        the shape depends on the input array dimensions.
    """
    r = cp.asarray(r)
    s = cp.asarray(s)
    t = cp.asarray(t)

    N = cp.empty((4,) + r.shape, dtype=float_cp)

    N[0] = 1.0 - r - s - t
    N[1] = r
    N[2] = s
    N[3] = t

    return N

@njit
def shape_functions_cpu(r, s, t):
    """
    Return the 4 standard shape function values at parametric coordinates
    (r, s, t) on CPU.

    Parameters
    ----------
    r : scalar
        First parametric coordinate in [0, 1].
    s : array_like or scalar
        Second parametric coordinate in [0, 1].
    t : array_like or scalar
        Third parametric coordinate in [0, 1].

    Returns
    -------
    numpy.ndarray
        An (4, ...) array of shape function values N_i(r, s, t) where
        the shape depends on the input array dimensions.
    """
    r = np.asarray(r)
    s = np.asarray(s)
    t = np.asarray(t)

    N = np.empty((4,) + r.shape, dtype=np.float64)

    N[0] = 1.0 - r - s - t
    N[1] = r
    N[2] = s
    N[3] = t

    return N

def shape_function_gradients(r, s, t):
    """
    Return the gradient of the 4 shape functions with respect to parametric
    coordinates on GPU.

    Parameters
    ----------
    r : array_like or scalar
        First parametric coordinate in [0, 1].
    s : array_like or scalar
        Second parametric coordinate in [0, 1].
    t : array_like or scalar
        Third parametric coordinate in [0, 1].

    Returns
    -------
    cupy.ndarray
        An (4, 3, ...) array of shape function derivatives.
    """
    r = cp.asarray(r)
    s = cp.asarray(s)
    t = cp.asarray(t)

    dN = cp.empty((4, 3) + r.shape, dtype=float_cp)

    dN[0, 0] = -cp.ones_like(r)
    dN[0, 1] = -cp.ones_like(s)
    dN[0, 2] = -cp.ones_like(t)

    dN[1, 0] = cp.ones_like(r)
    dN[1, 1] = cp.zeros_like(s)
    dN[1, 2] = cp.zeros_like(t)

    dN[2, 0] = cp.zeros_like(r)
    dN[2, 1] = cp.ones_like(s)
    dN[2, 2] = cp.zeros_like(t)

    dN[3, 0] = cp.zeros_like(r)
    dN[3, 1] = cp.zeros_like(s)
    dN[3, 2] = cp.ones_like(t)

    return dN

@njit
def shape_function_gradients_cpu(r, s, t):
    """
    Return the gradient of the 4 shape functions with respect to parametric
    coordinates on CPU.

    Parameters
    ----------
    r : array_like or scalar
        First parametric coordinate in [0, 1].
    s : array_like or scalar
        Second parametric coordinate in [0, 1].
    t : array_like or scalar
        Third parametric coordinate in [0, 1].

    Returns
    -------
    numpy.ndarray
        An (4, 3, ...) array of shape function derivatives.
    """
    dN = np.empty((4, 3), dtype=np.float64)

    dN[0, 0] = -1.0
    dN[0, 1] = -1.0
    dN[0, 2] = -1.0

    dN[1, 0] = 1.0
    dN[1, 1] = 0.0
    dN[1, 2] = 0.0

    dN[2, 0] = 0.0
    dN[2, 1] = 1.0
    dN[2, 2] = 0.0

    dN[3, 0] = 0.0
    dN[3, 1] = 0.0
    dN[3, 2] = 1.0

    return dN

@njit
def element_jacobian(xc, yc, zc, n):
    """
    Compute the Jacobian matrix for the transformation from reference to physical space.

    Parameters
    ----------
    xc : array_like
        Array of length 4 containing x-coordinates of tetrahedral nodes.
    yc : array_like
        Array of length 4 containing y-coordinates of tetrahedral nodes.
    zc : array_like
        Array of length 4 containing z-coordinates of tetrahedral nodes.
    n : int
        Gauss point index (unused for tetrahedral elements since Jacobian is constant).

    Returns
    -------
    numpy.ndarray
        A (3, 3) array representing the Jacobian matrix.
    """
    J = np.zeros((3, 3), dtype=np.float64)

    J[:, 0] = xc[1:] - xc[0]
    J[:, 1] = yc[1:] - yc[0]
    J[:, 2] = zc[1:] - zc[0]

    return J

@njit
def gauss_quadrature():
    """
    Return standard Gauss-Legendre quadrature points and weights for linear
    tetrahedral elements.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing:
        - points: (4, 3) array of integration points in barycentric coordinates (r, s, t)
        - weights: (4,) array of corresponding quadrature weights
    """
    a = 0.58541020
    b = 0.13819660
    
    GAUSS_POINTS = np.array([
        [b, b, b],
        [a, b, b],
        [b, a, b],
        [b, b, a]
    ], dtype=np.float64)

    GAUSS_WEIGHTS = np.ones(4, dtype=np.float64) / 24.0
    
    return GAUSS_POINTS, GAUSS_WEIGHTS

@njit
def corners_local():
    """
    Return the corners of the reference tetrahedral element.
    """
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

_N_data = np.array([
    [0.5854102, 0.1381966, 0.1381966, 0.1381966],
    [0.1381966, 0.5854102, 0.1381966, 0.1381966],
    [0.1381966, 0.1381966, 0.5854102, 0.1381966],
    [0.1381966, 0.1381966, 0.1381966, 0.5854102]
])

_dN_data = np.array([
    [-1.0, -1.0, -1.0],
    [1.0,   0.0,  0.0],
    [0.0,   1.0,  0.0],
    [0.0,   0.0,  1.0]
])

@njit
def get_dN(n):
    """
    Return the 4×3 array of shape‐function derivatives at the n-th Gauss point.

    Parameters
    ----------
    n : int
        Gauss‐point index (0 ≤ n ≤ 3).

    Returns
    -------
    numpy.ndarray
        A (4,3) array of ∂N_i/∂ξ_j at Gauss point n.
    """
    return _dN_data

@njit
def get_N(n):
    """
    Return the size 4 array of shape‐function at the n-th Gauss point.

    Parameters
    ----------
    n : int
        Gauss‐point index (0 ≤ n ≤ 3).

    Returns
    -------
    numpy.ndarray
        A (4,) array of N_i at Gauss point n.
    """
    return _N_data[n]

