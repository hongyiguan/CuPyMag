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
import cupyx.scipy.sparse.linalg as spla


def solve_cg(A, b, M=None, x0=None, tol=1e-7, maxiter=5000, use_init=False, system=None):
    """
    Solve Ax = b using CuPy's Conjugate Gradient (CG) method.

    Parameters
    ----------
    A : scipy.sparse matrix (CSR format)
        Coefficient matrix of the linear system. Must be symmetric positive definite.
    b : cupy.ndarray
        Right-hand side vector of the linear system.
    M : scipy.sparse matrix or cupy.ndarray, optional
        Preconditioner matrix to accelerate convergence. If None, no preconditioning
        is applied. Default is None.
    x0 : cupy.ndarray, optional
        Initial guess for the solution vector. Only used if use_init is True.
        Default is None.
    tol : float, optional
        Convergence tolerance for the relative residual norm. The algorithm
        terminates when ||r||/||b|| < tol. Default is 1e-7.
    maxiter : int, optional
        Maximum number of iterations allowed. Default is 5000.
    use_init : bool, optional
        Flag to determine whether to use the provided initial guess x0.
        Default is False.
    system : str, optional
        Identifier string for the linear system being solved. Used only for
        error reporting to provide context in failure messages. Default is None.

    Returns
    -------
    cupy.ndarray
        Solution vector x satisfying Ax = b within the specified tolerance.

    Raises
    ------
    RuntimeError
        If the Conjugate Gradient method fails to converge within maxiter iterations.
        The error message includes the convergence status code and final residual norm.
    """
    if use_init == False:
        xsol, info = spla.cg(A, b, M=M, x0=None, tol=tol, maxiter=maxiter)
    else:
        xsol, info = spla.cg(A, b, M=M, x0=x0, tol=tol, maxiter=maxiter)

    if info == 0:
        # Converged
        return xsol
    else:
        r = b - A @ xsol
        rnorm = cp.linalg.norm(r)
        if system is not None:
            error_message = f"Error! CG for {system} did not converge. info={info}, residual={rnorm}."
        else:
            error_message = f"Error! CG did not converge. info={info}, residual={rnorm}."
        raise RuntimeError(error_message)

