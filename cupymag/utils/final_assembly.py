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

import cupyx.scipy.sparse as cpx
from scipy.sparse import coo_matrix
import cupy as cp
import numpy as np

from cupymag.core.parameters import precision
from cupymag.utils.precision_select import get_float_type
float_cp = get_float_type(precision, backend="cupy")

int_cp = cp.int32

def extract_defect_dofs(elements, global_id):
    """
    This function identifies finite element nodes belonging to defect regions
    (marked by defect_flag=1) and returns the corresponding global degree of
    freedom indices as a CuPy array.

    Parameters
    ----------
    elements :
        Element connectivity array where each row contains the node indices
        for one element, with the last column containing the defect flag.
        Elements with defect_flag=1 are considered defect elements.
    global_id :
        Global degree of freedom mapping array where global_id[node_index]
        returns the global DOF indices associated with that node.

    Returns
    -------
    defect_dofs:
        Sorted array of unique global degree of freedom indices corresponding
        to all nodes within defect elements.
    """
    mask_defect = (elements[:, -1] == 1)
    if not cp.any(mask_defect):
        # no defect
        return cp.asarray([], dtype=int_cp)

    defect_elems = elements[mask_defect, 0:-1]
    defect_nodes = cp.unique(defect_elems.flatten())
    defect_dofs = cp.unique(global_id[defect_nodes])
    return defect_dofs

def enforce_defect_region_A(A, defect_dofs=None):
    """
    Modify sparse matrix A (CSR) on GPU to enforce m=0 in
    the defect region DOFs. Return A_new and the array of
    defect_dofs.

    Parameters
    ----------
    A : cupyx.scipy.sparse matrix (CSR format)
        System matrix to be modified. Typically represents the finite element
        discretization of the governing equations (e.g., stiffness matrix).
    defect_dofs : cupy.ndarray or numpy.ndarray
        Sorted array of unique global degree of freedom indices corresponding
        to all nodes within defect elements. If None, the original matrix
        is returned unchanged. Default is None.

    Returns
    -------
    cupyx.scipy.sparse matrix (CSR format)
        Modified system matrix A_new with defect DOFs pinned to 0.
    """
    if defect_dofs is None:
        return A

    if defect_dofs.size == 0:
        return A

    if isinstance(defect_dofs, np.ndarray):
        defect_dofs_cp = cp.asarray(defect_dofs, dtype=int_cp)
    else:
        defect_dofs_cp = defect_dofs

    A_coo = A.tocoo()
    row = A_coo.row
    col = A_coo.col
    data = A_coo.data

    mask = ~(cp.isin(row, defect_dofs_cp) | cp.isin(col, defect_dofs_cp))
    row_keep = row[mask]
    col_keep = col[mask]
    data_keep = data[mask]

    id_row = defect_dofs_cp
    id_col = defect_dofs_cp
    id_data = cp.ones_like(defect_dofs_cp, dtype=data.dtype)

    new_row = cp.concatenate([row_keep, id_row])
    new_col = cp.concatenate([col_keep, id_col])
    new_data = cp.concatenate([data_keep, id_data])

    A_new_coo = cpx.coo_matrix((new_data, (new_row, new_col)), shape=A.shape)
    A_new = A_new_coo.tocsr()

    return A_new


def enforce_defect_region_F(F, defect_dofs=None):
    """
    Modify sparse matrix F (CSR) on GPU to enforce rhs[defect_dofs] = 0.
    For the matrix F, we only need to zero out the rows corresponding to defect DOFs.
    We don't add diagonal entries since this is applied to the right-hand side.

    Parameters
    ----------
    F : cupyx.scipy.sparse.csr_matrix
        The F matrix in CSR format on GPU
    defect_dofs : numpy.ndarray or cupy.ndarray
        Array of DOF indices in the defect region

    Returns
    -------
    F_new : cupyx.scipy.sparse.csr_matrix
        Modified F matrix with rows corresponding to defect_dofs zeroed out
    """
    if defect_dofs is None:
        return F

    if defect_dofs.size == 0:
        return F

    if isinstance(defect_dofs, np.ndarray):
        defect_dofs_cp = cp.asarray(defect_dofs, dtype=int_cp)
    else:
        defect_dofs_cp = defect_dofs

    F_coo = F.tocoo()
    row = F_coo.row
    col = F_coo.col
    data = F_coo.data

    mask = ~cp.isin(row, defect_dofs_cp)

    row_keep = row[mask]
    col_keep = col[mask]
    data_keep = data[mask]

    F_new_coo = cpx.coo_matrix((data_keep, (row_keep, col_keep)), shape=F.shape)
    F_new = F_new_coo.tocsr()

    return F_new

def assemble_stiffness_matrix(rows_np, cols_np, vals_np, nDOFx, defect_dofs=None, nDOFy=None):
    """
    Parameters
    ----------
      rows_np: numpy coo rows
      cols_np: numpy coo cols
      vals_np: numpy coo vals
      defect_dofs: nodes that are defect (m=0)

    Returns
    -------
      A_cupy: cupy csr matrix
    """
    if nDOFy is None:
        nDOFy = nDOFx

    A_scipy = coo_matrix((vals_np, (rows_np, cols_np)), shape=(nDOFx, nDOFy)).tocsr()
    A_cupy = cpx.csr_matrix(A_scipy, dtype=float_cp)

    A_cupy = enforce_defect_region_A(A_cupy, defect_dofs)

    return A_cupy


def assemble_mass_matrix(rows_np, cols_np, vals_np, nDOFx, defect_dofs=None, nDOFy=None):
    """
    Parameters
    ----------
      rows_np: numpy coo rows
      cols_np: numpy coo cols
      vals_np: numpy coo vals
      defect_dofs: nodes that are defect (m=0)

    Returns
    -------
      F_cupy: cupy csr matrix
    """
    if nDOFy is None:
        nDOFy = nDOFx

    F_scipy = coo_matrix((vals_np, (rows_np, cols_np)), shape=(nDOFx, nDOFy)).tocsr()
    F_cupy = cpx.csr_matrix(F_scipy, dtype=float_cp)

    F_cupy = enforce_defect_region_F(F_cupy, defect_dofs)

    return F_cupy


def build_E0_from_m(lam100, lam111, m_nodes_cp):
    """
    Given m_nodes_cp: shape (N,3) [the magnetization at each node],
    return E0_cp: shape (6N,), containing the Voigt components
      ( exx, eyy, ezz, 2exy, 2eyz, 2exz )
    for each node, consistent with:

      E0_{ii} = (3/2) * lambda100 * (m_i^2 - 1/3)
      E0_{ij} = (3/2) * lambda111 * m_i m_j  (i != j)

    We'll store them in Voigt order per node:
      E0_cp[6*n+0] = exx
      E0_cp[6*n+1] = eyy
      E0_cp[6*n+2] = ezz
      E0_cp[6*n+3] = 2exy
      E0_cp[6*n+4] = 2eyz
      E0_cp[6*n+5] = 2exz

    Parameters
    ----------
    lam100: Magnetostriction constants for <100> direction.
    lam111: Magnetostriction constants for <111> direction.
    m_nodes_cp: The magnetization field at each node in cupy array.

    Returns
    -------
      E0_cp: The spontaneous strain E0(m) in cupy array.
    """
    # Ensure shape (N,3)
    if m_nodes_cp.ndim == 1:
        N = m_nodes_cp.size // 3
        m_nodes_cp = m_nodes_cp.reshape((N,3))

    N = m_nodes_cp.shape[0]
    E0_cp = cp.zeros((6*N,), dtype=float_cp)

    mx = m_nodes_cp[:,0]
    my = m_nodes_cp[:,1]
    mz = m_nodes_cp[:,2]

    one_third = cp.float64(1.0 / 3.0)
    three_over_2 = cp.float64(1.5)

    exx = three_over_2 * lam100 * (mx * mx - one_third)
    eyy = three_over_2 * lam100 * (my * my - one_third)
    ezz = three_over_2 * lam100 * (mz * mz - one_third)

    exy = three_over_2 * lam111 * (mx * my)  # exy
    eyz = three_over_2 * lam111 * (my * mz)  # eyz
    exz = three_over_2 * lam111 * (mx * mz)  # exz

    E0_cp[0::6] = exx
    E0_cp[1::6] = eyy
    E0_cp[2::6] = ezz
    E0_cp[3::6] = 2.0 * exy
    E0_cp[4::6] = 2.0 * eyz
    E0_cp[5::6] = 2.0 * exz

    return E0_cp
