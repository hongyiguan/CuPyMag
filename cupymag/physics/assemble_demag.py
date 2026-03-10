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

import numpy as np
from numba import float64, int64, int32
from numba.experimental import jitclass

from cupymag.core.parameters import grid_type
if grid_type == "Hex":
    from cupymag.mesh.gridHex import HexGrid
    from cupymag.mesh.ShapeHex import (
        gauss_quadrature, element_jacobian, get_dN, get_N
    )
elif grid_type == "Tet":
    from cupymag.mesh.gridTet import TetraGrid
    from cupymag.mesh.ShapeTet import (
        gauss_quadrature, element_jacobian, get_dN, get_N
    )
else:
    raise NotImplementedError(f"Grid type '{grid_type}' is not supported.")

int_numba = int32
int_np = np.int32

spec = {
    'node_coords': float64[:, :],
    'elements': int_numba[:, :],
    'Nnodes_per_element': int_numba,
    'gauss_points': float64[:, :],
    'gauss_weights': float64[:],
    'global_id': int_numba[:],
    'anchor_row': int64,
}


@jitclass(spec)
class AssembleDemag:
    """
    Finite element assembly class for magnetostatic equilibrium.

    This class handles the assembly of stiffness matrices and derivative operators
    for solving Laplace's equation in the context of demagnetization field calculations.
    The implementation uses Numba JIT compilation for performance optimization.

    Attributes
    ----------
    node_coords : float64[:, :]
        Nodal coordinates array of shape (n_nodes, 3)
    elements : int32[:, :]
        Element connectivity array including defect IDs
    Nnodes_per_element : int32
        Number of nodes per element (excluding defect ID column)
    gauss_points : float64[:, :]
        Gauss quadrature points in reference coordinates
    gauss_weights : float64[:]
        Gauss quadrature weights
    global_id : int32[:]
        Global node ID mapping array for periodic boundary condition
    anchor_row : int64
        Row index for anchoring one DOF to remove nullspace

    Methods
    -------
    compute_element_stiffness_cpu(xc, yc, zc)
        Compute element stiffness matrix for Laplace operator
    compute_element_F_cpu(xc, yc, zc)
        Compute element derivative operator matrices (F)
    build_coo_matrix_A_numba()
        Assemble global stiffness matrix in COO sparse format
    build_coo_matrices_F_numba()
        Assemble global F matrices in COO sparse format
    impose_anchor_node_dof0_coo(rows, cols, vals)
        Remove nullspace by anchoring one DOF to zero
    """
    def __init__(self, node_coords, elements, global_id=None):
        self.node_coords = node_coords
        self.elements = elements
        self.Nnodes_per_element = self.elements.shape[1] - 1
        self.gauss_points, self.gauss_weights = gauss_quadrature()
        self.global_id = global_id
        self.anchor_row = 0

    def compute_element_stiffness_cpu(self, xc, yc, zc):
        """
        Assemble the element stiffness matrix K_e:
          K_e[a,b] = (grad N_a, grad N_b).

        Parameters
        ----------
          xc, yc, zc : arrays containing the nodal coordinates for the element.

        Returns
        -------
          K_e : the assembled element stiffness matrix.
        """
        nNodes = self.Nnodes_per_element
        ngp = len(self.gauss_points)

        K_e = np.zeros((nNodes, nNodes), dtype=np.float64)

        for igp in range(ngp):
            J_ = element_jacobian(xc, yc, zc, igp)
            detJ = np.linalg.det(J_)
            if abs(detJ) < 1e-14:
                return np.zeros((nNodes, nNodes))
            J_inv = np.linalg.inv(J_)

            w = self.gauss_weights[igp]
            dN_rst = get_dN(igp)
            for a in range(nNodes):
                gx_a = J_inv[0, 0] * dN_rst[a, 0] + J_inv[0, 1] * dN_rst[a, 1] + J_inv[0, 2] * dN_rst[a, 2]
                gy_a = J_inv[1, 0] * dN_rst[a, 0] + J_inv[1, 1] * dN_rst[a, 1] + J_inv[1, 2] * dN_rst[a, 2]
                gz_a = J_inv[2, 0] * dN_rst[a, 0] + J_inv[2, 1] * dN_rst[a, 1] + J_inv[2, 2] * dN_rst[a, 2]
                for b in range(nNodes):
                    gx_b = J_inv[0, 0] * dN_rst[b, 0] + J_inv[0, 1] * dN_rst[b, 1] + J_inv[0, 2] * dN_rst[b, 2]
                    gy_b = J_inv[1, 0] * dN_rst[b, 0] + J_inv[1, 1] * dN_rst[b, 1] + J_inv[1, 2] * dN_rst[b, 2]
                    gz_b = J_inv[2, 0] * dN_rst[b, 0] + J_inv[2, 1] * dN_rst[b, 1] + J_inv[2, 2] * dN_rst[b, 2]
                    dot_ab = gx_a * gx_b + gy_a * gy_b + gz_a * gz_b
                    K_e[a, b] += dot_ab * (detJ * w)
        return K_e

    def compute_element_F_cpu(self, xc, yc, zc):
        """
        Assemble the element F matrices:
          Fex[a,b] = (N_a, dN_b/dx),
          Fey[a,b] = (N_a, dN_b/dy),
          Fez[a,b] = (N_a, dN_b/dz).

        Parameters
        ----------
          xc, yc, zc : arrays containing the nodal coordinates for the element.

        Returns
        -------
          Fex, Fey, Fez : the assembled mass matrices.
        """
        nNodes = self.Nnodes_per_element
        ngp = len(self.gauss_points)

        Fex = np.zeros((nNodes, nNodes), dtype=np.float64)
        Fey = np.zeros((nNodes, nNodes), dtype=np.float64)
        Fez = np.zeros((nNodes, nNodes), dtype=np.float64)

        for igp in range(ngp):
            w = self.gauss_weights[igp]

            J_ = element_jacobian(xc, yc, zc, igp)

            detJ = np.linalg.det(J_)
            if abs(detJ) < 1e-14:
                Z = np.zeros((nNodes, nNodes))
                return Z, Z, Z
            J_inv = np.linalg.inv(J_)

            dN_rst = get_dN(igp)
            N_rst = get_N(igp)
            grad_xyz = np.zeros((nNodes, 3))

            for a in range(nNodes):
                grad_xyz[a, 0] = J_inv[0, 0] * dN_rst[a, 0] + J_inv[0, 1] * dN_rst[a, 1] + J_inv[0, 2] * dN_rst[
                    a, 2]
                grad_xyz[a, 1] = J_inv[1, 0] * dN_rst[a, 0] + J_inv[1, 1] * dN_rst[a, 1] + J_inv[1, 2] * dN_rst[
                    a, 2]
                grad_xyz[a, 2] = J_inv[2, 0] * dN_rst[a, 0] + J_inv[2, 1] * dN_rst[a, 1] + J_inv[2, 2] * dN_rst[
                    a, 2]

            for a in range(nNodes):
                for b in range(nNodes):
                    Fex[a, b] += N_rst[a] * grad_xyz[b, 0] * (detJ * w)
                    Fey[a, b] += N_rst[a] * grad_xyz[b, 1] * (detJ * w)
                    Fez[a, b] += N_rst[a] * grad_xyz[b, 2] * (detJ * w)
        return Fex, Fey, Fez

    def build_coo_matrix_A_numba(self):
        """
        Build the global stiffness matrix in COO format.
        Automatically paralleled by Numba.

        Returns
        -------
          rows_np, cols_np, vals_np  (all 1D arrays for COO format).
        """
        node_coords_np = self.node_coords
        elements_np = self.elements
        global_id_np = self.global_id

        Ne = elements_np.shape[0]

        nnz = Ne * (self.Nnodes_per_element)**2

        rows_np = np.empty(nnz, dtype=int_np)
        cols_np = np.empty(nnz, dtype=int_np)
        vals_np = np.empty(nnz, dtype=np.float64)

        idx = 0
        for e in range(Ne):
            corner_ids = elements_np[e, 0:self.Nnodes_per_element]

            xc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            yc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            zc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            for i in range(self.Nnodes_per_element):
                nid = corner_ids[i]
                xc[i] = node_coords_np[nid, 0]
                yc[i] = node_coords_np[nid, 1]
                zc[i] = node_coords_np[nid, 2]

            K_e = self.compute_element_stiffness_cpu(xc, yc, zc)

            for a in range(self.Nnodes_per_element):
                ra = global_id_np[corner_ids[a]]
                for b in range(self.Nnodes_per_element):
                    cb = global_id_np[corner_ids[b]]
                    rows_np[idx] = ra
                    cols_np[idx] = cb
                    vals_np[idx] = K_e[a, b]
                    idx += 1

        # Anchor a single DOF to 0
        rows_np, cols_np, vals_np = self.impose_anchor_node_dof0_coo(rows_np, cols_np, vals_np)
        return rows_np, cols_np, vals_np

    def build_coo_matrices_F_numba(self):
        """
        Build the global F matrix in COO format.
        Automatically paralleled by Numba.

        Returns
        -------
          rows_np, cols_np, vals_np  (all 1D arrays for COO format).
        """
        node_coords_np = self.node_coords
        elements_np = self.elements
        global_id_np = self.global_id

        Ne = elements_np.shape[0]

        nnz = Ne * (self.Nnodes_per_element)**2

        Fx_rows = np.empty(nnz, dtype=int_np)
        Fx_cols = np.empty(nnz, dtype=int_np)
        Fx_vals = np.zeros(nnz, dtype=np.float64)

        Fy_rows = np.empty(nnz, dtype=int_np)
        Fy_cols = np.empty(nnz, dtype=int_np)
        Fy_vals = np.zeros(nnz, dtype=np.float64)

        Fz_rows = np.empty(nnz, dtype=int_np)
        Fz_cols = np.empty(nnz, dtype=int_np)
        Fz_vals = np.zeros(nnz, dtype=np.float64)

        idx = 0
        for e in range(Ne):
            corner_ids = elements_np[e, 0:self.Nnodes_per_element]

            xc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            yc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            zc = np.zeros(self.Nnodes_per_element, dtype=np.float64)
            for i in range(self.Nnodes_per_element):
                nid = corner_ids[i]
                xc[i] = node_coords_np[nid, 0]
                yc[i] = node_coords_np[nid, 1]
                zc[i] = node_coords_np[nid, 2]

            Fex, Fey, Fez = self.compute_element_F_cpu(xc, yc, zc)

            for a in range(self.Nnodes_per_element):
                ra = global_id_np[corner_ids[a]]
                for b in range(self.Nnodes_per_element):
                    cb = global_id_np[corner_ids[b]]
                    Fx_rows[idx] = ra
                    Fx_cols[idx] = cb
                    Fx_vals[idx] = Fex[a, b]

                    Fy_rows[idx] = ra
                    Fy_cols[idx] = cb
                    Fy_vals[idx] = Fey[a, b]

                    Fz_rows[idx] = ra
                    Fz_cols[idx] = cb
                    Fz_vals[idx] = Fez[a, b]

                    idx += 1

        # F is also served as derivative operators
        # So we do not impose one DOF to 0 here
        #Fx_rows, Fx_cols, Fx_vals = self.impose_anchor_node_dof0_coo_F(Fx_rows, Fx_cols, Fx_vals)
        #Fy_rows, Fy_cols, Fy_vals = self.impose_anchor_node_dof0_coo_F(Fy_rows, Fy_cols, Fy_vals)
        #Fz_rows, Fz_cols, Fz_vals = self.impose_anchor_node_dof0_coo_F(Fz_rows, Fz_cols, Fz_vals)

        return (Fx_rows, Fx_cols, Fx_vals,
                Fy_rows, Fy_cols, Fy_vals,
                Fz_rows, Fz_cols, Fz_vals)

    def impose_anchor_node_dof0_coo(self, rows_np, cols_np, vals_np):
        """
        Removes the Laplacian nullspace by pinning the potential at DOF=0 => U(0)=0.

        Parameters
        ----------
          rows_np, cols_np, vals_np initially without one DOF pinned.

        Returns
        -------
          rows_np, cols_np, vals_np with one DOF pinned.
        """
        anchor = self.anchor_row

        for i in range(vals_np.size):
            if rows_np[i] == anchor or cols_np[i] == anchor:
                vals_np[i] = 0.0

        N = rows_np.size

        new_N = N + 1
        new_rows = np.empty(new_N, dtype=rows_np.dtype)
        new_cols = np.empty(new_N, dtype=cols_np.dtype)
        new_vals = np.empty(new_N, dtype=vals_np.dtype)

        for i in range(N):
            new_rows[i] = rows_np[i]
            new_cols[i] = cols_np[i]
            new_vals[i] = vals_np[i]

        new_rows[N] = anchor
        new_cols[N] = anchor
        new_vals[N] = 1.0

        return new_rows, new_cols, new_vals

    def impose_anchor_node_dof0_coo_F(self, rows_np, cols_np, vals_np):
        """
        Removes the Laplacian nullspace by pinning the potential at DOF=0 => U(0)=0.
        For the matrix F, zero out one row. So that rhs[anchor] = F.m[anchor]=0.

        Parameters
        ----------
          rows_np, cols_np, vals_np initially without one DOF pinned.

        Returns
        -------
          rows_np, cols_np, vals_np with one DOF pinned.
        """
        anchor = self.anchor_row
        for i in range(vals_np.size):
            if rows_np[i] == anchor:
                vals_np[i] = 0.0

        return rows_np, cols_np, vals_np
