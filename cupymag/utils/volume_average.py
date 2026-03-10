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
import cupy as cp

from cupymag.core.parameters import grid_type, precision
if grid_type == "Hex":
    from cupymag.mesh.gridHex import HexGrid
    from cupymag.mesh.ShapeHex import (
        gauss_quadrature, shape_functions, shape_function_gradients
    )
elif grid_type == "Tet":
    from cupymag.mesh.gridTet import TetraGrid
    from cupymag.mesh.ShapeTet import (
        gauss_quadrature, shape_functions, shape_function_gradients
    )
else:
    raise NotImplementedError(f"Grid type '{grid_type}' is not supported.")

from cupymag.utils.precision_select import get_float_type
float_cp = get_float_type(precision, backend="cupy")

import meshio


class VolumeAverage:
    """
    GPU-accelerated volume averaging for finite element meshes using Gauss quadrature.

    This class provides efficient volume-weighted averaging of nodal fields over elements,
    with support for defective elements and periodic boundary conditions. It also includes
    visualization capabilities through ParaView vtu file export with interpolation blending
    between nodal and cell-centered representations.

    Attributes
    ----------
        coords: Node coordinates array (nNodes, 3)
        elements: Element connectivity with defect flags (nElems, nNodesPerElem+1)
        original_global_id: Global DOF mapping for periodic boundaries (nNodes,)
        n_nodes_per_elem: Number of nodes per element
        corner_dofs: Element corner DOF indices (nElems, nNodesPerElem)
        defect_flags: Element defect indicators (nElems,)
        gauss_points: Gauss quadrature points on GPU
        gauss_weights: Gauss quadrature weights on GPU
        detJ: Jacobian determinants at Gauss points (nElems, nGaussPoints)
        N_gauss_gpu: Shape function values at Gauss points (nGaussPoints, nNodesPerElem)
        W_gauss_gpu: Gauss weights on GPU

    Methods
    -------
        compute_detJ_gpu_vectorized(): Compute Jacobian determinants for all elements
        build_N_gauss(): Evaluate shape functions at Gauss points
        compute_average_field_gpu(): Perform volume-weighted field averaging
        write_to_paraview(): Export mesh and fields to ParaView with interpolation options
    """
    def __init__(self, coords, elements, global_id):
        self.coords = coords
        self.elements = elements

        self.original_global_id = global_id

        self.n_nodes_per_elem = elements.shape[1] - 1
        self.corner_dofs = global_id[elements[:, 0:self.n_nodes_per_elem]]
        self.defect_flags = elements[:, -1]

        self.gauss_points, self.gauss_weights = gauss_quadrature()
        self.gauss_points = cp.asarray(self.gauss_points, dtype=float_cp)
        self.gauss_weights = cp.asarray(self.gauss_weights, dtype=float_cp)

        # Compute necessary quantities for volume averaging
        self.detJ = self.compute_detJ_gpu_vectorized()
        self.N_gauss_gpu = self.build_N_gauss()
        self.W_gauss_gpu = self.gauss_weights

    def compute_detJ_gpu_vectorized(self):
        """
        Compute Jacobian determinants at all Gauss points for all elements.

        Uses vectorized GPU operations for efficient computation across the entire mesh.
        Takes absolute value to handle potentially inverted elements.

        Returns
        -------
            detJ_g: Jacobian determinants (nElems, nGaussPoints)
        """
        coords = self.coords
        corner_dofs = self.corner_dofs

        n_elems = corner_dofs.shape[0]

        gauss_points = self.gauss_points
        ngp = len(gauss_points)

        # Gather corner node coordinates: (nElems, nNodesPerElem, 3)
        corner_coords = coords[corner_dofs]

        detJ_g = cp.zeros((n_elems, ngp), dtype=float_cp)

        for igp in range(ngp):
            r, s, t = gauss_points[igp]

            dN = shape_function_gradients(r, s, t)
            
            # Ensure we only use the correct number of shape functions for this element type
            dN = dN[:self.n_nodes_per_elem, :]

            # 'enk,nj->ekj': e=element, n=node, k=component(x,y,z), j=derivative(r,s,t)
            J = cp.einsum('enk,nj->ekj', corner_coords, dN)

            detJ = (J[:, 0, 0] * (J[:, 1, 1] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 1]) -
                    J[:, 0, 1] * (J[:, 1, 0] * J[:, 2, 2] - J[:, 1, 2] * J[:, 2, 0]) +
                    J[:, 0, 2] * (J[:, 1, 0] * J[:, 2, 1] - J[:, 1, 1] * J[:, 2, 0]))

            detJ = cp.abs(detJ)

            detJ_g[:, igp] = detJ

        return detJ_g

    def build_N_gauss(self):
        """
        Evaluate shape functions at all Gauss points.

        Returns
        -------
            N_gauss: Shape function values (nGaussPoints, nNodesPerElem)
        """
        gauss_points, gauss_weights = self.gauss_points, self.gauss_weights
        ngp = len(gauss_weights)
        n_nodes_per_elem = self.n_nodes_per_elem

        N_gauss = cp.zeros((ngp, n_nodes_per_elem), dtype=float_cp)

        for igp in range(ngp):
            r, s, t = gauss_points[igp]
            Nvals = shape_functions(r, s, t)
            for b in range(n_nodes_per_elem):
                N_gauss[igp, b] = Nvals[b]

        return N_gauss

    def compute_average_field_gpu(self, m, defect_flag=None):
        """
        Compute volume-weighted average of a nodal field using Gauss quadrature.

        Performs efficient GPU-based integration over all elements (or non-defective
        elements if defect_flag is specified).

        Parameters
        ----------
            m: Nodal field values (nDOF,) or (nDOF, nComp)
            defect_flag: Optional element defect flags (nElems,) - excludes defective elements

        Returns
        -------
            avg_vec_gpu: Volume-weighted average as scalar (for vector fields)
                or array (for tensor fields)
        """
        corner_dofs = self.corner_dofs
        N_gauss = self.N_gauss_gpu
        W_gauss = self.W_gauss_gpu
        detJ_g = self.detJ

        is_scalar = len(m.shape) == 1

        if is_scalar:
            m = m.reshape(-1, 1)

        corner_U = m[corner_dofs]

        MGP = cp.einsum('ebk,gb->egk', corner_U, N_gauss)

        volume_weights = detJ_g * W_gauss
        volume_weights_3 = volume_weights[:, :, None]

        partial_integration = MGP * volume_weights_3

        partial_integration_e = cp.sum(partial_integration, axis=1)

        partial_volume = cp.sum(volume_weights, axis=1)

        if defect_flag is not None:
            df_float = defect_flag.astype(float_cp)
            is_not_defect = (1.0 - df_float)
            partial_integration_e *= is_not_defect[:, None]
            partial_volume_summed = cp.sum(partial_volume * is_not_defect)
        else:
            partial_volume_summed = cp.sum(partial_volume)

        total_integration = cp.sum(partial_integration_e, axis=0)
        total_volume = partial_volume_summed

        # Ensure non-zero volume to avoid division by zero
        if abs(total_volume) < 1e-6:
            print("WARNING: Total volume is very small, results may be unstable")
            total_volume = max(abs(total_volume), 1e-6)

        avg_vec_gpu = total_integration / total_volume

        if is_scalar:
            return avg_vec_gpu[0]
        else:
            return avg_vec_gpu


    def write_to_paraview(self, field_dict, filename, alpha=0.5, eps=1e-14):
        """
        Visualize fields in Paraview with proper handling of periodic boundaries.

        Parameters:
        -----------
        field_dict : dict
            Dictionary of fields to visualize. Keys are field names, values are (nDOF,) or (nDOF, nComp) arrays.
        filename : str
            Output filename for visualization
        alpha : float
            Blending parameter: 1.0 = purely nodal, 0.0 = purely cell-centered
        """
        xp = cp.get_array_module(self.coords) if (cp and hasattr(self.coords, "device")) else np
        asxp = lambda a: xp.asarray(a)

        coords = asxp(self.coords)
        cells = asxp(self.elements)[:, :self.n_nodes_per_elem]
        gid = asxp(self.original_global_id).astype(xp.int64)

        n_nodes, n_elems = coords.shape[0], cells.shape[0]
        n_per_elem = self.n_nodes_per_elem

        N_g = asxp(self.N_gauss_gpu)
        W_g = asxp(self.W_gauss_gpu)
        detJ = asxp(self.detJ)
        cDOF = asxp(self.corner_dofs)

        from scipy.sparse import coo_matrix
        row = cells.reshape(-1)
        col = xp.repeat(xp.arange(n_elems), n_per_elem)
        data = xp.ones_like(row, dtype=xp.float32)

        A = coo_matrix(
            (data.get() if xp is cp else data,
             (row.get() if xp is cp else row,
              col.get() if xp is cp else col)),
            shape=(n_nodes, n_elems)
        ).tocsr()
        A_sum = xp.asarray(A.sum(axis=1)).ravel()  # (#nodes,)

        uniq, inverse, counts = xp.unique(gid, return_inverse=True, return_counts=True)
        dof_inv_cnt = (1.0 / counts)[inverse][:, None]  # (#nodes,1)

        point_data, cell_data = {}, {}

        for name, f in field_dict.items():
            f = asxp(f)
            if f.ndim == 1:
                f = f[:, None]
            nC = f.shape[1]

            node_val = f[gid]  # (#nodes,nC)

            if alpha < 0.999:
                f_e = f[cDOF]  # (e,n,C)
                # (e,g,C)
                F_e_g = xp.tensordot(f_e, N_g, axes=(1, 1)).transpose(0, 2, 1)
                weight = detJ * W_g  # (e,g)

                num = xp.einsum("eg,egc->ec", weight, F_e_g)
                den = xp.sum(weight, axis=1, keepdims=True)  # (e,1)

                elem_val = xp.where(den > eps, num / den, xp.mean(f_e, axis=1))

                cc_val = A @ (elem_val.get() if xp is cp else elem_val)  # (n,C)
                cc_val = xp.asarray(cc_val)
                cc_val = xp.where(A_sum[:, None] > 0,
                                  cc_val / A_sum[:, None],
                                  node_val)

                buf = xp.zeros((uniq.size, nC), dtype=cc_val.dtype)
                xp.add.at(buf, inverse, cc_val)
                cc_val = buf[inverse] * dof_inv_cnt

                node_val = alpha * node_val + (1.0 - alpha) * cc_val

            if nC == 1:
                point_data[name] = xp.asnumpy(node_val[:, 0]) if xp is cp else node_val[:, 0]
            else:
                for c in range(nC):
                    point_data[f"{name}_{c + 1}"] = (
                        xp.asnumpy(node_val[:, c]) if xp is cp else node_val[:, c]
                    )

        if getattr(self, "defect_flags", None) is not None:
            df = asxp(self.defect_flags)
            cell_data["defect_flag"] = [xp.asnumpy(df) if xp is cp else df]

        elem_type = {4: "tetra", 8: "hexahedron"}[n_per_elem]
        meshio.write(
            filename,
            meshio.Mesh(
                points=xp.asnumpy(coords),
                cells=[(elem_type, xp.asnumpy(cells))],
                point_data=point_data,
                cell_data=cell_data,
            ),
        )
        print(f"Field written to {filename}.")
