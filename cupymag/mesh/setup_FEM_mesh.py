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

from cupymag.core.parameters import *
from cupymag.utils.defect_shapes import get_defect_shape_function
from cupymag.utils.final_assembly import extract_defect_dofs

if grid_type == "Hex":
    from cupymag.mesh.gridHex import HexGrid
elif grid_type == "Tet":
    from cupymag.mesh.gridTet import TetraGrid
else:
    raise NotImplementedError(f"Grid type '{grid_type}' is not supported.")

from cupymag.utils.precision_select import get_float_type
float_cp = get_float_type(precision, backend="cupy")

class FEMMesh:
    """
    Finite Element Method mesh handler.

    This class encapsulates all mesh-related operations and data
    """
    DefDOF = None
    _mesh_computed = False

    def __init__(self):
        """
        Initialize FEM mesh.

        Note
        ----
        grid_type : str
            Type of grid ('Hex' or 'Tet')
        """
        self.grid_type = grid_type
        self.float_cp = float_cp
        self.fem_grid = None
        self.node_coords_np = None
        self.node_coords_cp = None
        self.elements_np = None
        self.elements_cp = None
        self.global_id_np = None
        self.global_id_cp = None
        self.n_dof = None
        self.n_nodes = None
        self.n_elements = None

        if not FEMMesh._mesh_computed:
            self.generate_mesh()
            FEMMesh._mesh_computed = True

            if FEMMesh.DefDOF is None:
                FEMMesh.DefDOF = extract_defect_dofs(self.elements_cp, self.global_id_cp)

    def generate_mesh(self):
        """
        Generate the finite element mesh based on grid type.
        """
        defect_center = globals().get("defect_center", None)

        if self.grid_type == "Hex":
            self.fem_grid = HexGrid(Nx, Ny, Nz, Ndx, Ndy, Ndz, defect_center)
        elif self.grid_type == "Tet":
            shape_fn = get_defect_shape_function(defect_shape, shape_params)
            self.fem_grid = TetraGrid(mesh_file, shape_fn, defect_center)

        self.node_coords_cp, self.elements_cp = self.fem_grid.std_fem_mesh()

        self.node_coords_np = self.node_coords_cp.get()
        self.elements_np = self.elements_cp.get()

        self.global_id_np = self.fem_grid.build_periodic_node_map(self.node_coords_np)
        self.global_id_cp = cp.asarray(self.global_id_np)

        self.n_dof = len(np.unique(self.global_id_np))
        self.n_nodes = self.node_coords_cp.shape[0]
        self.n_elements = self.elements_cp.shape[0]

        return self

    def get_mesh_data(self):
        """
        Get mesh data for computation.

        Returns
        -------
        dict
            Dictionary containing all mesh-related data
        """
        return {
            'node_coords_np': self.node_coords_np,
            'node_coords_cp': self.node_coords_cp,
            'elements_np': self.elements_np,
            'elements_cp': self.elements_cp,
            'global_id_np': self.global_id_np,
            'global_id_cp': self.global_id_cp,
            'n_dof': self.n_dof,
            'n_nodes': self.n_nodes,
            'n_elements': self.n_elements
        }