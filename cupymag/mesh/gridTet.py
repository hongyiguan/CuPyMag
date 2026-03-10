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
from cupymag.utils.defect_shapes import *

import pandas as pd
from io import StringIO

int_np = np.int32
int_cp = cp.int32

class TetraGrid:
    """
     A tetrahedral finite element mesh processor for 3D domains with embedded defect regions.

     This class reads tetrahedral meshes from Nastran (.nas) format files and processes them
     to identify elements within a specified defect region.

     Parameters
     ----------
     nas_filename : str
         Path to the Nastran (.nas) format file containing the tetrahedral mesh.
         The file should contain GRID entries for node coordinates and CTETRA entries
         for tetrahedral element connectivity.
     defect_shape : callable
         A function that defines the defect region geometry.
     defect_center : array_like, optional
         Coordinates of the defect region center as [x, y, z]. If None, the geometric
         center of the mesh bounding box is used automatically.

     Attributes
     ----------
     nas_filename : str
         Path to the input Nastran file.
     defect_shape : callable
         Defect region geometry function.
     defect_center : cupy.ndarray
         Coordinates of the defect center.
     node_ids : numpy.ndarray
         Original node IDs from the Nastran file.
     node_coords : numpy.ndarray
         Node coordinates array with shape (n_nodes, 3).
     element_ids : numpy.ndarray
         Original element IDs from the Nastran file.
     element_nodes : numpy.ndarray
         Element connectivity as zero-based node indices with shape (n_elements, 4).

     Methods
     -------
     read_nas_file_pd()
         Parse the Nastran file to extract mesh data using pandas for robust parsing.
     std_fem_mesh()
         Generate standard FEM format arrays with defect classification.
     build_periodic_node_map(node_coords, tol=1e-12)
         Create node mapping for periodic boundary conditions.
     write_mesh_to_paraview(filename="tetrahedral_mesh.vtu")
         Export mesh to VTU format for visualization in ParaView.
    """
    def __init__(self, nas_filename, defect_shape, defect_center=None):
        self.nas_filename = nas_filename
        self.defect_shape = defect_shape
        self.defect_center = defect_center

        self.node_ids, self.node_coords, self.element_ids, self.element_nodes = self.read_nas_file_pd()
        
        # If defect_center not provided, use center of the mesh
        if self.defect_center is None:
            min_coords = cp.min(self.node_coords, axis=0)
            max_coords = cp.max(self.node_coords, axis=0)
            self.defect_center = (max_coords + min_coords) / 2

        self.defect_center = cp.asarray(self.defect_center)

    def read_nas_file_pd(self):
        """
        Read a NAS file and extract node coordinates and element connectivity
        using pandas (with whitespace-delimited parsing).

        Returns
        -------
        node_ids : np.ndarray
            Array of node IDs
        node_coords : np.ndarray
            Array of node coordinates with shape (n_nodes, 3)
        element_ids : np.ndarray
            Array of element IDs
        element_nodes_idx : np.ndarray
            Array of element connectivity (as node indices) with shape (n_elements, 4)
        """
        grids, tetras = [], []
        with open(self.nas_filename, "r") as fh:
            for raw in fh:
                if raw.startswith('$') or raw.isspace():
                    continue
                if raw.lstrip().startswith('ENDDATA'):
                    break

                line = raw.rstrip()

                if line.startswith('GRID'):
                    grids.append(line)
                elif line.startswith('CTETRA'):
                    tetras.append(line)

        if not grids or not tetras:
            raise ValueError("File contains no GRID or CTETRA cards")

        csv_opts = dict(
            header=None,
            engine="c",  # fast C engine
            sep=",",
            skipinitialspace=True,
            dtype=str
        )

        df_grid = pd.read_csv(StringIO("\n".join(grids)),
                              names=['GRID', 'ID', 'CP', 'X', 'Y', 'Z'],
                              **csv_opts)[['ID', 'X', 'Y', 'Z']]
        df_grid = df_grid[['ID', 'X', 'Y', 'Z']]

        df_tet = pd.read_csv(StringIO("\n".join(tetras)),
                             names=['CTETRA', 'EID', 'PID', 'N1', 'N2', 'N3', 'N4'],
                             **csv_opts)[['EID', 'N1', 'N2', 'N3', 'N4']]
        df_tet = df_tet[['EID', 'N1', 'N2', 'N3', 'N4']]

        node_ids = df_grid['ID'].astype(np.int32).to_numpy(copy=False)
        node_xyz = df_grid[['X', 'Y', 'Z']].astype(np.float64).to_numpy(copy=False)
        elem_ids = df_tet['EID'].astype(np.int32).to_numpy(copy=False)
        elem_nodes = df_tet[['N1', 'N2', 'N3', 'N4']].astype(np.int32).to_numpy(copy=False)

        sort_idx = np.argsort(node_ids)
        sorted_node_ids = node_ids[sort_idx]

        elem_nodes_idx = sort_idx[np.searchsorted(sorted_node_ids,
                                                  elem_nodes,
                                                  side='left')]

        return node_ids, node_xyz, elem_ids, elem_nodes_idx

    def std_fem_mesh(self):
        """
        Return the tetrahedral mesh in standard FEM format with appropriate node ordering.

        Returns:
        --------
        node_coords : cupy.ndarray
            Array of node coordinates with shape (n_nodes, 3)
        elements : cupy.ndarray
            Array of shape (n_elements, 5) where the first 4 columns are node indices
            and the last column is the defect flag (0 or 1)
        """
        node_coords_cp = cp.array(self.node_coords)

        n_elements = len(self.element_nodes)
        elements = cp.zeros((n_elements, 5), dtype=int_cp)
        elements[:, :4] = cp.array(self.element_nodes)

        centers = node_coords_cp[elements[:, :4]].mean(axis=1)
        rel_coords = centers - self.defect_center

        mask = self.defect_shape(rel_coords)
        elements[:, 4] = mask.astype(int_cp)

        return node_coords_cp, elements

    def build_periodic_node_map(self, node_coords, tol=1e-12):
        """
        Identify which nodes are 'the same' under x/y/z periodicity,
        and unify them. Returns global_id[node] = compressed DOF index.
        
        Parameters:
        -----------
        node_coords : np.ndarray or cp.ndarray
            Array of node coordinates
        tol : float
            Tolerance for considering nodes as the same
            
        Returns:
        --------
        global_id : np.ndarray
            Array mapping from node index to global DOF index
        """
        if hasattr(node_coords, "get"):
            coords = node_coords.get()
        else:
            coords = np.asarray(node_coords)

        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        
        def wrap_dim(val, v0, v1):
            if abs(val - v1) < tol:
                return float(v0)
            return float(val)

        canon_dict = {}
        global_id = np.zeros(coords.shape[0], dtype=int_np)
        next_id = 0

        for n in range(coords.shape[0]):
            xx = wrap_dim(coords[n, 0], min_coords[0], max_coords[0])
            yy = wrap_dim(coords[n, 1], min_coords[1], max_coords[1])
            zz = wrap_dim(coords[n, 2], min_coords[2], max_coords[2])
            key = (round(xx, 12), round(yy, 12), round(zz, 12))
            if key not in canon_dict:
                canon_dict[key] = next_id
                next_id += 1
            global_id[n] = canon_dict[key]

        print(f"Periodic node mapping: {coords.shape[0]} nodes -> {len(np.unique(global_id))} unique DOFs")
        return global_id

    def write_mesh_to_paraview(self, filename="tetrahedral_mesh.vtu"):
        """
        Write the mesh to a .vtu file for Paraview visualization.
        
        Parameters:
        -----------
        filename : str
            Output VTU filename
        """
        import meshio

        node_coords_cp, elements_cp = self.std_fem_mesh()

        node_coords = node_coords_cp.get()
        elements = elements_cp.get()

        tetra_cells = elements[:, :4].astype(int_np)
        defect_flags = elements[:, 4].astype(int_np)

        mesh = meshio.Mesh(
            points=node_coords,
            cells=[("tetra", tetra_cells)],
            cell_data={"defect_flag": [defect_flags]}
        )

        meshio.write(filename, mesh)
        print(f"VTU file successfully written: {filename}")


