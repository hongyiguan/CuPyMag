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

int_np = np.int32
int_cp = cp.int32

class HexGrid:
    """
    A 3D hexahedral finite element mesh generator for cubic domains with embedded defect regions.

    This class generates structured hexahedral meshes on rectangular domains [0, Nx] × [0, Ny] × [0, Nz]
    with the capability to distinguish between two material regions: a cubic defect region
    and the surrounding magnetic material region.

    Parameters
    ----------
    Nx, Ny, Nz : int
        Number of elements along x, y, and z directions respectively.
        The domain extends from 0 to Nx in x, 0 to Ny in y, and 0 to Nz in z.
    Ndx, Ndy, Ndz : int
        Dimensions of the centered defect region in x, y, and z directions respectively.
        The defect region center is set by the user input and is (Nx/2, Ny/2, Nz/2) by default.

    Attributes
    ----------
    Nx, Ny, Nz : int
        Grid dimensions in number of elements.
    Ndx, Ndy, Ndz : int
        Defect region dimensions.

    Methods
    -------
    std_fem_square_mesh()
        Generate node coordinates and element connectivity arrays in standard FEM format.
    write_mesh_to_paraview()
        Export mesh to VTU format for visualization in ParaView.
    build_periodic_node_map(node_coords, tol=1e-12)
        Build mapping for periodic boundary conditions by identifying equivalent nodes.
    """

    def __init__(self, Nx, Ny, Nz, Ndx, Ndy, Ndz, defect_center=None):
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Ndx = Ndx
        self.Ndy = Ndy
        self.Ndz = Ndz
        self.defect_center = defect_center
        self.x = cp.linspace(0, Nx, Nx + 1)
        self.y = cp.linspace(0, Ny, Ny + 1)
        self.z = cp.linspace(0, Nz, Nz + 1)

    def std_fem_mesh(self):
        """
        Generate node coordinates and element connectivity directly.

        Returns
        -------
          node_coords   : (Nnodes, 3) cupy array with 3D coordinates of each node
          elements      : (Nelems, 9) cupy array where elements[e,0..7] = corner node indices
                          and elements[e,8] in {0,1} is the defect flag for that element.
        """
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        Ndx, Ndy, Ndz = self.Ndx, self.Ndy, self.Ndz

        x = self.x
        y = self.y
        z = self.z

        i3, j3, k3 = cp.meshgrid(
            cp.arange(Nx + 1),
            cp.arange(Ny + 1),
            cp.arange(Nz + 1),
            indexing='ij'
        )

        node_coords = cp.stack((
            x[i3.ravel()],
            y[j3.ravel()],
            z[k3.ravel()]
        ), axis=1)

        # Calculate defect boundaries
        if self.defect_center is None:
            cx, cy, cz = Nx / 2.0, Ny / 2.0, Nz / 2.0
        else:
            cx, cy, cz = self.defect_center

        x_min, x_max = cx - Ndx / 2.0, cx + Ndx / 2.0
        y_min, y_max = cy - Ndy / 2.0, cy + Ndy / 2.0
        z_min, z_max = cz - Ndz / 2.0, cz + Ndz / 2.0

        # Generate elements and defect flags
        Ne = Nx * Ny * Nz
        elements = cp.zeros((Ne, 9), dtype=int_cp)

        elem_idx = 0
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    def node_id(ii, jj, kk):
                        return ii * (Ny + 1) * (Nz + 1) + jj * (Nz + 1) + kk

                    elements[elem_idx, 0] = node_id(i, j, k)
                    elements[elem_idx, 1] = node_id(i + 1, j, k)
                    elements[elem_idx, 2] = node_id(i + 1, j + 1, k)
                    elements[elem_idx, 3] = node_id(i, j + 1, k)
                    elements[elem_idx, 4] = node_id(i, j, k + 1)
                    elements[elem_idx, 5] = node_id(i + 1, j, k + 1)
                    elements[elem_idx, 6] = node_id(i + 1, j + 1, k + 1)
                    elements[elem_idx, 7] = node_id(i, j + 1, k + 1)

                    # Check if element center is in defect region
                    xc = i + 0.5
                    yc = j + 0.5
                    zc = k + 0.5

                    is_defect = (
                            (xc >= x_min) and (xc < x_max) and
                            (yc >= y_min) and (yc < y_max) and
                            (zc >= z_min) and (zc < z_max)
                    )

                    elements[elem_idx, 8] = int(is_defect)
                    elem_idx += 1

        return node_coords, elements

    def write_mesh_to_paraview(self, filename="example_3d_mesh.vtu"):
        """
        Write the mesh to a .vtu file for Paraview visualization.

        Parameters
        ----------
        filename : str, optional
            Output filename for the VTU file. Default is "example_3d_mesh.vtu".
        """
        import meshio

        node_coords, elements = self.std_fem_mesh()

        if hasattr(node_coords, 'get'):
            points = node_coords.get()
        else:
            points = np.asarray(node_coords)

        if hasattr(elements, 'get'):
            elements_np = elements.get()
        else:
            elements_np = np.asarray(elements)

        connectivity = elements_np[:, :8]
        defect_flags = elements_np[:, 8]

        mesh = meshio.Mesh(
            points=points,
            cells=[("hexahedron", connectivity)],
            cell_data={"defect_or_mag": [defect_flags]}
        )

        meshio.write(filename, mesh)
        print(f"VTU file successfully written: {filename}")

    def build_periodic_node_map(self, node_coords, tol=1e-12):
        """
        Identify which nodes are 'the same' under x/y/z periodicity,
        and unify them. Returns global_id[node] = compressed DOF index.
        """
        x = self.x.get()
        y = self.y.get()
        z = self.z.get()

        if hasattr(node_coords, "get"):
            coords = node_coords.get()
        else:
            coords = np.asarray(node_coords)

        def wrap_dim(val, v0, v1):
            if abs(val - v1) < tol:
                return float(v0)
            return float(val)

        canon_dict = {}
        global_id = np.zeros(coords.shape[0], dtype=int_np)
        next_id = 0

        for n in range(coords.shape[0]):
            xx = wrap_dim(coords[n, 0], x[0], x[-1])
            yy = wrap_dim(coords[n, 1], y[0], y[-1])
            zz = wrap_dim(coords[n, 2], z[0], z[-1])
            key = (round(xx, 12), round(yy, 12), round(zz, 12))
            if key not in canon_dict:
                canon_dict[key] = next_id
                next_id += 1
            global_id[n] = canon_dict[key]

        print(f"Periodic node mapping: {coords.shape[0]} nodes -> {len(np.unique(global_id))} unique DOFs")
        return global_id
