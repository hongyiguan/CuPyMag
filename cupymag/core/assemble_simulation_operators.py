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

from cupymag.mesh.setup_FEM_mesh import FEMMesh
from cupymag.core.parameters import alpha, alpha_damp, ME
from cupymag.core.parameters import c11, c12, c44, lambda100, lambda111
from cupymag.physics.assemble_demag import AssembleDemag
from cupymag.physics.assemble_Gauss_Seidel import AssembleGaussSeidel
from cupymag.physics.assemble_elasticity import AssembleElasticity
from cupymag.utils.final_assembly import *
from cupymag.utils.compute_derivatives import ComputeDerivatives
from cupymag.utils.volume_average import VolumeAverage


class SimulationOperators(FEMMesh):
    """
    Assemble and return all the operators used throughout the simulation.
    Including:
    1. Linear operators for magnetostatic equilibrium system
    2. Linear operators for Gauss-Seidel projection method (GSPM) system
    3. Linear operators for mechanical (elastic) equilibrium system
    4. Linear operators for spatial derivatives
    5. Volume average operators
    """
    def __init__(self):
        super().__init__()
        self.alpha1 = alpha
        self.alpha2 = alpha * alpha_damp

    def csr_demag_and_deriv(self):
        """
        Use the COO sparse format data from AssembleDemag class to assemble final
        CSR cupy matrices.

        Returns
        -------
          A_demag : the global stiffness matrix.
          Fx, Fy, Fz: the global mass matrices, also used as spacial derivatives.
        """
        DemagAssembler = AssembleDemag(self.node_coords_np, self.elements_np, self.global_id_np)
        rows_np, cols_np, vals_np = DemagAssembler.build_coo_matrix_A_numba()
        A_demag = assemble_stiffness_matrix(rows_np, cols_np, vals_np, nDOFx=self.n_dof)

        (Fx_r, Fx_c, Fx_v,
         Fy_r, Fy_c, Fy_v,
         Fz_r, Fz_c, Fz_v) = DemagAssembler.build_coo_matrices_F_numba()
        Fx = assemble_mass_matrix(Fx_r, Fx_c, Fx_v, nDOFx=self.n_dof)
        Fy = assemble_mass_matrix(Fy_r, Fy_c, Fy_v, nDOFx=self.n_dof)
        Fz = assemble_mass_matrix(Fz_r, Fz_c, Fz_v, nDOFx=self.n_dof)

        Deriv = ComputeDerivatives(Fx, Fy, Fz)

        return A_demag, Fx, Fy, Fz, Deriv

    def csr_GS(self):
        """
        Use the COO sparse format data from AssembleGaussSeidel class to assemble final
        CSR cupy matrices.

        Returns
        -------
          A1_GS : the global stiffness matrix for the equation alpha = alpha1
          A2_GS : the global stiffness matrix for the equation alpha = alpha2
          F_GS : the global mass matrix
        """
        GSAssembler = AssembleGaussSeidel(self.node_coords_np, self.elements_np, self.global_id_np)
        rows_np, cols_np, vals_np, Fx_r, Fx_c, Fx_v = GSAssembler.build_coo_matrices_numba()

        A1_GS = assemble_stiffness_matrix(rows_np, cols_np, vals_np * self.alpha1 + Fx_v, nDOFx=self.n_dof, defect_dofs=self.DefDOF)
        A2_GS = assemble_stiffness_matrix(rows_np, cols_np, vals_np * self.alpha2 + Fx_v, nDOFx=self.n_dof, defect_dofs=self.DefDOF)
        F_GS = assemble_mass_matrix(Fx_r, Fx_c, Fx_v, nDOFx=self.n_dof, defect_dofs=self.DefDOF)

        return A1_GS, A2_GS, F_GS

    def csr_elasticity(self):
        """
        Use the COO sparse format data from AssembleElasticity class to assemble final
        CSR cupy matrices.

        Returns
        -------
          A_el : the global stiffness matrix
          F_el : the global mass matrix
        """
        if not ME:
            return None, None

        ElasticityAssembler = AssembleElasticity(self.node_coords_np, self.elements_np, self.global_id_np,
                                                 c11, c12, c44, lambda100, lambda111)
        rows_np, cols_np, vals_np = ElasticityAssembler.build_coo_matrix_A_numba()
        Fx_r, Fx_c, Fx_v = ElasticityAssembler.build_coo_matrices_F_numba()
        A_el = assemble_stiffness_matrix(rows_np, cols_np, vals_np, nDOFx=self.n_dof * 3)
        F_el = assemble_mass_matrix(Fx_r, Fx_c, Fx_v, nDOFx=self.n_dof * 3, nDOFy=self.n_dof * 6)

        return A_el, F_el

    def volume_average(self):
        """
        Precompute the Jacobians and shape functions with VolumeAverage class

        Returns
        -------
          Avg : A configured instance of the VolumeAverage class
        """
        Avg = VolumeAverage(self.node_coords_cp, self.elements_cp, self.global_id_cp)
        return Avg