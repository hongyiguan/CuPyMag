# Copyright (c) 2025 Hongyi Guan
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

# ex4.py: Solve 3D mechanical equilibrium with cubic symmetry and linear strains,
#         and visualize the strains.

from cupymag.core.assemble_simulation_operators import SimulationOperators
from cupymag.solvers.linear_solvers import solve_cg
import cupy as cp
import cupyx.scipy.sparse as cpx

# Set up the FEM system
SimOp = SimulationOperators()
nDOF = SimOp.n_dof

# Generate a ramdom field
# Force field has 3N DOFs [fx|fy|fz]
f = cp.random.rand(3*nDOF)

# Assemble Linear operators and solve
# A_el: stiffness matrix, M_el: mass matrix, b_el: RHS vector
_, _, M = SimOp.csr_GS()  # M: mass matrix for scalar field

# For PBC, we need to ensure f to have 0 M-weighted mean
fx, fy, fz = f[:nDOF].copy(), f[nDOF:2*nDOF].copy(), f[2*nDOF:].copy()
one = cp.ones(nDOF)

M1 = M @ one
den = float(one @ M1)
def demean_M(g):
    mu = float(one @ (M @ g)) / den
    return g - mu*one

fx = demean_M(fx); fy = demean_M(fy); fz = demean_M(fz)
f = cp.concatenate([fx, fy, fz])

I3 = cpx.identity(3, format='csr')
M_el = cpx.kron(I3, M, format='csr')
A_el, _ = SimOp.csr_elasticity()
b_el = -M_el @ f
b_el[0] = 0.0  # Pin 3 DOFs for mechanical equilibrium with PBC
b_el[nDOF] = 0.0
b_el[2*nDOF] = 0.0
u_disp = solve_cg(A_el, b_el)

# Obtain strain
_, _, _, _, Deriv = SimOp.csr_demag_and_deriv()
E = Deriv.compute_E_from_u(nDOF, u_disp)

# Visualization
Avg = SimOp.volume_average()
Avg.write_to_paraview({"E11": E[:, 0], "E22": E[:, 1], "E33": E[:, 2], "E12": E[:, 3],
                           "E23": E[:, 4], "E13": E[:, 5]}, "strains.vtu", 0.05)