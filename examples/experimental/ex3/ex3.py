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

# ex3.py: Solve a 3D Poisson equation and visualize the solution.

from cupymag.core.assemble_simulation_operators import SimulationOperators
import cupy as cp
from cupymag.solvers.linear_solvers import solve_cg

# Set up the FEM system
SimOp = SimulationOperators()
nDOF = SimOp.n_dof

# Generate a ramdom field
f = cp.random.rand(nDOF)

# Assemble Poisson operators and solve
# A: stiffness matrix, M: mass matrix, b: RHS vector
A, _, M = SimOp.csr_GS()
b = M @ f
b[0] = 0.0  # Pin 1 DOF for PBC
x = solve_cg(A, b)

# Visualization
Avg = SimOp.volume_average()
Avg.write_to_paraview({"x":x}, "x_sol.vtu", 0.05)


