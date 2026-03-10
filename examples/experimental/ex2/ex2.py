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

# ex2.py: Compute volume average and spatial derivatives of an arbitrary field.

from cupymag.core.assemble_simulation_operators import SimulationOperators
import cupy as cp

# Set up the FEM system
SimOp = SimulationOperators()
nDOF = SimOp.n_dof

# Generate a ramdom vector field
f = cp.random.rand(nDOF, 3)

# Volume averages
Avg = SimOp.volume_average()
avg_f = Avg.compute_average_field_gpu(f)
print(f"The volume average is {avg_f}.")

# Spatial derivatives
_, _, _, _, Deriv = SimOp.csr_demag_and_deriv()
fx, fy, fz = [-x for x in Deriv.compute_Hd_from_U(f)] # Hd = -\grad U
# fx, fy, fz are spatial derivatives of f with respect to x, y, z
# For visualization (reuse the Gauss points, Jacobians etc. computed in VolumeAverage class)
Avg.write_to_paraview({"f":f, "fx":fx, "fy":fy, "fz":fz}, "f_and_gradf.vtu", 0.05)