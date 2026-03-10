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


class ComputeDerivatives:
    """
    A class for computing derivatives and demagnetization fields in micromagnetic simulations.

    This class handles the computation of strain fields and demagnetization fields in FEM.

    Attributes
    ----------
        Dx: (n × n) cupy csr matrix, derivative‐matrix w.r.t. x (-Fx)
        Dy: (n × n) cupy csr matrix, derivative‐matrix w.r.t. y (-Fy)
        Dz: (n × n) cupy csr matrix, derivative‐matrix w.r.t. z (-Fz)
    """
    def __init__(self, Fx, Fy, Fz):
        self.Dx = Fx
        self.Dy = Fy
        self.Dz = Fz

    def compute_Hd_from_U(self, U):
        """
        Compute demagnetization field Hd from demagnetization potential U

        Parameters
        ----------
            U    : Demagnetization potential

        Returns
        -------
            Hd1 : (n, ) cupy array, demagnetization field x component
            Hd2 : (n, ) cupy array, demagnetization field y component
            Hd3 : (n, ) cupy array, demagnetization field z component
        """
        Dx = self.Dx
        Dy = self.Dy
        Dz = self.Dz

        Hd1 = -Dx @ U
        Hd2 = -Dy @ U
        Hd3 = -Dz @ U

        return Hd1, Hd2, Hd3

    def compute_E_from_u(self, n, u, R=None):
        """
        Compute strain from displacement u, optionally rotating derivative operators by R
        without ever forming new n×n matrices.

        Parameters
        ----------
            n    : int, nDOF (number of scalar DOFs per coordinate)
            u    : (3·n) cupy vector, concatenation [u_x; u_y; u_z], already in the x‐basis
            R    : optional (3 × 3) cupy array, rotation from x' -> x.
                   If R is None, no rotation is done and we fall back to the original Dx,Dy,Dz.

        Returns
        -------
            E : (n, 6) cupy array of Voigt‐strains at each DOF in the x‐basis,
                with ordering [E11, E22, E33, E12, E23, E13].
        """
        Dx = self.Dx
        Dy = self.Dy
        Dz = self.Dz

        u_x = u[0: n]
        u_y = u[n: 2 * n]
        u_z = u[2 * n: 3 * n]

        if R is None:
            E11 = Dx @ u_x
            E22 = Dy @ u_y
            E33 = Dz @ u_z

            E12 = Dy @ u_x + Dx @ u_y
            E23 = Dz @ u_y + Dy @ u_z
            E13 = Dz @ u_x + Dx @ u_z

        else:
            dxx = Dx @ u_x
            dyx = Dy @ u_x
            dzx = Dz @ u_x

            dxy = Dx @ u_y
            dyy = Dy @ u_y
            dzy = Dz @ u_y

            dxz = Dx @ u_z
            dyz = Dy @ u_z
            dzz = Dz @ u_z

            E11 = (R[0, 0] * dxx) + (R[1, 0] * dyx) + (R[2, 0] * dzx)

            E22 = (R[0, 1] * dxy) + (R[1, 1] * dyy) + (R[2, 1] * dzy)

            E33 = (R[0, 2] * dxz) + (R[1, 2] * dyz) + (R[2, 2] * dzz)

            tmp1 = (R[0, 1] * dxx) + (R[1, 1] * dyx) + (R[2, 1] * dzx)
            tmp2 = (R[0, 0] * dxy) + (R[1, 0] * dyy) + (R[2, 0] * dzy)
            E12 = tmp1 + tmp2

            tmp3 = (R[0, 2] * dxy) + (R[1, 2] * dyy) + (R[2, 2] * dzy)
            tmp4 = (R[0, 1] * dxz) + (R[1, 1] * dyz) + (R[2, 1] * dzz)
            E23 = tmp3 + tmp4

            tmp5 = (R[0, 2] * dxx) + (R[1, 2] * dyx) + (R[2, 2] * dzx)
            tmp6 = (R[0, 0] * dxz) + (R[1, 0] * dyz) + (R[2, 0] * dzz)
            E13 = tmp5 + tmp6

        return cp.stack([E11, E22, E33, E12, E23, E13], axis=-1)