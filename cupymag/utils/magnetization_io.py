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
import h5py
from pathlib import Path


def write_array(array, filename):
    """
    Write CuPy array to HDF5 file.

    Parameters:
    -----------
    array : cupy.ndarray
        Array to write to disk
    filename : str or Path
        Output filename (with or without .h5 extension)
    """
    filename = Path(filename)
    if filename.suffix != '.h5':
        filename = filename.with_suffix('.h5')

    data = cp.asnumpy(array)

    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=data, compression='gzip', compression_opts=6)
        f.attrs['shape'] = data.shape
        f.attrs['dtype'] = str(data.dtype)


def read_array(filename, nDOF):
    """
    Read CuPy array from HDF5 file.

    Parameters:
    -----------
    filename : str or Path
        Input filename (with or without .h5 extension)
    nDOF : int
        Number of degrees of freedom. Ensure that the array has the correct shape (nDOF, 3)

    Returns:
    --------
    cupy.ndarray or None
        Array loaded from disk if shape is (nDOF, 3), otherwise None
    """
    if filename is None:
        print("Initial magnetization file not found, start with uniform magnetization.")
        return None

    filename = Path(filename)
    if filename.suffix != '.h5':
        filename = filename.with_suffix('.h5')

    with h5py.File(filename, 'r') as f:
        data = f['data'][:]

    if data.shape != (nDOF, 3):
        print(f"Initial magnetization shape mismatch: expected ({nDOF}, 3), got {data.shape}.")
        print("Start with uniform magnetization.")
        return None

    array = cp.asarray(data)
    return array


def initialize_m(restart_m, restart, initial_m, nDOF, DefDOF, float_cp):
    """
    Initialize magnetization for simulation

    Parameters:
    -----------
    restart_m : str or Path
        Input filename (with or without .h5 extension)
    restart : bool
        Indicate whether restart from an input magnetization file
    initial_m: array
        The uniform magnetization if restart is false
    nDOF : int
        Number of degrees of freedom.
    DefDOF :  cupy.ndarray
        Sorted array of unique global degree of freedom indices corresponding
        to all nodes within defect elements.
    float_cp : cupy.dtype
        Data type of magnetization

    Returns:
    --------
    m : cupy.ndarray
        Initial magnetization
    """
    m = read_array(restart_m, nDOF) if restart else None
    if m is None:
        if initial_m is None:
            initial_m = cp.array([1.0, 0.0, 0.0])
        else:
            initial_m = cp.array(initial_m)
            norm = cp.linalg.norm(initial_m)
            if norm > 0:
                initial_m /= norm
            else:
                raise ValueError("Initial magnetization vector cannot be zero.")
   
        m = cp.zeros((nDOF, 3), dtype=float_cp)
        m[:, 0] = initial_m[0]
        m[:, 1] = initial_m[1]
        m[:, 2] = initial_m[2] 

    m[DefDOF, :] = 0.0

    return m