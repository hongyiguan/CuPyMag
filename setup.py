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

from setuptools import setup, find_packages
from pathlib import Path
import os

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

os.environ['HDF5_MPI'] = 'OFF'

setup(
    name="cupymag",
    version="0.9.1",
    author="Hongyi Guan",
    author_email="hongyi_guan@ucsb.edu",
    description="GPU-Accelerated Micromagnetics Simulation Package using CuPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hongyiguan/CuPyMag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "cupy>=10.0.0",
        "pyyaml>=5.4",
        "numba>=0.53",
        "meshio>=4.3.0",
        "pandas>=1.3.0",
        "h5py>=3.1.0",
    ],
    entry_points={
        "console_scripts": [
            "cupymag=cupymag.cli:main",
        ],
    }
)
