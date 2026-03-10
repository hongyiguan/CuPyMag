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

__version__ = "0.9.1"
__author__ = "Hongyi Guan"
__email__ = "hongyi_guan@ucsb.edu"
__license__ = "Apache 2.0"

try:
    import cupy as cp
    if not cp.cuda.is_available():
        raise RuntimeError("CUDA is not available. CuPyMag requires CUDA-enabled GPU.")
except ImportError:
    raise ImportError("CuPy is required but not installed. Please install cupy-cuda11x or cupy-cuda12x")

