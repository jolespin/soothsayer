#!/usr/bin/env python
# =============================
# Creator: Josh L. Espinoza (J. Craig Venter Institute)
# Date[0]:Init: 2018-April-02
# Date[-1]:Current: 2018-August-28
# =============================
# BSD License
# =============================
# Copyright 2018 Josh L. Espinoza
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Version
__version_soothsayer__ = "v1"

# Built-ins
import os, sys, argparse, importlib

# Accepted algorithms
accepted_algorithms = ["clairvoyance"]

# Controller
def main(argv=None):
    parser = argparse.ArgumentParser(prog="soothsayer",description="soothsayer:{}".format(__version_soothsayer__), add_help=True)
    parser.add_argument("algorithm", choices=accepted_algorithms, help="`soothsayer` algorithm for analysis")
    opts = parser.parse_args(argv)
    return opts.algorithm


# Initialize
if __name__ == "__main__":
    # Check version
    python_version = sys.version.split(" ")[0]
    condition_1 = int(python_version.split(".")[0]) == 3
    condition_2 = int(python_version.split(".")[1]) >= 6
    assert all([condition_1, condition_2]), "Python version must be >= 3.6.  You are running: {}\n{}".format(python_version, sys.executable)
    # Get the algorithm
    algorithm = main([sys.argv[1]])
    module = importlib.import_module(algorithm)
    module.main(sys.argv[2:])
