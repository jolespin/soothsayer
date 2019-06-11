# ==============
# ∆ Soothsayer ∆
# ==============
# High-level API for (bio-)informatics
# ------------------------------------
# GitHub: https://github.com/jolespin/soothsayer
# PyPI: https://pypi.org/project/soothsayer/
# ------------------------------------
# =======
# Contact
# =======
# Producer: Josh L. Espinoza
# Contact: jespinoz@jcvi.org, jol.espinoz@gmail.com
# NCBI Bibliography: https://www.ncbi.nlm.nih.gov/myncbi/browse/collection/50383860/?sort=date&direction=descending
# PDF Downloads: https://github.com/jolespin/publications
# =======
# License
# =======
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
#
# =======
# Version
# =======
__version__= "2019.06"
__author__ = "Josh L. Espinoza"
__email__ = "jespinoz@jcvi.org, jol.espinoz@gmail.com"
__url__ = "https://github.com/jolespin/soothsayer"
__cite__ = "TBD"

# =======
# Shortcuts
# =======
# Core
from .core import Dataset
# Classification
from .classification import classification
# Database
from .db import db
# Feature Extraction
from .feature_extraction import feature_extraction
# Hierarchy
from .hierarchy import Agglomerative, Topology
# I/O
from .io import read_dataframe, read_fasta, write_dataframe, write_fasta, read_object, write_object, read_textfile
# Microbiome
from .microbiome import microbiome
# Networks
from .networks import networks, Hive
# Ordination
from .ordination import PrincipalComponentAnalysis, PrincipalCoordinatesAnalysis, Manifold
# Symmetry
from .symmetry import Symmetric
# Transmute
from .transmute import transmute
# Tree
from .tree import tree
# Utilities
from .utils import utils, Chromatic, Command
# Visualizations
from .visuals import visuals

# =======
# Direct Exports
# =======
_submodules = ["core", "classification", "db", "feature_extraction", "hierarchy", "io", "microbiome", "ordination", "networks", "symmetry", "transmute", "tree", "utils", "visuals"]
_core = ["Dataset"]
_hierarchy = ["Agglomerative", "Topology"]
_io = ["read_dataframe", "read_fasta", "write_dataframe", "write_fasta", "read_object", "write_object", "read_textfile"]
_networks = ["Hive"]
_ordination = ["PrincipalComponentAnalysis", "PrincipalCoordinatesAnalysis", "Manifold"]
_symmetry = ["Symmetric"]
_utils = ["Chromatic", "Command"]

__all__ = _submodules + _core + _hierarchy + _io + _networks + _ordination + _symmetry + _utils
__all__ = sorted(__all__)
