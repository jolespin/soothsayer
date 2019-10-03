# ==============
# Soothsayer
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
# Google Scholar: https://scholar.google.com/citations?user=r9y1tTQAAAAJ&hl
# =======
# License BSD-3
# =======
# https://opensource.org/licenses/BSD-3-Clause
#
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
import datetime
__version__= "2019.10"
#datetime.datetime.utcnow().strftime("%Y.%m")
__version_specific__ = "2019.10.02" #datetime.datetime.utcnow().strftime("%Y.%m.%d")
__author__ = "Josh L. Espinoza"
__email__ = "jespinoz@jcvi.org, jol.espinoz@gmail.com"
__url__ = "https://github.com/jolespin/soothsayer"
__cite__ = "https://github.com/jolespin/soothsayer"
__license__ = "BSD-3"
__developmental__ = True

# =======
# Shortcuts
# =======
# Core
from .core import core, Dataset
# Classification
from .classification import classification
# Database
from .db import db
# Feature Extraction
from .feature_extraction import feature_extraction
# Hierarchy
from .hierarchy import hierarchy, Agglomerative, Topology
# I/O
from .io import io
# Microbiome
from .microbiome import microbiome
# Networks
from .networks import networks, Hive, TemporalGraph
# Ordination
from .ordination import ordination, PrincipalComponentAnalysis, PrincipalCoordinatesAnalysis, Manifold
# Regression
from .regression import regression
# Statistics
from .statistics import statistics
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
_submodules = ["core", "classification", "db", "feature_extraction", "hierarchy", "io", "microbiome",  "networks", "ordination", "regression", "statistics", "symmetry", "transmute", "tree", "utils", "visuals"]
_core = ["Dataset"]
_hierarchy = ["Agglomerative", "Topology"]
_networks = ["Hive", "TemporalGraph"]
_ordination = ["PrincipalComponentAnalysis", "PrincipalCoordinatesAnalysis", "Manifold"]
_symmetry = ["Symmetric"]
_utils = ["Chromatic", "Command"]

__all__ = _submodules + _core + _hierarchy + _networks + _ordination + _symmetry + _utils
__all__ = sorted(__all__)
