# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time

import pandas as pd
import numpy as np

# Soothsayer
from ..r_wrappers import *
from soothsayer.utils import *
# ==============================================================================
# R Imports
# ==============================================================================
from rpy2 import robjects, rinterface
from rpy2.robjects.packages import importr
from rpy2.rinterface import RRuntimeError
from rpy2.robjects import pandas2ri
pandas2ri.activate()
R = robjects.r
NULL = robjects.rinterface.NULL
rinterface.set_writeconsole_regular(None)

# R packages
dynamicTreeCut = R_package_retrieve("dynamicTreeCut")
fastcluster = R_package_retrieve("fastcluster")
# ==============================================================================
# Exports
# ==============================================================================
__all__ = ["cutree_dynamic"]
# ==============================================================================
# dynamicTreeCut
# ==============================================================================
def cutree_dynamic(kernel, cut_method="hybrid", method="ward", minClusterSize=20, name=None, **args):
    """
    dynamicTreeCut: cutreeDynamic
        function (dendro, cutHeight = NULL, minClusterSize = 20, method = "hybrid",
                   distM = NULL, de <...> it = deepSplit, minModuleSize = minClusterSize))
    """
    accepted_kernel_types = {"DataFrame", "Symmetric"}
    assert is_query_class(kernel, query=accepted_kernel_types), f"`kernel` type must be one of the following: {accepted_kernel_types}"
    # pd.DataFrame -> Symmetric
    if method == "ward":
        method = "ward.D2"
    if is_query_class(kernel, "Symmetric"):
        kernel = kernel.to_dense()
    if isinstance(kernel, pd.DataFrame):
        rkernel = pandas2ri.py2ri(kernel)

    Z = fastcluster.hclust(R["as.dist"](rkernel), method=method)
    treecut_output = dynamicTreeCut.cutreeDynamic(dendro=Z, method=cut_method, distM=rkernel, minClusterSize = minClusterSize, **args)
    Se_treecut = pd.Series(pandas2ri.ri2py(treecut_output), index=kernel.index, name=name).astype(int)
    return (Se_treecut - 1).sort_values() # Make outliers -1 and largest cluster 0
