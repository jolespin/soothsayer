# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time

import pandas as pd
import numpy as np

# Soothsayer
from ..r_wrappers import *
from soothsayer.utils import check_packages, is_query_class
# ==============================================================================
# R Imports
# ==============================================================================
if "rpy2" in sys.modules:
    from rpy2 import robjects as ro
    from rpy2 import rinterface as ri

    from rpy2.robjects.packages import importr
    try:
        from rpy2.rinterface import RRuntimeError
    except ImportError:
        from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects import pandas2ri
    # pandas2ri.activate()
    R = ro.r
    NULL = ri.NULL
    #rinterface.set_writeconsole_regular(None)

# R packages
# dynamicTreeCut = R_package_retrieve("dynamicTreeCut")
# fastcluster = R_package_retrieve("fastcluster")
# ==============================================================================
# Exports
# ==============================================================================
__all__ = ["cutree_dynamic"]
# ==============================================================================
# dynamicTreeCut
# ==============================================================================
@check_packages(["dynamicTreeCut", "fastcluster"], language="r", import_into_backend=False)
def cutree_dynamic(kernel, cut_method="hybrid", method="ward", minClusterSize=20, name=None, **args):
    """
    dynamicTreeCut: cutreeDynamic
        function (dendro, cutHeight = NULL, minClusterSize = 20, method = "hybrid",
                   distM = NULL, de <...> it = deepSplit, minModuleSize = minClusterSize))
    """
    # Imports
    dynamicTreeCut = R_package_retrieve("dynamicTreeCut")
    fastcluster = R_package_retrieve("fastcluster")

    accepted_kernel_types = {"DataFrame", "Symmetric"}
    assert is_query_class(kernel, query=accepted_kernel_types), f"`kernel` type must be one of the following: {accepted_kernel_types}"
    # pd.DataFrame -> Symmetric
    if method == "ward":
        method = "ward.D2"
    if is_query_class(kernel, "Symmetric"):
        kernel = kernel.to_dense()
    if isinstance(kernel, pd.DataFrame):
        rkernel = pandas_to_rpy2(kernel)

    Z = fastcluster.hclust(R["as.dist"](rkernel), method=method)
    treecut_output = dynamicTreeCut.cutreeDynamic(dendro=Z, method=cut_method, distM=rkernel, minClusterSize = minClusterSize, **args)
    Se_treecut = pd.Series(rpy2_to_pandas(treecut_output), index=kernel.index, name=name).astype(int)
    return (Se_treecut - 1).sort_values() # Make outliers -1 and largest cluster 0
