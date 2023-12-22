# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time

import pandas as pd
import numpy as np

# Soothsayer
from ..r_wrappers import *
from soothsayer.utils import check_packages

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
# WGCNA = R_package_retrieve("WGCNA")
#
# def TOMsimilarity(adjacency,  TOMType="unsigned",  TOMDenom="min"):
#     """
#     WGCNA: TOMsimilarity
#         function (adjMat, TOMType = "unsigned", TOMDenom = "min", verbose = 1,
#                    indent = 0)
#     """
#     assert np.all(adjacency.index == adjacency.columns), "pd.DataFrame must have symmetric labels for index and columns"
#     nodes = adjacency.index
#     rDF_adj = pandas_to_rpy2(adjacency)
#     rDF_tom = wgcna.TOMsimilarity(R["as.matrix"](rDF_adj), TOMType=TOMType, TOMDenom=TOMDenom, verbose=0)
#     Ar_tom = rpy2_to_pandas(rDF_tom)
#     return pd.DataFrame(Ar_tom, index=nodes, columns=nodes)
#
# def bicor(X, axis=1, n_jobs=-1):
#     """
#     WGCNA: bicor
#         function (x, y = NULL, robustX = TRUE, robustY = TRUE, use = "all.obs",
#                    maxPOutliers = 1, qu <...> dian absolute deviation, or zero variance."))
#     """
#     if n_jobs == -1:
#         n_jobs = multiprocessing.cpu_count()
#
#     if axis == 1: X_copy = X.copy()
#     if axis == 0: X_copy = X.copy().T
#     labels = X_copy.columns
#     rDF_sim = WGCNA.bicor(pandas_to_rpy2(X_copy))
#     df_bicor = pd.DataFrame(rpy2_to_pandas(rDF_sim), index=labels, columns=labels)
#     return df_bicor

@check_packages(["WGCNA"], language="r", import_into_backend=False)
def pickSoftThreshold_fromSimilarity(df_adj, query_powers):
    WGCNA = R_package_retrieve("WGCNA")

    # Run pickSoftThreshold.fromSimilarity
    query_powers = ro.IntVector(list(query_powers))
    r_adj = pandas_to_rpy2(df_adj)
    rDF_scalefreetopology = WGCNA.pickSoftThreshold_fromSimilarity(R["as.matrix"](r_adj), powerVector = query_powers, verbose=0, moreNetworkConcepts=True)[1]
    return rpy2_to_pandas(rDF_scalefreetopology)
