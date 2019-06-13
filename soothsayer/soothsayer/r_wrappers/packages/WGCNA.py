# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time, multiprocessing

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
wgcna = R_package_retrieve("WGCNA")

def TOMsimilarity(adjacency,  TOMType="unsigned",  TOMDenom="min"):
    """
    WGCNA: TOMsimilarity
        function (adjMat, TOMType = "unsigned", TOMDenom = "min", verbose = 1,
                   indent = 0)
    """
    assert np.all(adjacency.index == adjacency.columns), "pd.DataFrame must have symmetric labels for index and columns"
    nodes = adjacency.index
    rDF_adj = pandas2ri.py2ri(adjacency)
    rDF_tom = wgcna.TOMsimilarity(R["as.matrix"](rDF_adj), TOMType=TOMType, TOMDenom=TOMDenom, verbose=0)
    Ar_tom = pandas2ri.ri2py(rDF_tom)
    return pd.DataFrame(Ar_tom, index=nodes, columns=nodes)

def bicor(X, axis=1, n_jobs=-1):
    """
    WGCNA: bicor
        function (x, y = NULL, robustX = TRUE, robustY = TRUE, use = "all.obs",
                   maxPOutliers = 1, qu <...> dian absolute deviation, or zero variance."))
    """
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if axis == 1: X_copy = X.copy()
    if axis == 0: X_copy = X.copy().T
    labels = X_copy.columns
    rDF_sim = wgcna.bicor(pandas2ri.py2ri(X_copy))
    df_bicor = pd.DataFrame(pandas2ri.ri2py(rDF_sim), index=labels, columns=labels)
    return df_bicor

def pickSoftThreshold_fromSimilarity(df_adj, query_powers):
    # Run pickSoftThreshold.fromSimilarity
    rDF_scalefreetopology = wgcna.pickSoftThreshold_fromSimilarity(R["as.matrix"](pandas2ri.py2ri(df_adj)), powerVector = query_powers, verbose=0, moreNetworkConcepts=True)[1]
    return pandas2ri.ri2py(rDF_scalefreetopology)
