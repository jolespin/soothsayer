# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time
from collections import namedtuple, OrderedDict
import pandas as pd
import numpy as np
import xarray as xr

# Soothsayer
from ..r_wrappers import *
from soothsayer.utils import assert_acceptable_arguments, check_packages

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

# R Packages
# propr = R_package_retrieve("propr")

# ==============================================================================
# Exports
# ==============================================================================
__all__ = ["propr"]

# ==============================================================================
# Functions
# ==============================================================================
# Proportionality
@check_packages(["propr"], language="r", import_into_backend=False)
def proportionality(X:pd.DataFrame, metric="rho", transformation="clr", symmetrize=False,  permutations=100,  store_counts=False, store_transformed_components=True, name=None, into=OrderedDict, **kwargs):
    """
    https://github.com/tpq/propr
    https://rdrr.io/cran/propr/man/propr.html
    https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004075
    
    X
    A pd.DataFrame. Stores the original "count matrix" input.

    alpha
    A double. Stores the alpha value used for transformation.

    metric
    A character string. The metric used to calculate proportionality.

    ivar
    A vector. The reference used to calculate proportionality.

    logratio
    A data.frame. Stores the transformed "count matrix".

    matrix
    A matrix. Stores the proportionality matrix.

    pairs
    A vector. Indexes the proportional pairs of interest.

    results
    A data.frame. Stores the pairwise propr measurements.

    permutes
    A list. Stores the shuffled transformed "count matrix" instances, used to reproduce permutations of propr.

    fdr
    A data.frame. Stores the FDR cutoffs for propr.
    """
    # Imports
    propr = R_package_retrieve("propr")

    # Assert "into" types
    assert_acceptable_arguments([into], {dict, OrderedDict, namedtuple})
    # Run propr
    obj_propr = propr.propr(pandas_to_rpy2(X), metric=metric, ivar=transformation,  p=permutations, symmetrize=symmetrize,**kwargs )
#     if update_cutoffs:
#         if n_jobs == -1:
#             n_jobs = multiprocessing.cpu_count()
#         # Cutoffs at which to estimate FDR
#         propr.updateCutoffs(obj_propr, cutoff = ro.FloatVector(cutoff), ncores = n_jobs) 
    
    # Free up some space
    results = dict(obj_propr.slots)
    del obj_propr
    
    # Get useful results
    output = OrderedDict()
    
    # Counts
    if store_counts:
        df = rpy2_to_pandas(results['counts'])
        df.index = X.index
        output["counts"] = df
        
    # Proportinoality
    output["proportionality"]  = pd.DataFrame(np.asarray(results["matrix"]), index=X.columns, columns=X.columns)
    
    # Transformed components
    output["transformation"] = results["ivar"]
    if store_transformed_components:
        output["transformed_components"] = results["logratio"]
        
    # Permutations
    data = np.asarray(list(map(np.asarray, results["permutes"])))
    output["permutations"] = xr.DataArray(data=data, dims=["Permutations", "Samples", "Components"], coords={"Permutations":range(permutations), "Samples":X.index, "Components":X.columns})
    
    # FDR 
    output["fdr"] = rpy2_to_pandas(results["fdr"])
    # Output
    if into is namedtuple:
        if name is None:
            name = "Proportionality"
        Proportionality = namedtuple(name, field_names=output.keys())
        return Proportionality(**output)
    else:
        return into(output)