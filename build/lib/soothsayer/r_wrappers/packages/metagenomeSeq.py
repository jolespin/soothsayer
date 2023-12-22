# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time

import pandas as pd

# Soothsayer
from ..r_wrappers import *
# from soothsayer.symmetry import *
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
    R = ro.r
    NULL = ri.NULL
    #rinterface.set_writeconsole_regular(None)

# R packages
# metagenomeSeq = R_package_retrieve("metagenomeSeq")

# CSS Normalization
@check_packages(["metagenomeSeq"], language="r", import_into_backend=False)
def normalize_css(X):
    """
    metagenomeSeq: https://github.com/HCBravoLab/metagenomeSeq
    Tranposes pd.DataFrame object | attributes = columns --> attributes = rows
    Calculates NormFactors via metagenomeSeq
    Scales counts with scaling factors
    """
    metagenomeSeq = R_package_retrieve("metagenomeSeq")

    assert isinstance(X, pd.DataFrame), "type(X) must be pd.DataFrame"
    # metagenomeSeq = R_package_retrieve("metagenomeSeq")
    # Run R-Pipeline
    def run_css(X):
        # Convert pd.DataFrame to R-object
        rX = pandas_to_rpy2(X)
        # Instantiate metagenomeSeq experiment
        obj = metagenomeSeq.newMRexperiment(rX)
        # Calculate NormFactors
        p = metagenomeSeq.cumNormStatFast(obj)
        obj = metagenomeSeq.cumNorm(obj, p=p)#     # Calculate Library Size
        # CSS Normalization
        rDF_css = metagenomeSeq.MRcounts(obj, norm = True, log = False)
        return rpy2_to_pandas(rDF_css)

    # Gene axis as rows
    X = X.T

    # Labels
    idx_attrs = X.index
    idx_obsvs = X.columns

    # Compute CSS
    df_css = pd.DataFrame(run_css(X), index=idx_attrs, columns=idx_obsvs)

    # Adjust attribute axis
    return df_css.T
