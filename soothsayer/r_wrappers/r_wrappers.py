# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time, multiprocessing
# ==============================================================================
# R Compatibility
# ==============================================================================
from rpy2 import robjects as ro
from rpy2 import rinterface as ri
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2 import __version__ as rpy2_version
rpy2_version_major = int(rpy2_version.split(".")[0])
rpy2_version_minor = int(rpy2_version.split(".")[1])
# rpy2_version_micro = int(rpy2.__version__[2])
if rpy2_version_major == 2:
    from rpy2.rinterface import RRuntimeError
    pandas2ri.activate()
    ri.set_writeconsole_regular(None) # How do I do this v3?
if rpy2_version_major == 3:
    from rpy2.rinterface_lib.embedded import RRuntimeError
    from rpy2.robjects.conversion import localconverter
assert rpy2_version_major > 1, "Please update your rpy2 version"

R = ro.r
NULL = ri.NULL

# Converters
def pandas_to_rpy2(df, rpy2_version_major=None):
    if rpy2_version_major is None:
        from rpy2 import __version__ as rpy2_version
        rpy2_version_major = int(rpy2_version.split(".")[0])
    if rpy2_version_major == 2: # v2.x
        return pandas2ri.py2ri(df)
    if rpy2_version_major == 3: # v3.x
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.py2rpy(df)
def rpy2_to_pandas(r_df, rpy2_version_major=None):
    if rpy2_version_major is None:
        from rpy2 import __version__ as rpy2_version
        rpy2_version_major = int(rpy2_version.split(".")[0])
    if rpy2_version_major == 2: # v2.x
        return pandas2ri.ri2py(r_df)
    if rpy2_version_major == 3: # v3.x
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.rpy2py(r_df)

# Load R Packages
R_packages = dict()
def R_package_retrieve(package):
    """
    ["dynamicTreeCut", "WGCNA", "fastcluster", "phyloseq", "philr", "ape", "metagenomeSeq", "edgeR"]
    """
    try:
        return R_packages[package]
    except KeyError:
        try:
            R_packages[package] = importr(package)
            if package == "WGCNA":
                R_packages[package].enableWGCNAThreads(multiprocessing.cpu_count())
            return R_packages[package]
        except RRuntimeError:
            print(f"{package} is not available", file=sys.stderr)










#
