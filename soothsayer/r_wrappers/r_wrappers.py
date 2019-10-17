# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time, multiprocessing
# ==============================================================================
# R Imports
# ==============================================================================
from rpy2 import robjects, rinterface
from rpy2.robjects.packages import importr
try:
    from rpy2.rinterface import RRuntimeError
except ImportError:
    from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
pandas2ri.activate()
R = robjects.r
NULL = robjects.rinterface.NULL
rinterface.set_writeconsole_regular(None)
# ==============================================================================
# All
# ==============================================================================



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
