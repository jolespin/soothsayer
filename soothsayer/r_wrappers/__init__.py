import sys 
__all__ = list()
# if "rpy2" in sys.modules:
    # from rpy2.robjects.packages import importr
    # from rpy2 import __version__ as rpy2_version
    # rpy2_version_major = int(rpy2_version.split(".")[0])
    # assert rpy2_version_major > 1, "Please update your rpy2 version"
    # if rpy2_version_major == 2:
    #     from rpy2.rinterface import RRuntimeError
    #     importing_error = RRuntimeError
    # if rpy2_version_major == 3:
    #     from rpy2.robjects.packages import PackageNotInstalledError
    #     importing_error = PackageNotInstalledError
__all__ += ["R_package_retrieve", "R_packages", "pandas_to_rpy2", "rpy2_to_pandas"]
from .r_wrappers import *
from .packages import dynamicTreeCut
from .packages import WGCNA
from .packages import philr
from .packages import edgeR
from .packages import metagenomeSeq
from .packages import propr

__all__ += ["dynamicTreeCut", "WGCNA", "philr", "edgeR", "metagenomeSeq" ,"propr"]

__all__ = sorted(__all__)
