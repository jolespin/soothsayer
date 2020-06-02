# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys
# ==============================================================================
# R Compatibility
# ==============================================================================
from ..utils import check_packages
# if "rpy2" in sys.modules:
from rpy2 import robjects as ro
from rpy2 import rinterface as ri
from rpy2.robjects import pandas2ri
from rpy2 import __version__ as rpy2_version
rpy2_version_major = int(rpy2_version.split(".")[0])
rpy2_version_minor = int(rpy2_version.split(".")[1])
# rpy2_version_micro = int(rpy2.__version__[2])
assert rpy2_version_major > 1, "Please update your rpy2 version"


# Converters
@check_packages(["rpy2"], language="python", import_into_backend=False)
def pandas_to_rpy2(df, rpy2_version_major=None):
    if rpy2_version_major is None:
        from rpy2 import __version__ as rpy2_version
        rpy2_version_major = int(rpy2_version.split(".")[0])
    if rpy2_version_major == 2: # v2.x
        return pandas2ri.py2ri(df)
    if rpy2_version_major == 3: # v3.x
        from rpy2.robjects.conversion import localconverter
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.py2rpy(df)

@check_packages(["rpy2"], language="python", import_into_backend=False)
def rpy2_to_pandas(r_df, rpy2_version_major=None):
    if rpy2_version_major is None:
        from rpy2 import __version__ as rpy2_version
        rpy2_version_major = int(rpy2_version.split(".")[0])
    if rpy2_version_major == 2: # v2.x
        return pandas2ri.ri2py(r_df)
    if rpy2_version_major == 3: # v3.x
        from rpy2.robjects.conversion import localconverter
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.rpy2py(r_df)

# Load R Packages
R_packages = dict() # Remove this in future versions
@check_packages(["rpy2"], language="python", import_into_backend=False)
def R_package_retrieve(package, not_exists_ok=True):
    """
    ["dynamicTreeCut", "WGCNA", "fastcluster", "phyloseq", "philr", "ape", "metagenomeSeq", "edgeR"]
    """
    from rpy2.robjects.packages import importr
    if rpy2_version_major == 2:
        from rpy2.rinterface import RRuntimeError
        importing_error = RRuntimeError
        pandas2ri.activate()
        ri.set_writeconsole_regular(None) # How do I do this v3?
    if rpy2_version_major == 3:
        # from rpy2.rinterface_lib.embedded import RRuntimeError
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import PackageNotInstalledError
        importing_error = PackageNotInstalledError
    try:
        return R_packages[package]
    except KeyError:
        try:
            R_packages[package] = importr(package)
            return R_packages[package]
        except importing_error:
            msg = f"{package} is not available"
            if not_exists_ok:
                print(msg, file=sys.stderr)
            else:
                raise ImportError(msg)










#
