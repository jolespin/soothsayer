# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time
from collections import OrderedDict
import pandas as pd
import numpy as np
# import xarray as xr

# Soothsayer
from ..r_wrappers import *
from soothsayer.utils import assert_acceptable_arguments, check_packages, is_dict

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

# ======================================================================
# Functions
# ==============================================================================
@check_packages(["ALDEx2"], language="r", import_into_backend=False)
def run_aldex2(X:pd.DataFrame, y:pd.Series, reference_class=None, into=pd.DataFrame, aldex2_kws=dict(), random_state=0):
    # Wrappers        
    def _run(X, y, kws):
        r_X = pandas_to_rpy2(X.T)
        r_y = pandas_to_rpy2(y)
        results = aldex2.aldex(
            reads=r_X, 
            conditions=r_y,
            **kws,
        )
        return rpy2_to_pandas(results)
    
    # Assertions
    return_dict = is_dict(into())
    return_dataframe = isinstance(into(), pd.DataFrame)
    assert any([return_dict, return_dataframe]), "`into` must be either dict-type or pd.DataFrame"
    assert np.all(X.shape[0] == y.size), "X.shape[0] != y.size"
    assert np.all(X.index == y.index), "X.index != y.index"
    
    # ALDEx2
    ro.r('set.seed')(random_state)
    
    # Package
    aldex2 = R_package_retrieve("ALDEx2")
    _aldex2_kws=dict(test="t", effect=True, mc_samples=128, denom="all")
    _aldex2_kws.update(aldex2_kws)

    # Multiclass
    classes = set(y.unique())
    if len(classes) > 2:
        assert reference_class is not None, "Please provided a `reference_class` control condition"
        assert reference_class in classes, "`reference_class={}` is not in `y`".format(reference_class)

        multiclass_results = OrderedDict()
        for query_class in sorted(classes - {reference_class}):

            # Subset the samples to include `query_class` and `reference_class`
            y_subset = y[y.map(lambda id_sample: id_sample in {query_class, reference_class})]
            X_subset = X.loc[y_subset.index]

            # Run ALDEx2
            multiclass_results[query_class] = _run(X=X_subset, y=y_subset, kws=_aldex2_kws)

        # Return a dictionary object {query_class:results}
        if return_dict:
            return multiclass_results
        
        # Return a multiindex pd.DataFrame
        if return_dataframe:
            dataframes = list()
            for id_class, df in multiclass_results.items():
                df.columns = df.columns.map(lambda metric: (id_class, reference_class, metric))
                df.columns.names = ["Treatment", "Reference", "Metric"]
                dataframes.append(df)
            return pd.concat(dataframes, axis=1)
            
    # 2 Classes
    else:
        return _run(X=X, y=y, kws=_aldex2_kws)