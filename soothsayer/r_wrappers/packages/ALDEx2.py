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
from soothsayer.utils import assert_acceptable_arguments, check_packages, is_dict, Suppress, format_header

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
def run_aldex2(X:pd.DataFrame, y:pd.Series, reference_class, into=pd.DataFrame, aldex2_kws=dict(), random_state=0, show_console=False):
    """
    Note, this implementation differs ever so slightly.  To guarantee specific directionalities of differences relative to a specified reference,
    this implementation first relabels the reference class 'Reference' and the treatment class 'Treatment' (only 2 classes at a time).  The actual
    ALDEx2 package sorts the classes in alphabetical order and uses the earlier class as the reference.  After running the algorithm, the original
    labels are restored.  I've noticed that there is a slight difference in `diff.btw`, `diff.win`, `effect`, and `overlap` but the `*.ep` and `*.eBH`
    values are the same.

    Relevant GitHub Issue: 
    https://github.com/ggloor/ALDEx2_dev/issues/21

    Please refer to documentation:
    https://bioconductor.org/packages/release/bioc/manuals/ALDEx2/man/ALDEx2.pdf

    # aldex.effect
    rab.all: 
        a vector containing the median clr value for each feature
    rab.win.conditionA: 
        a vector containing the median clr value for each feature in condition A
    rab.win.conditionB:
        a vector containing the median clr value for each feature in condition B
    diff.btw:
        a vector containing the per-feature median difference between condition A and B
    diff.win:
        a vector containing the per-feature maximum median difference between Dirichlet instances within conditions
    effect a vector containing the per-feature effect size
    overlap a vector containing the per-feature proportion of effect size that is 0 or less
    
    # aldex.kw
    kw.ep: 
        a vector containing the expected p-value of the Kruskal-Wallis test for each feature
    kw.eBH:
        a vector containing the corresponding expected value of the Benjamini-Hochberg
    corrected p-value for each feature
    glm.ep:
        a vector containing the expected p-value of the glm ANOVA for each feature
    glm.eBH:
        a vector containing the corresponding expected value of the Benjamini-Hochberg
    corrected p-value for each feature
    
    # aldex.ttest 
    we.ep:
        a vector containing the expected p-value of Welch’s t-test for each feature
    we.eBH:
        a vector containing the corresponding expected value of the Benjamini-Hochberg
    corrected p-value for each feature
    wi.ep:
        a vector containing the expected p-value of the Wilcoxon Rank Sum test for
    each feature
    wi.eBH:
        a vector containing the corresponding expected value of the Benjamini-Hochberg
    corrected p-value for each feature
    """

    # Wrappers        
    def _run(X, y, reference_class, kws):
        assert y.nunique() == 2, "Must have 2 classes: {}".format(list(y.unique()))
        # Encode classes to keep reference condition directionality
        encode = dict()
        decode = dict()
        for y_i in y:
            if y_i == reference_class:
                encode[y_i] = "Reference"
                decode["Reference"] = y_i
            else:
                encode[y_i] = "Treatment"
                decode["Treatment"] = y_i
        y = y.map(lambda y_i: encode[y_i])
        
        # Informative Labels
#         informative_labels = {
#             "rab.all":"median_clr(all)",
#             "rab.win.Reference":"median_clr({})".format(decode["Reference"]),
#             "rab.win.Treatment":"median_clr({})".format(decode["Treatment"]),
#             "diff.btw":
#             "diff.win":
#             "effect":
#             "overlap":
#             "kw.ep": 
#             "kw.eBH":
#             "glm.ep":
#             "glm.eBH":
#             "aldex.ttest":
#             "we.ep":
#             "we.eBH":
#             "wi.ep":
#             "wi.eBH":
#     }
        
        # Run ALDEx2
        r_X = pandas_to_rpy2(X.T)
        r_y = pandas_to_rpy2(y)
        with Suppress(show_stdout=show_console, show_stderr=show_console):
            header = format_header("ALDEx2: Reference='{}', Treatment='{}'".format(decode["Reference"], decode["Treatment"]))
            print(header, file=sys.stderr)
            results = aldex2.aldex(
                reads=r_X, 
                conditions=r_y,
                **kws,
            )
            print(file=sys.stderr)
        df_output = rpy2_to_pandas(results)
        
        # Decode classes to keep reference condition directionality
        columns_decoded = list()
        for label in df_output.columns:
            if any([
                "Reference" in label,
                "Treatment" in label,
            ]):
                if "Reference" in label:
                    label = label.replace("Reference", decode["Reference"])
                else:
                    label = label.replace("Treatment", decode["Treatment"])
            columns_decoded.append(label)
        df_output.columns = columns_decoded
        df_output.index.name = "ComponentID"
        return df_output
    
    # Assertions
    classes = set(y.unique())
    return_dict = is_dict(into())
    return_dataframe = isinstance(into(), pd.DataFrame)
    assert any([return_dict, return_dataframe]), "`into` must be either dict-type or pd.DataFrame"
    assert np.all(X.shape[0] == y.size), "X.shape[0] != y.size"
    assert np.all(X.index == y.index), "X.index != y.index"
    assert reference_class is not None, "Please provided a `reference_class` control condition"
    assert reference_class in classes, "`reference_class={}` is not in `y`".format(reference_class)
    
    # ALDEx2
    ro.r('set.seed')(random_state)
    
    # Package
    aldex2 = R_package_retrieve("ALDEx2")
    _aldex2_kws=dict(test="t", effect=True, mc_samples=128, denom="all")
    _aldex2_kws.update(aldex2_kws)

    # Multiclass
    

    if len(classes) > 2:
        multiclass_results = OrderedDict()
        for query_class in sorted(classes - {reference_class}):

            # Subset the samples to include `query_class` and `reference_class`
            y_subset = y[y.map(lambda id_sample: id_sample in {query_class, reference_class})]
            X_subset = X.loc[y_subset.index]

            # Run ALDEx2
            multiclass_results[query_class] = _run(X=X_subset, y=y_subset, reference_class=reference_class, kws=_aldex2_kws)

        # Return a dictionary object {query_class:results}
        if return_dict:
            return multiclass_results
        
        # Return a multiindex pd.DataFrame
        if return_dataframe:
            dataframes = list()
            for id_class, df in multiclass_results.items():
                df.columns = df.columns.map(lambda metric: (reference_class, id_class,  metric))
                df.columns.names = ["Reference", "Treatment",  "Metric"]
                dataframes.append(df)
            return pd.concat(dataframes, axis=1)
            
    # 2 Classes
    else:
        return _run(X=X, y=y, reference_class=reference_class, kws=_aldex2_kws)