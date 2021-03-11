# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time
import numpy as np
import pandas as pd
from collections import OrderedDict

# Soothsayer
from ..r_wrappers import *
# from soothsayer.symmetry import *
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
    R = ro.r
    NULL = ri.NULL

# R packages
# edgeR = R_package_retrieve("edgeR")

# Normalize using various methods
@check_packages(["edgeR"], language="r", import_into_backend=False)
def normalize_edgeR(X:pd.DataFrame, method:str="tmm", length:pd.Series=None, p=0.75, **kws):
    """
    X: pd.DataFrame where rows are samples and columns are genes

    methods: ("tmm","rle","upperquartile")
        "TMM" is the weighted trimmed mean of M-values (to the reference) proposed by Robinson and Oshlack (2010), where the weights are from the delta method on Binomial data.
        "RLE" is the scaling factor method proposed by Anders and Huber (2010). We call it "relative log expression", as median library is calculated from the geometric mean of all columns and the median ratio of each sample to the median library is taken as the scale factor.
        "upperquartile" is the upper-quartile normalization method of Bullard et al (2010), in which the scale factors are calculated from the 75% quantile of the counts for each library, after removing genes which are zero in all libraries. This idea is generalized here to allow scaling by any quantile of the distributions.
        "GeTMM" Gene length corrected trimmed mean of M-values. Must include gene lengths. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2246-7#MOESM4
    edgeR: http://bioconductor.org/packages/release/bioc/html/edgeR.html

    """
    edgeR = R_package_retrieve("edgeR")

    assert isinstance(X, pd.DataFrame), "type(df_counts) must be pd.DataFrame"
    # Method formatting
    assert_acceptable_arguments(query=[method.lower()], target={"tmm", "rle", "upperquartile", "getmm"})
    if method in {"tmm", "rle"}:
        method = method.upper()

    # Check RPK for GeTMM
    if method.lower() == "getmm":
        assert length is not None, "If GeTMM is chosed as the method then `length` cannot be None.  It must be either a pd.Series of sequences or sequence lengths"
        length = length[X.columns]
        assert length.isnull().sum() == 0, "Not all of the genes in `X.columns` are in `length.index`.  Either use a different normalization or get the missing sequence lengths"
        # If sequences are given then convert to length (assumes CDS and no introns)
        if pd.api.types.is_string_dtype(length):
            length = length.map(len)
        X = X/length
        method = "TMM"

    # Gene axis as rows
    X = X.T
    # Labels
    idx_attrs = X.index
    idx_obsvs = X.columns

    # Convert pd.DataFrame to R-object
    rX = pandas_to_rpy2(X)
    d = edgeR.DGEList(counts=rX)

    # Calculate NormFactors
    normalization_factors = edgeR.calcNormFactors(d, method=method, p=p, **kws)

    # Normalized counts
    normalized_counts = edgeR.cpm(normalization_factors)
    X_tmm = pd.DataFrame(rpy2_to_pandas(normalized_counts), index=idx_attrs, columns=idx_obsvs).T

    return X_tmm


# ======================================================================
# Functions
# ==============================================================================
@check_packages(["edgeR"], language="r", import_into_backend=False)
def run_edger_exact_test(X:pd.DataFrame, y:pd.Series, reference_class, into=pd.DataFrame, edger_kws=dict(), random_state=0, show_console=False, dispersion_type="global"):
    """
    edger_kws:
        "calculate_normalization_factors":True runs edgeR.calcNormFactors
        "estimate_tagwise_dispersion":True runs edgeR.estimateTagwiseDisp
        
    dispersion_type: {global, local}
        If 'global' then dispersion is calculated for all data in `X`.  If dispersion is 'local' then dispersion is calculated for only `reference` and `treatment` groups.
    """
    def _build_model(X,y,kws):
        components = X.columns
        
        # Run edgeR
        r_X = pandas_to_rpy2(X.T)
        r_y = pandas_to_rpy2(y) # r_y = ro.vectors.StrVector(y)
        r_components = ro.vectors.StrVector(components)

        
        # Give DGEList the gene expression table with groups
        model = edgeR.DGEList(
            counts= r_X, 
            genes = r_components, 
            group=r_y,
        )
        
        # Normalize the expression values
        if kws["calculate_normalization_factors"]:
            model = edgeR.calcNormFactors(model)
            
        # Calculate dispersion
        model = edgeR.estimateCommonDisp(model)
        
        # Estimate tagwise dispersion
        if kws["estimate_tagwise_dispersion"]:
            model = edgeR.estimateTagwiseDisp(model)
            
        return model
        
 
    # Wrappers        
    def _run(X, y, reference_class, kws, model=None):
        assert y.nunique() == 2, "Must have 2 classes: {}".format(list(y.unique()))
        treatment_class = y[lambda x: x != reference_class].unique()[0]
        components = X.columns
        
        if model is None:
            model = _build_model(X=X, y=y, kws=kws)

        # Run ExactTest
        with Suppress(show_stdout=show_console, show_stderr=show_console):
            header = format_header("EdgeR (ExactTest): Reference='{}', Treatment='{}'".format(reference_class, treatment_class))
            print(header, file=sys.stderr)

            # Exact Test
            r_exactTest = edgeR.exactTest(model, pair=ro.vectors.StrVector([reference_class, treatment_class]))
            
            # Results
            r_topTags = edgeR.topTags(r_exactTest, n=X.shape[1], sort_by="none")
        # Convert from R to Python
        py_topTags = OrderedDict(zip(r_topTags.names, map(list,list(r_topTags))))["table"]
        df_output = pd.DataFrame(py_topTags[1:], columns=py_topTags[0], index=["logFC", "logCPM", "PValue", "FDR"]).T
        
        try:
            df_output = df_output.loc[components]
        except KeyError:
            pass
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
    assert_acceptable_arguments(dispersion_type, {"global", "local"})
    
    # edgeR
    ro.r('set.seed')(random_state)
    
    # Package
    edgeR = R_package_retrieve("edgeR")
    _edger_kws={
        "calculate_normalization_factors":True,
        "estimate_tagwise_dispersion":True,
    }
    _edger_kws.update(edger_kws)

    # Multiclass
    if len(classes) > 2:
        model = None # Default: dispersion_type == "local".  More concise than elif dispersion_type == "local": model = None
        if dispersion_type == "global":
            model = _build_model(X=X, y=y, kws=_edger_kws)
            
        multiclass_results = OrderedDict()
        for query_class in sorted(classes - {reference_class}):
            # Subset the samples to include `query_class` and `reference_class`
            y_subset = y[y.map(lambda id_sample: id_sample in {query_class, reference_class})]
            X_subset = X.loc[y_subset.index]
            # Run edgeR
            multiclass_results[query_class] = _run(X=X_subset, y=y_subset, reference_class=reference_class, kws=_edger_kws, model=model)

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
        return _run(X=X, y=y, reference_class=reference_class, kws=_edger_kws, model=None)

@check_packages(["edgeR", "limma"], language="r", import_into_backend=False)
def run_edger_glm(X:pd.DataFrame, y:pd.Series, reference_class, design_matrix:pd.DataFrame=None, into=pd.DataFrame, edger_kws=dict(), random_state=0, show_console=False, dispersion_type="global"):
    """
    To Do: 
        * Make compatible with patsy
        * Fix this discrepancy: 
            * AssertionError: All classes in `y` must be columns in `design_matrix` (e.g. pd.get_dummies(y))
            In [40]: design_matrix
            Out[40]:
                    species[setosa]  species[versicolor]  species[virginica]
            iris_0                1.0                  0.0                 0.0
            iris_1                1.0                  0.0                 0.0
            iris_2                1.0                  0.0                 0.0
            iris_3                1.0                  0.0                 0.0
    """
    def _build_model(X, y, design_matrix, kws):
        components = X.columns
        
        # Run edgeR
        r_X = pandas_to_rpy2(X.T)
        r_y = pandas_to_rpy2(y) # r_y = ro.vectors.StrVector(y)
        r_components = ro.vectors.StrVector(components)
        r_design = pandas_to_rpy2(design_matrix)

        # Give DGEList the gene expression table with groups
        model = edgeR.DGEList(
            counts= r_X, 
            genes = r_components, 
            group = r_y,
        )
       
        # Normalize the expression values
        if kws["calculate_normalization_factors"]:
            model = edgeR.calcNormFactors(model)
            
        # Calculate dispersion
        model = edgeR.estimateGLMCommonDisp(model, r_design)
        
        if kws["estimate_glm_trended_dispersion"]:
            model = edgeR.estimateGLMTrendedDisp(model, r_design)
        
        # Estimate tagwise dispersion
        if kws["estimate_glm_tagwise_dispersion"]:
            model = edgeR.estimateGLMTagwiseDisp(model, r_design)
            
        # Fit
        fit = edgeR.glmFit(model, r_design)
        
        return (model, fit)
        
    def _run(X, y, design_matrix, reference_class, model, fit, kws):
        assert y.nunique() == 2, "Must have 2 classes: {}".format(list(y.unique()))
        treatment_class = y[lambda x: x != reference_class].unique()[0]
        components = X.columns
        r_design = pandas_to_rpy2(design_matrix.astype(float))
        
        if all([(model is None),
                (fit is None),
               ]):
            model, fit = _build_model(X=X, y=y, design_matrix=design_matrix, kws=kws)

        # Run ExactTest
        with Suppress(show_stdout=show_console, show_stderr=show_console):
            header = format_header("EdgeR (GLM): Reference='{}', Treatment='{}'".format(reference_class, treatment_class))
            print(header, file=sys.stderr)

            # LRT Test
            r_glmtest = edgeR.glmLRT(fit, contrast=limma.makeContrasts(f"{treatment_class} - {reference_class}", levels=r_design ) )
            
            # Results
            r_topTags = edgeR.topTags(r_glmtest, n=X.shape[1], sort_by="none")
            
        # Convert from R to Python
        py_topTags = OrderedDict(zip(r_topTags.names, map(list,list(r_topTags))))["table"]
        df_output = pd.DataFrame(py_topTags[1:], columns=py_topTags[0], index=["logFC", "logCPM",  "LR", "PValue", "FDR"]).T
        
        try:
            df_output = df_output.loc[components]
        except KeyError:
            pass
        df_output.index.name = "ComponentID"
        return df_output
            
    # Assertions
    classes = set(y.unique())
    return_dict = is_dict(into())
    return_dataframe = isinstance(into(), pd.DataFrame)
    if design_matrix is None:
        design_matrix = pd.get_dummies(y).astype(int)
    assert isinstance(design_matrix, pd.DataFrame)
    # assert set(y.unique()) <= set(design_matrix.columns), "All classes in `y` must be columns in `design_matrix` (e.g. pd.get_dummies(y))"
    assert any([return_dict, return_dataframe]), "`into` must be either dict-type or pd.DataFrame"
    assert np.all(X.shape[0] == y.size), "X.shape[0] != y.size"
    assert np.all(X.index == y.index), "X.index != y.index"
    assert np.all(X.index == design_matrix.index), "X.index != design_matrix.index"
    assert reference_class is not None, "Please provided a `reference_class` control condition"
    assert reference_class in classes, "`reference_class={}` is not in `y`".format(reference_class)
    assert_acceptable_arguments(dispersion_type, {"global", "local"})
    # if dispersion_type == "local":
    #     Exception("dispersion_type == 'local' currently is not supported for edgeR::glmLRT.  Please use `dispersion_type=='global' or subset your data manually")
    warnings.warn("edgeR::glmLRT is experimental at the moment as design matrices from patsy are not supported and currently only one-hot-encoding is supported (e.g. pd.get_dummies)")
    # edgeR
    ro.r('set.seed')(random_state)
    
    # Package
    edgeR = R_package_retrieve("edgeR")
    limma = R_package_retrieve("limma")

    _edger_kws={
        "calculate_normalization_factors":True,
        "estimate_glm_trended_dispersion":True,
        "estimate_glm_tagwise_dispersion":True
    }
    _edger_kws.update(edger_kws)

    # Multiclass
    if len(classes) > 2:
        model = None # Default: dispersion_type == "local".  More concise than elif dispersion_type == "local": model = None
        fit = None
        if dispersion_type == "global":
            model, fit = _build_model(X=X, y=y, design_matrix=design_matrix, kws=_edger_kws)
            
        multiclass_results = OrderedDict()
        for query_class in sorted(classes - {reference_class}):
            classes_not_used = classes - {reference_class, query_class}
            # Subset the samples to include `query_class` and `reference_class`
            y_subset = y[y.map(lambda id_sample: id_sample in {query_class, reference_class})]
            X_subset = X.loc[y_subset.index]
            # Run edgeR
            if dispersion_type == "global":
                dm = design_matrix
            elif dispersion_type == "local":
                dm = design_matrix.loc[y_subset.index].drop(classes_not_used, axis=1) # This part can't be used with patsy design_matrix
            multiclass_results[query_class] = _run(
                X=X_subset,
                y=y_subset, 
                design_matrix=dm, 
                reference_class=reference_class, 
                model=model, 
                fit=fit, 
                kws=_edger_kws,
            )

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
        return _run(X=X, y=y, design_matrix=design_matrix, reference_class=reference_class, model=None, fit=None, kws=_edger_kws)