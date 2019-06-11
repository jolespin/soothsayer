# ==============================================================================
# Imports
# ==============================================================================
# Built-ins
import os, sys, time

import pandas as pd

# Soothsayer
from ..r_wrappers import *
from soothsayer.symmetry import *
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
edgeR = R_package_retrieve("edgeR")

# Normalize using various methods
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
    rX = pandas2ri.py2ri(X)
    d = edgeR.DGEList(counts=rX)

    # Calculate NormFactors
    normalization_factors = edgeR.calcNormFactors(d, method=method, p=p, **kws)

    # Normalized counts
    normalized_counts = edgeR.cpm(normalization_factors)
    X_tmm = pd.DataFrame(pandas2ri.ri2py(normalized_counts), index=idx_attrs, columns=idx_obsvs).T

    return X_tmm
