
# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, multiprocessing, itertools, copy, datetime, warnings
from collections import OrderedDict, defaultdict

# PyData
import pandas as pd
import numpy as np
import xarray as xr
import networkx as nx

# Biology
from Bio import SeqIO, Seq
import ete3
import skbio

# SciPy
from scipy import stats
from scipy.spatial import distance
try:
    from fastcluster import linkage
except ImportError:
    from scipy.cluster.hierarchy import linkage
    print("Could not import `linkage` from `fastcluster` and using `scipy.cluster.hierarchy.linkage` instead", file=sys.stderr)

# Statistics and Miscellaneous Machine Learning
from astropy.stats import biweight_midcorrelation, median_absolute_deviation
from sklearn.metrics.pairwise import pairwise_distances

# Soothsayer
from ..transmute.conversion import linkage_to_newick, name_ete_nodes
from ..r_wrappers.packages.WGCNA import bicor
from ..utils import check_symmetry, force_symmetry

__all__ = ["Symmetric", "pairwise", "pairwise_tree_distance", "pairwise_logfc"]
__all__ = sorted(__all__)


# =======================================================
# Pairwise calculations
# =======================================================


# Symmetrical dataframes represented as augment pd.Series
class Symmetric(object):
    """
    An indexable symmetric matrix stored as the lower triangle for space

    devel
    =====
    2018-August-16
    * Added __add__, __sub__, etc.
    * Removed conversion to dissimilarity for tree construction
    * Added .iteritems method

    Future:
    Take in a Symmetric object and pd.Series with a frozenset index
    Fix the diagonal arithmetic
    """
    def __init__(self, X:pd.DataFrame, data_type=None, metric_type=None, func_metric=None, name=None, mode="infer", metadata=dict(), force_the_symmetry=True):
        acceptable_modes = ["similarity", "dissimilarity", "statistical_test", "infer"]
        assert mode in acceptable_modes, f"`mode` must be in {acceptable_modes}"
        if mode == "infer":
            mode = self._infer_mode(X)
        assert np.any(X.isnull()) == False, "Check what is causing NaN and remove them."
        if force_the_symmetry:
            X = force_symmetry(X)
        else:
            assert check_symmetry(X, tol=1e-10), "X is not symmetric"

        self.mode = mode
        self.data_type = data_type
        self.func_metric = func_metric
        if func_metric is not None:
            assert hasattr(func_metric, "__call__"), "`func_metric` isn't a function"
            if metric_type is None:
                metric_type = func_metric.__name__
        self.metric_type = metric_type
        self.name = name
        self.labels = X.index
        self.diagonal = np.copy(np.diagonal(X))

        diagonal_elements = set(self.diagonal)
        warnings.warn('If operations are applied to the Symmetric object they will not be represented by the diagonal when converting back to dense form. Fix this in future version.')
        self.data = self._dense_to_condensed(X)
        self.__synthesized__ = datetime.datetime.utcnow()
        self.num_nodes = X.shape[0]
        # Metadata
        self.metadata = dict([
            ("name",name),
            ("data_type",data_type),
            ("synthesized",self.__synthesized__.strftime("%Y-%m-%d %H:%M:%S")),
            ("num_nodes",self.num_nodes),
            ("mode", mode),
            ("metric_type", metric_type),
        ]
        )
        self.metadata.update(metadata)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.metadata)[1:-1]})"

    # ==
    # QC
    # ==
    def _infer_mode(self, X):
        diagonal = np.diagonal(X)
        diagonal_elements = set(diagonal)
        assert len(diagonal_elements) == 1, "Cannot infer relationships from diagonal because multiple values"
        assert diagonal_elements <= {0,1}, "Diagonal should be either 0.0 for dissimilarity or 1.0 for similarity"
        return {0.0:"dissimilarity", 1.0:"similarity"}[list(diagonal_elements)[0]]

    # =======
    # Built-in
    # =======
    def __getitem__(self, key):
        assert type(key)  not in [int,float,str], "`key` must be a non-string iterable"
        assert len(key) >= 2, "`key` must have at least 2 identifiers. e.g. ('A','B')"
        key = frozenset(key)
        if len(key) == 1:
            return {"similarity":1.0, "dissimilarity":0.0, "statistical_test":0.0}[self.mode]
        else:
            if len(key) > 2:
                key = [*map(frozenset, itertools.combinations(key, r=2))]
            return self.data[key]
    def __len__(self):
        return self.num_nodes
    def __iter__(self):
        for v in self.data:
            yield v
    def iteritems(self):
        return self.data.iteritems()

    # Maths
    def __add__(self, x):
        symmetric_clone = copy.deepcopy(self)
        symmetric_clone.data += x
        # self.diagonal += x
        return symmetric_clone
    def __radd__(self, x):
        self.diagonal += x
        return Symmetric.__add__(self,x)
    def __sub__(self, x):
        symmetric_clone = copy.deepcopy(self)
        symmetric_clone.data -= x
        return symmetric_clone
    def __rsub__(self, x):
        symmetric_clone = copy.deepcopy(self)
        symmetric_clone.data = x - symmetric_clone.data
        return symmetric_clone
    def __mul__(self, x):
        symmetric_clone = copy.deepcopy(self)
        symmetric_clone.data *= x
        return symmetric_clone
    def __rmul__(self, x):
        return Symmetric.__mul__(self,x)
    def __truediv__(self, x):
        symmetric_clone = copy.deepcopy(self)
        symmetric_clone.data /= x
        return symmetric_clone
    def __rtruediv__(self, x):
        symmetric_clone = copy.deepcopy(self)
        symmetric_clone.data = x / symmetric_clone.data
        return symmetric_clone
    def __floordiv__(self, x):
        symmetric_clone = copy.deepcopy(self)
        symmetric_clone.data //= x
        return symmetric_clone
    def __rfloordiv__(self, x):
        symmetric_clone = copy.deepcopy(self)
        symmetric_clone.data = x // symmetric_clone.data
        return symmetric_clone

    # =======
    # Utility
    # =======
    def set_diagonal(self, values):
        self.diagonal = values
        return self

    # ==========
    # Conversion
    # ==========
    def to_dense(self, subset=None, diagonal=None):
        data_dense = distance.squareform(self.data.values)
        if diagonal is None:
            if diagonal == "infer":
                diagonal = {"similarity":1.0, "dissimilarity":0.0, "statistical_test":0.0}[self.mode]
            else:
                diagonal = self.diagonal
        np.fill_diagonal(data_dense, diagonal)
        df_dense = pd.DataFrame(data_dense, index=self.labels, columns=self.labels)
        if subset is None:
            return df_dense
        else:
            # There should be a better way to do this
            return df_dense.loc[subset,subset]

    def to_condensed(self):
        return self.data

    def _dense_to_condensed(self, X):
        np.fill_diagonal(X.values, 0)
        index=pd.Index([*map(frozenset,itertools.combinations(self.labels, 2))], name=self.name)
        data = distance.squareform(X)
        return pd.Series(data, index=index, name=self.name)

    def _condensed_to_dense(self, y):
        y.index = pd.MultiIndex.from_tuples(y.index, names=[None,None])
        X = y.unstack().loc[self.labels,self.labels]
        mask_null = X.isnull().values
        X.values[mask_null] = X.T.values[mask_null]
        return X

    def as_tree(self, method="ward", into=ete3.Tree, node_prefix="y"):
        assert self.mode == "dissimilarity", "mode must be dissimilarity to construct tree"
        if not hasattr(self,"Z"):
            self.Z = linkage(self.data.values,method=method)
        if not hasattr(self,"newick"):
            self.newick = linkage_to_newick(self.Z, self.labels)
        tree = into(newick=self.newick, name=self.name)
        return name_ete_nodes(tree, node_prefix)

    def as_graph(self, graph=None):
        if graph is None:
            graph = nx.Graph()
        graph.name = self.name
        for (node_A, node_B), weight in self.data.iteritems():
            graph.add_edge(node_A, node_B, weight=weight)
        return graph

    def copy(self):
        return copy.deepcopy(self)

# Pairwise interactions
def pairwise(X, metric="euclidean", axis=1, name=None, into=pd.DataFrame, mode="infer", signed=True, n_jobs=-1, check_metrics=True, symmetric_kws=dict()):
    """
    Compute pairwise interactions

    01-August-2018
    """

    def get_data_object(df_dense, into, mode, func_signed, metric, symmetric_kws, CORRELATION_METRICS, DISTANCE_METRICS, name, check_metrics):
        if hasattr(metric, "__call__"):
            metric_name = metric.__name__
        else:
            metric_name = metric

        if mode == "similarity":
            if check_metrics:
                assert metric_name not in DISTANCE_METRICS, f"Cannot compute simlilarity for {metric.__name__}"

        if mode == "dissimilarity":
            if metric_name in CORRELATION_METRICS:
                df_dense = 1 - func_signed(df_dense)
        if into in [Symmetric, pd.Series]:
            kernel = Symmetric(df_dense, mode=mode, name=name, **symmetric_kws)
            if into == Symmetric:
                return kernel
            if into == pd.Series:
                return kernel.data
        else:
            return df_dense


    fill_diagonal="auto"
    CORRELATION_METRICS = {"kendalltau", "pearson", "spearman", 'biweight_midcorrelation', 'bicor'}
    STATISTICAL_TEST_METRICS = {'ks_2samp', 'ttest_1samp', 'ttest_ind', 'ttest_ind_from_stats', 'ttest_rel', "wilcoxon","ranksums", "mannwhitneyu"}
    DISTANCE_METRICS = distance.__dict__["__all__"]

    # Assertions
    accepted_modes = ["similarity", "dissimilarity", "statistical_test", "infer"]
    assert mode in accepted_modes, f"{mode} is not in {accepted_modes}"

    # Metric name
    if hasattr(metric,"__call__"):
        metric_name = metric.__name__
    else:
        metric_name = metric
    # Infer mode
    if mode == "infer":
        if metric_name in CORRELATION_METRICS:
            mode = "similarity"
        if metric_name in STATISTICAL_TEST_METRICS:
            mode = "statistical_test"
        if metric_name in DISTANCE_METRICS:
            mode = "dissimilarity"
        assert mode != "infer", f"Unable to infer mode from metric = {metric}"
        print(f"Inferred mode as `{mode}`", file=sys.stderr)

    # Similarity transformation function
    functions = {
        True: lambda X: (X+1)/2,
        False: lambda X:np.abs(X),
        None: lambda X:X,
    }
    func_signed = functions[signed]

    # Axis
    if axis == 0: X_copy = X.copy()
    if axis == 1: X_copy = X.copy().T

    # Locals
    Ar_X = X_copy.values
    labels = X_copy.index

    # Placeholder
    df_dense = None

    # Pairwise interactions
    while df_dense is None:
        # Statistical test
        if metric_name in STATISTICAL_TEST_METRICS:
            metric = lambda u,v: getattr(stats, metric_name)(u,v)[1]
            df_dense = pd.DataFrame(distance.squareform(distance.pdist(Ar_X, metric=metric)), index=labels, columns=labels)
            fill_diagonal = 0.0
            if mode != "statistical_test":
                print(f"Convert `mode` from `{mode}` to `statistical_test`", file=sys.stderr)
                mode = "statistical_test"
            break
        # Euclidean distance
        if metric_name == "euclidean":
            df_dense = pd.DataFrame(distance.squareform(distance.pdist(Ar_X, metric="euclidean")), index=labels, columns=labels)
            fill_diagonal = 0.0
            break
        # Biweight midcorrelation
        if metric == "bicor":
            metric = biweight_midcorrelation
            try:
                df_dense = bicor(X_copy.T, condensed=False, self_interactions=True)
                break
            except:
                pass

        # Compute correlation and send the rest down the pipeline
        if metric_name in set(CORRELATION_METRICS) - set(["bicor", "biweight_midcorrelation"]):
            metric = metric_name
        if type(metric) == str:
            # Correlation
            if metric in CORRELATION_METRICS:
                df_dense = X_copy.T.corr(method = metric)
                break
            # Distance
            if metric_name in DISTANCE_METRICS:
                metric = getattr(distance, metric)

        # Check metric
        assert hasattr(metric, "__call__"), f"metric `{metric}` is unrecognized and should be converted into a callable at this point of the function"

        # Pairwise computation
        df_dense = pd.DataFrame(pairwise_distances(X_copy, metric=metric, n_jobs=n_jobs), index=labels, columns=labels)

        # Diagonal
        if fill_diagonal is not None:
            if fill_diagonal == "auto":
                fill_diagonal = {"similarity":1.0, "dissimilarity":0.0}[mode]
            fill_diagonal = np.repeat(fill_diagonal, X_copy.shape[0]).tolist()
            np.fill_diagonal(df_dense.values, fill_diagonal)

    # Check dense
    assert df_dense is not None, f"`df_dense` is None.  Check the following: metric={metric}"

    # Check NaN
    df_isnull = df_dense.isnull()
    if np.any(df_isnull):
        num_null = df_isnull.values.ravel().sum()
        warnings.warn(f"There are {num_null} null values")

    # Symmetric keywords
    _symmetric_kws={
        "metric_type":metric_name,
        "func_metric":metric if hasattr(metric, "__call__") else None,
        "metadata":{
            "input_shape":X.shape,
        },
    }
    _symmetric_kws.update(symmetric_kws)

    # Return object
    return get_data_object(df_dense, into, mode, func_signed, metric, _symmetric_kws, CORRELATION_METRICS, DISTANCE_METRICS, name, check_metrics)

# Pairwise distance on ete trees
def pairwise_tree_distance(tree, topology_only=True):
    d_topology = defaultdict(dict)
    leaves = tree.get_leaf_names()
    for i, id_obsv_A in enumerate(leaves):
        for j in range(i+1, len(leaves)):
            id_obsv_B = leaves[j]
            if id_obsv_A != id_obsv_B:
                d_topology[id_obsv_A][id_obsv_B] =  d_topology[id_obsv_B][id_obsv_A] = tree.get_distance(target=id_obsv_A, target2=id_obsv_B, topology_only=topology_only)
    df_dism_tree = pd.DataFrame(d_topology)
    np.fill_diagonal(df_dism_tree.values, 0)
    return df_dism_tree

# Pairwise log fold-change
def pairwise_logfc(X, idx_ctrl, idx_treatment, func_log=np.log2, name_ctrl="control", name_treatment="treatment"):
    """
    Positive values means they are greater in idx_treatment than in idx_ctrl.
    Negative values means they are greater in idx_ctrl than in idx_treatment.
    """
    # Init
    idx_attr = X.columns

    idx_ctrl = sorted(idx_ctrl)
    idx_treatment = sorted(idx_treatment)

    # Log Transform
    X_logscale = func_log(X)

    # Groups
    A_ctrl = X_logscale.loc[idx_ctrl,:].values
    A_treatment = X_logscale.loc[idx_treatment,:].values

    # Pairwise profiles
    logfc_profiles = np.vstack(A_treatment[:, np.newaxis] - A_ctrl)

    # Labels
    idx_pairwise_labels = itertools.product(idx_treatment, idx_ctrl)
    names = [name_treatment, name_ctrl]
    return pd.DataFrame(logfc_profiles, index=pd.MultiIndex.from_tuples(idx_pairwise_labels, names=names), columns=idx_attr)
