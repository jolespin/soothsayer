#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================
# Creator: Josh L. Espinoza (J. Craig Venter Institute)
# Date[0]:Init: 2021-September-14
# =============================
# BSD License
# =============================
# Copyright 2018 Josh L. Espinoza
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# =============================
# Future
# =============================
# (1) Add method to load in weights
# (2) Load in kernel
# (5) More model types (SVC?)
# (6) *** Once the model finds the best option, predict each of the left out samples and record the probabilities for further diagnosis
# (7) Add model to synopsis (actual predictive model)
# (8) Reverse ordering of plot in linear?
# (9) Change the phrasing `q` to something else because now we are actually using percentiles and p=0.1 is misleaning b/c it looks like a p-value
# (10) Run skopt BayesSearchCV for ensemble versions
# (11) Add a plot that gives the best scores for all of the percentiles for each model_type
# (12) Add a force_overwrite and make it so any changes in percentile automatically propogate up and the higher percentiles (dependent on lower) are recomputed
# ==============================
# Devel Log
# ==============================
# (1) Make copy of stdout as a summary file in __synopsis
# (2) Add sem to accuracy plots
# (3) Added option to transform data (2018-May-03)
# (4) Added support for cv=0 skipping cross-validation.  Basically, just get the weights out (2018-May-07)
# (5) Create separate logs for tree, logistic, and info (2018-June-04)
# (6) Adding iterative method via percentiles (2018-May-31)
# (7) Adding baseline (2018-May-31)
# (8) Adding log info (2018-May-31)
# (9) Fixed loading previous data (2018-June-05)
# (10) Removing `linear` method, adding `early_stopping`, and making cross-validation go from smallest to largest
# (11) Fixed the .compress compatibility for pandas v1.x
# (12) Got error from bzip2: TypeError: object of type 'pickle.PickleBuffer' has no len() . Changing bz2 to no compression
# (13) Removed extra normalization/transformations
# ==============================
# Current
# ==============================


# ==============================
# Insects
# ==============================
# Cross validating model subsets for percentile 0.9 and min_threshold 0.9:
# /usr/local/devel/ANNOTATION/jespinoz/anaconda/lib/python3.6/site-packages/matplotlib/pyplot.py:524: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning, see the rcParam `figure.max_open_warning`).
#   max_open_warning, RuntimeWarning)
# The scores_ file has a .1 appended to the n_features_included b/c of the set_index.  This needs to be fixed after running this on model_v5.4b.  The fix will be easy, just don't include n_features_included and then change the best_results to X.shape[1] - n_features_excluded

# Version
__version_clairvoyance__ = "v2022.01.06"

# Built-ins
import os, sys, re, multiprocessing, itertools, argparse, subprocess, time, logging, datetime, shutil, copy, warnings
from collections import *
from io import StringIO, TextIOWrapper, BytesIO
import pickle, gzip, bz2, zipfile


warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# Blocks the ConvergenceWarning...it's really annoying
# ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.

os.environ['KMP_DUPLICATE_LIB_OK']='True' # In case you have 2 versions of OMP installed

# # Custom Soothsayer Scripts
# script_directory = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(f"{script_directory}/src/")
# from soothsayer_functions import *

# PyData
import pandas as pd
import numpy as np
import xarray as xr

## Plotting
from matplotlib import use as mpl_backend
mpl_backend("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from scipy import stats
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import clone

# Scikit-bio
from skbio.stats.composition import clr

# Parallel
from joblib import Parallel, delayed

# Utility
from tqdm import tqdm, trange

pd.set_option('max_colwidth',300)




# Functions
legend_kws = dict(fontsize=10, frameon=True, facecolor="white", edgecolor="black", loc='center left', bbox_to_anchor=(1, 0.5))
to_precision = lambda x, precision=5, return_type=float: return_type(("{0:.%ie}" % (precision-1)).format(x))

# Get duration
def format_duration(start_time):
    """
    Adapted from @john-fouhy:
    https://stackoverflow.com/questions/538666/python-format-timedelta-to-string
    """
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


# Log files
def create_logfile(name, path,level=logging.INFO, mode="w"):
    """Function setup as many loggers as you want"""
    # https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    handler = logging.FileHandler(path, mode=mode)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def determine_mode_for_logfiles(path, force_overwrite):
    if force_overwrite:
        mode_for_logfiles = "w"
    else:
        if os.path.exists(path):
            mode_for_logfiles = "a"
        else:
            mode_for_logfiles = "w"
    return mode_for_logfiles

# Filter based on percentile
def filter_percentile(Se_data, q=0.9, nonzero=True, mode="highpass", bandpass_kws={"low":{}, "high":{}}):
    if nonzero == True:
        Se_data = Se_data[Se_data > 0]
    threshold = np.percentile(Se_data, q=100*q)
    if mode == "highpass":
        return Se_data[Se_data > threshold]
    if mode == "lowpass":
        return Se_data[Se_data < threshold]
    if mode == "bandpass":
        Se_copy = Se_data.copy()
        Se_copy = filter_percentile(Se_copy, mode="highpass", q=bandpass_kws["low"])
        Se_copy = filter_percentile(Se_copy, mode="lowpass",  q=bandpass_kws["high"])
        return Se_copy

def read_dataframe(path, sep="\t", index_col=0, header=0, compression="infer", pickled="infer", func_index=None, func_columns=None, engine="c", verbose=True, **args):
    start_time = time.time()
    _ , ext = os.path.splitext(path)
    # Serialization
    if pickled == "infer":
        if ext in {".pkl", ".pgz", ".pbz2"}:
            pickled = True
        else:
            pickled = False
    # Compression
    if compression == "infer":
        if pickled:
            if ext == ".pkl":
                compression = None
            if ext == ".pgz":
                compression = "gzip"
            if ext == ".pbz2":
                compression = "bz2"
        else:
            if path.endswith(".gz"):
                compression = "gzip"
            if path.endswith(".bz2"):
                compression = "bz2"
            if path.endswith(".zip"):
                compression = "zip"
    if pickled == False:
        df = pd.read_csv(path, sep=sep, index_col=index_col, header=header, compression=compression, engine=engine, **args)
    if pickled == True:
        df = pd.read_pickle(path, compression=compression)
    # Map indices
    if func_index is not None:
        df.index = df.index.map(func_index)
    if func_columns is not None:
        df.columns = df.columns.map(func_columns)

    if verbose:
        print("{} | Dimensions: {} | Time: {}".format(path.split('/')[-1], df.shape, format_duration(start_time)), file=sys.stderr)
    return df

# Reading serial object
def read_object(path, compression="infer", serialization_module=pickle):
    if compression == "infer":
        _ , ext = os.path.splitext(path)
        if (ext == ".pkl") or (ext == ".dill"):
            compression = None
        if ext == ".pgz":
            compression = "gzip"
        if ext == ".pbz2":
            compression = "bz2"
    if compression is not None:
        if compression == "gzip":
            f = gzip.open(path, "rb")
        if compression == "bz2":
            f = bz2.open(path, "rb")
    else:
        f = open(path, "rb")
    obj = serialization_module.load(f)
    f.close()
    return obj

def write_object(obj, path, compression="infer", serialization_module=pickle, protocol=pickle.HIGHEST_PROTOCOL, *args):
    """
    Extensions:
    pickle ==> .pkl
    dill ==> .dill
    gzipped-pickle ==> .pgz
    bzip2-pickle ==> .pbz2
    """
    if compression == "infer":
        _ , ext = os.path.splitext(path)
        if (ext == ".pkl") or (ext == ".dill"):
            compression = None
        if ext == ".pgz":
            compression = "gzip"
        if ext == ".pbz2":
            compression = "bz2"
    if compression is not None:
        if compression == "bz2":
            f = bz2.BZ2File(path, "wb")
        if compression == "gzip":
            f = gzip.GzipFile(path, "wb")
    else:
        f = open(path, "wb")
    serialization_module.dump(obj, f, protocol=protocol, *args)
    f.close()

# ==============================================================================
# Normalization
# ==============================================================================
# Total sum scaling
def transform_tss(X):
    sum_ = X.sum(axis=1)
    return (X.T/sum_).T

# Center Log-Ratio
def transform_clr(X):
    X_tss = transform_tss(X)
    X_log = np.log(X_tss)
    geometric_mean = X_log.mean(axis=1)
    return (X_log - geometric_mean.values.reshape(-1,1))



# Normalization
def transform(X, method="tss", axis=1):
    """
    Assumes X_data.index = Samples, X_data.columns = features
    axis = 0 = cols, axis = 1 = rows
    e.g. axis=1, method=ratio: transform for relative abundance so each row sums to 1.
    "tss" = total-sum-scaling
    "clr" = center log-ratio

    """
    # Transpose for axis == 0
    if axis == 0:
        X = X.T
    # Common
    if method == "tss":
        df_transformed = transform_tss(X)
    if method == "clr":
        df_transformed = transform_clr(X)

    # Transpose back
    if axis == 0:
        df_transformed = df_transformed.T
    return df_transformed


# Clairvoyant
class Clairvoyant(object):
    """
    Clairvoyant is an algorithm that weights features that can optimize the predictive capacity of a dataset

    devel
    ============
    # 2018-April-18
    (1) Added decision tree option
    (2) Added repr for relevant metadata and parameters
    (3) Added a metadata container
    (4) Added an indexable pd.DataFrame with all of the model indicies and the accuracies from the intial fitting
    (5) Added default base_model for cross-validation using hyperparameters with best accuracy from initial fitting
    (6) Adjusted (5) to sort by accuracy and if there are multiple equally good hyperparameters, it favors settings closer to the default settings
    (7) When min_threshold is None it calculates the weights from `threshold_range` (100 linear spaced segments)
    # 2018-April-24
    (8) Added standard error of the mean in the plots
    (9) Added sem to scores output
    (10) Added feature set to scores output
    (11) Fixed a HUGE error where the X.index and y.index were getting offset because I was looking for overlaps
    (12) Added functionality for cross-validation to include a target_score.  Mostly useful for commandline version
    # 2018-June-01
    (13) Added value of min_threshold in the weights
    (14) Added a default scaling for weights in mode=0 and mode=2 so it does not need to be done in mode=1
    (15) Removing n_space and making it default number of hyperparameter configurations for both logistic and tree (n=40)
    # 2018-August-22
    (16) Added early_stopping
    (17) Added weights for specific classes for LogisticRegression (tree-based models will be next)
    # 2018-October-08
    (18) Fixed error introduced from process tuples as indicies in pandas 0.23.4: https://github.com/pandas-dev/pandas/issues/22832

    Previous Versions
    =================
    seeker
    seeker_linear

    Future
    ======
    Refine option for softmax and collapsing groups
    Set up models to work on any combination of parameters
    Add other models (SVC?)
    Randomly sample features

    Benchmarks
    ==========
    sklearn.__version__ = 0.19.1
    (No early_stopping)
    Dataset: (150, 54)
    LogisticRegression:
    Fitting = 2.66 ms ± 111 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
    Predicting = 82.2 µs ± 214 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    DecisionTreeClassifier:
    Fitting = 1.7 ms ± 14.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    Predicting = 81.5 µs ± 1.26 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    Notes
    ====
    Note, only set up for specific parameters (i.e. only use C,penalty for lr and criterion,max_features,min_samples_leaf for tree)
    """
    def __init__(
        self,
        # Base
        name=None,
        # Iteration
        n_iter=10,
        # Modeling
        random_state=0,
        model_type="logistic",
        n_jobs=-1,
        random_mode=0,
        # Logistic Regression
        lr_space={
            "C":np.linspace(1e-10,1.0,20),
            "penalty":["l1", "l2"],
        },
        lr_kws=dict(max_iter=10000, solver="liblinear"),
        # Decision Tree
        tree_space={
            "criterion":["gini","entropy"],
            "max_features":["log2", "sqrt", None, 0.382],
            "min_samples_leaf":[1,2,3,5,8],
        },
        tree_kws={
        },
        # Labeling
        map_encoding=None,
        feature_type="feature",
        class_type="class",
        verbose=True,
        metadata=dict(),
    ):
        """
        Formula for `shape_output`:
        n_models = 2*n_iter*40parameterconfigs = 80*n_iter

        Default values result in 40 parameter configurations, fitted on set A and B, resulting in 80 models per iteration.


        """
        self.name = name
        self.n_iter = n_iter
        if isinstance(random_state, int):
            random_state = np.random.RandomState(int(random_state))
        self.random_state= random_state
        self.random_state_value = random_state.get_state()[1][0]
        self.random_mode = random_mode
        # Logistic Regression
        self.lr_space = copy.deepcopy(lr_space)
        self.lr_kws = lr_kws
        # Decision Tree
        self.tree_space = copy.deepcopy(tree_space) # May not need the deepcopy here
        self.tree_kws = tree_kws
        self.model_type = model_type.lower()
        self.models = self._placeholder_models()
        self.n_jobs = n_jobs
        self.map_encoding = map_encoding
        self.feature_type = feature_type
        self.class_type = class_type
        self.verbose = verbose
        # Metadata
        self.metadata = {
            "name":self.name,
            "n_models":self.n_models,
            "model_type":self.model_type
        }
        self.metadata.update(metadata)

    def __repr__(self):
        class_name = self.__class__.__name__
        return "{}({})".format(class_name, str(self.metadata)[1:-1])
    def copy(self):
        return copy.deepcopy(self)

    # =================================================================================
    # Fitting
    # =================================================================================
    def _placeholder_models(self):
        """
        Internal: Label models
        Need to extend this for using different hyperparameters
        """
        self.param_index = OrderedDict()
        def _get_random_state(mode, i):
            d_random = {
                0: self.random_state,
                1: np.random.RandomState(self.random_state_value+i),
                2: np.random.RandomState(np.random.randint(low=0, high=int(1e5)))
            }
            return d_random[mode]

        # Indexing for hyperparameters and models
        models = list()
        a = np.ones(self.n_iter*2)
        b = np.arange(a.size)
        if self.model_type == "logistic":
            assert all(list(map(lambda x: x in self.lr_space, ["penalty", "C"]))), "`lr_kws` must contain `penalty` and `C`"
            param_set = list(map(lambda x:dict(zip(self.lr_space.keys(), x)), itertools.product(*self.lr_space.values())))
            c = np.asarray(len(param_set)*a*b, dtype=int)
            for i, hyperparameters in enumerate(param_set):
                # Random State
                rs = _get_random_state(mode=self.random_mode, i=i)
                # Construct model
                model = LogisticRegression(random_state=rs, multi_class="ovr", **hyperparameters, **self.lr_kws)
                models.append(model)
                # Store the resulting indicies that will correspond with these hyperparameters
                self.param_index[tuple(hyperparameters.items())] = c + i

        if self.model_type == "tree":
            assert all(list(map(lambda x: x in self.tree_space, ["criterion", "max_features", "min_samples_leaf"]))), "`tree_kws` must contain `criterion`, `max_features`, and `min_samples_leaf`"
            param_set = list(map(lambda x:dict(zip(self.tree_space.keys(), x)), itertools.product(*self.tree_space.values())))
            c = np.asarray(len(param_set)*a*b, dtype=int)
            for i, hyperparameters in enumerate(param_set):
                # Random State
                rs = _get_random_state(mode=self.random_mode, i=i)
                # Construct model
                model = DecisionTreeClassifier(random_state=rs, **hyperparameters, **self.tree_kws)
                models.append(model)
                # Store the resulting indicies that will correspond with these hyperparameters
                self.param_index[tuple(hyperparameters.items())] = c + i
        self.n_models = len(models)*self.n_iter*2
        return models

    def _fit_partitions(self, X_A, X_B, y_A, y_B, model, weight_type):
        """
        Internal: Get coefs
        accuracy_A_B means accuracy of B trained on A
        """

#         print(X_A.shape, X_B.shape, y_A.shape, y_B.shape, model, weight_type)
        # Subset A
        model.fit(X_A, y_A)
        weight_A = getattr(model, weight_type)
        accuracy_A_B = np.mean(model.predict(X_B) == y_B)
        # Subset B
        model.fit(X_B, y_B)
        weight_B = getattr(model, weight_type)
        accuracy_B_A = np.mean(model.predict(X_A) == y_A)
        return [weight_A, weight_B], [accuracy_A_B, accuracy_B_A]

    def _merge_partitions(self, X, y, stratify, subset, desc):
        """
        Internal: Distribute tasks
        # (n_space_per_iteration, 2_splits, n_classifications, n_features)
        """

        if self.verbose:
            iterable = tqdm(range(self.random_state_value, self.random_state_value + self.n_iter), desc=desc)
        else:
            iterable = range(self.random_state_value, self.random_state_value + self.n_iter)

        # Iterate through n_iter
        weights = list()
        accuracies = list()

        for i in iterable:
            rs = np.random.RandomState(i)
            X_A, X_B, y_A, y_B = model_selection.train_test_split(X,y,test_size=subset,stratify=stratify, random_state=rs)
            params = {"X_A":X_A, "X_B":X_B, "y_A":y_A, "y_B":y_B}
            parallel_results = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_partitions)(**params,
                                                                                          model=model,
                                                                                          weight_type={"logistic":"coef_","tree":"feature_importances_"}[self.model_type]) for model in self.models)

            # Extract weights and accuracy
            # Convert to array and collapse (X_A and X_B)
            # Store the weights and accuracies

            # Weights
            if self.model_type == "logistic":
                # Might be able to condense the w for both
                w = np.concatenate(list(map(lambda x: x[:-1], parallel_results)), axis=0) #w.shape ==> (n_space_per_iteration, 2_splits, n_classifications, n_features)
                w = np.concatenate([w[:,0,:,:], w[:,1,:,:]], axis=0) # w shape:  (2*n_space_per_iteration, n_classifications, n_features)
            if self.model_type == "tree":
                w = np.asarray(list(map(lambda x: x[0], parallel_results)))
                w = np.concatenate([w[:,0], w[:,1]], axis=0)
            weights.append(w)
            # Accuracies
            acu = np.asarray(list(map(lambda x: x[-1], parallel_results))) #acu shape ==> (n_space_per_iteration, 2_splits)
            acu = np.concatenate([acu[:,0], acu[:,1]], axis=0) #acu shape ==> (2*n_space_per_iteration, )
            accuracies.append(acu)

        # Merge all of the weights and accuracies
        weights = np.concatenate(weights, axis=0)
        accuracies = np.concatenate(accuracies, axis=0)

        return (weights, accuracies)

    def fit(self, X, y, stratify=True, subset=0.5, desc="Permuting samples and fitting models"):
        """
        Fit data in parallel. Rewrite this to enable loading data.
        """
        self.X = X.copy()
        self.y = y.copy()
        self.observation_ids = self.X.index
        self.feature_ids = self.X.columns
        self.shape_input_ = self.X.shape
        self.class_labels = sorted(y.unique())
        self.class_labels_encoding = self.class_labels
        if self.map_encoding is not None:
            self.class_labels = list(map(lambda x:self.map_encoding[x], self.class_labels))
        if stratify:
            stratify = self.y
        (self.kernel_, self.accuracies_) = self._merge_partitions(X=self.X, y=self.y, stratify=stratify, subset=subset, desc=desc)

        self.shape_ = self.kernel_.shape
        max_accuracy = self.extract_hyperparameters()["accuracy"].max()
        idx_maxacu = self.extract_hyperparameters()["accuracy"][lambda x: x == max_accuracy].index
        if len(idx_maxacu) > 1:
            if self.model_type == "logistic":
                # Prefer l2 with higher C
                self.best_hyperparameters_ = pd.DataFrame(list(map(dict,idx_maxacu))).sort_values(["penalty", "C"], ascending=[False,False]).iloc[0,:].to_dict()
            elif self.model_type == "tree":
                # Prefer lower min_samples_leaf, gini, and sqrt
                self.best_hyperparameters_ = pd.DataFrame(list(map(dict,idx_maxacu))).sort_values(["min_samples_leaf", "criterion", "max_features"], ascending=[True,False, False]).iloc[0,:].to_dict()
        else:
            self.best_hyperparameters_ = dict(idx_maxacu[0])
        self.metadata["kernel_shape"] = self.shape_
        self.metadata["best_hyperparameters"] = self.best_hyperparameters_
        return self


    # =================================================================================
    # Extracting
    # =================================================================================
    def extract_kernel(self, into="numpy", name=None, min_threshold=None):
        """
        Extract coefficients from training
        if penalty is specific, an array for l1 or l2 loss specifically is output.  Does not support xarray
        """
        min_threshold = 0 if min_threshold in [False, None] else min_threshold

        W = self.kernel_[self.accuracies_ > min_threshold]

        if into == "numpy":
            return W

        if self.model_type == "logistic":
            if into == "xarray":
                if W.shape[1] > 1:
                    return xr.DataArray(W, dims=["index_model", self.class_type, self.feature_type], coords=[range(W.shape[0]), self.class_labels, self.feature_ids], name=name)
                else:
                    return xr.DataArray(np.squeeze(W), dims=["index_model",  self.feature_type], coords=[range(W.shape[0]), self.feature_ids], name=name)
            if into == "pandas":
                panel = pd.Panel(W,
                             major_axis=pd.Index(range(W.shape[1]), level=0, name=self.class_type),
                             minor_axis=pd.Index(self.feature_ids, level=1, name=self.feature_type))
                df = panel.to_frame().T
                df.index.name = "index"
                return df
        elif self.model_type == "tree":
            if into == "xarray":
                return xr.DataArray(W, dims=["index_model", self.feature_type], coords=[range(W.shape[0]),  self.feature_ids], name=name)
            if into == "pandas":
                return pd.DataFrame(W, index=pd.Index(range(W.shape[0]), name="index_model"), columns=pd.Index(self.feature_ids, name=self.feature_type))

    def extract_accuracies(self, into="numpy", name=None):
        """
        Extract coefficients from training
        if penalty is specific, an array for l1 or l2 loss specifically is output.  Does not support xarray
        """
        if into == "numpy":
            return self.accuracies_
        if into == "pandas":
            return pd.Series(self.accuracies_, name=name)

    def extract_weights(self, min_threshold=None, ascending=False, func_reduce=np.sum, func_weights=None, name=None, mode=0, threshold_range=(0,0.9)):
        """
        Extract weights
        """
        if all([(mode == 0), (min_threshold is None)]):
            mode = 2
        if mode == 0:
            # Extract weights from the kernel
            W = np.abs(self.extract_kernel(into="numpy", min_threshold=min_threshold))

            if self.model_type == "logistic":
                weights_ = pd.Series(func_reduce(func_reduce(W, axis=0), axis=0), index=self.feature_ids, name=name)
            if self.model_type == "tree":
                weights_ = pd.Series(func_reduce(W, axis=0), index=self.feature_ids, name=name)
            if func_weights is not None:
                weights_ = func_weights(weights_)

            weights_ = weights_/weights_.sum()

            if ascending is None:
                return weights_
            else:
                return weights_.sort_values(ascending=ascending)
        if mode == 1:
            # Extract a range of weights from the kernel
            data = dict()
            threshold_values = np.linspace(*threshold_range,100)
            if min_threshold is not None:
                threshold_values = sorted(set(threshold_values) | set([min_threshold]))
            for i in threshold_values:
                w = self.extract_weights(min_threshold=i, ascending=None, mode=0)
                data[i] = w
            df_w = pd.DataFrame(data).T
            df_w.index.name = "threshold"
            df_w.columns.name = "weights"
            return df_w.sort_index()
        if mode == 2:
            # Reduce all of the weights into a single vector
            weights_ = func_reduce(self.extract_weights(ascending=ascending, mode=1), axis=0)
            weights_ = weights_/weights_.sum()
            if ascending is None:
                return weights_
            else:
                return weights_.sort_values(ascending=ascending)
        if mode == 3:
            # Mode 3: Returns specific weights for each class
            # Note, this only works for logistic regression.  Future versions will use OneVsRestClassifier with DecisionTree
            # model = OneVsRestClassifier(DecisionTreeClassifier(random_state=0), n_jobs=-1)
            # model.fit(X,y)
            # np.asarray([*map(lambda x: x.feature_importances_, model.estimators_)])

            A = np.abs(self.extract_kernel(into="numpy", min_threshold=min_threshold))
            A = func_reduce(A, axis=0)
            W = A/A.sum(axis=1).reshape(-1,1)
            return pd.DataFrame(W,
                                index=pd.Index(self.class_labels, name=self.class_type),
                                columns=pd.Index(self.feature_ids, name=self.feature_type)).T



    def extract_hyperparameters(self, ascending=None):
        index = pd.Index(list(map(frozenset, self.param_index.keys())), dtype="category")
        data = list(map(np.asarray, self.param_index.values()))
        df = pd.Series(data, index=index).to_frame("index_model")
        try:
            Se_hyperparam_acu = df["index_model"].map(lambda x: np.mean(self.accuracies_[x]))
            df.insert(loc=0, column="accuracy", value=Se_hyperparam_acu)
            if ascending in [True, False]:
                df = df.sort_values("accuracy", ascending=ascending)

        except AttributeError:
            print("Must run `.fit` before accuracies can be added", file=sys.stderr)
        df.index.name = "hyperparameters"
        return df
    ## set_weights is DEPRACATED because information is collected during the fitting process
    # def set_weights(self, external=None, *args):
    #
    #     if external is not None:
    #         self.weights_ = external
    #     else:
    #         self.weights_ = self.extract_weights(*args)
    #     if self.verbose:
    #         print("Updating weights...", file=sys.stderr)

    # # Load fitted kernel
    # def load_kernel(self, path):
    #     print("Eventually...")

    # =================================================================================
    # Cross Validation
    # =================================================================================
    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def _cross_val_score_parallel(self, model, X, y, Ar_X, Ar_y, y_groupby, idx_tr, idx_te, func_groupby, class_labels_encoding):
        model.fit(Ar_X[idx_tr,:],Ar_y[idx_tr])
        df_proba = pd.DataFrame(model.predict_proba(Ar_X[idx_te,:]), index=X.index[idx_te], columns=class_labels_encoding)
        df_groupby = df_proba.groupby(func_groupby).sum().apply(softmax, axis=1)
        y_hat = df_groupby.apply(np.argmax, axis=1)
        y_true = y_groupby[y_hat.index]
        return metrics.accuracy_score(y_true, y_hat)


### REPLACE WITH GROUPBY?
#     def _parse_compound(self,x):
#         return "-".join(x.split("|__|")[0].split("-")[:-1])
#     def _parse_replicate(self,x):
#         return x.split("|__|")[0]

    def cross_validate(self, model="auto", cv=5, cv_labels=None, n_jobs=-1, method="bruteforce", adaptive_range = [0, 0.5, 1.0], adaptive_steps = [1, 10], early_stopping=100, verbose=True, func_groupby=None, min_threshold=None, path_save=None, target_score=0, desc="Cross validating model subsets", log_file=sys.stderr, log_prefix=""):
        """
        2018-02-15
            * Added `bruteforce`
        Notes:
            * Automatically called extract weights?
        """

        self.performance_ = None
        # Hardcoded placeholder
        scoring="accuracy"
        # Model
        if model == "auto":
            if self.model_type == "logistic":
                model = LogisticRegression(random_state=self.random_state, multi_class="ovr",  **self.best_hyperparameters_, **self.lr_kws)
            elif self.model_type == "tree":
                model = DecisionTreeClassifier(random_state=self.random_state, **self.best_hyperparameters_, **self.tree_kws)

        # If labels are given instead of numeric indices
        condition_A = isinstance(cv, int) == False
        condition_B = type(cv) != list
        if all([condition_A, condition_B]):
            data_sample = cv[0][0]
            if set(data_sample) <= set(self.X.index):
                cv_idx = list()
                for idx_tr, idx_te in cv:
                    cv_idx.append((idx_tr.map(lambda x:self.X.index.get_loc(x)), idx_te.map(lambda x:self.X.index.get_loc(x))))
                cv = cv_idx

        # feature iterables
        if method == "bruteforce":
            if self.verbose:
                iterable = tqdm(range(1, self.X.shape[1]), desc=desc)
            else:
                iterable = range(1, self.X.shape[1])
        if method == "adaptive":
            positions = (np.asarray(adaptive_range)*self.X.shape[1]).astype(int)
            iterable = list()
            for i in range(len(adaptive_range)-1):
                pos = np.arange(positions[i], positions[i+1], adaptive_steps[i])
                iterable += pos.tolist()
            iterable = sorted(set(iterable) - set([0]))
            if self.verbose:
                iterable = tqdm(np.asarray(iterable), desc=desc)
            else:
                iterable = np.asarray(iterable) # Is np.asarray necessary?
        # Weights
        weights_ = self.extract_weights(min_threshold=min_threshold, ascending=False )

        # Cross Validate
        cv_summary = defaultdict(dict)
        cv_results = dict()
        label_features_included = "number_of_features_included"
        label_features_excluded = "number_of_features_excluded"

        # Cross validation
        scores = list()
        Ar_y = self.y.values

        # Baseline
        baseline_score  = np.mean(model_selection.cross_val_score(model, X=self.X.values, y=Ar_y, cv=cv, n_jobs=n_jobs, scoring=scoring))
        early_stopping_counter = 0
        for i,pos in enumerate(iterable):
            idx_query_features = weights_.index[:pos]
            if idx_query_features.size > 0:
                Ar_X = self.X.loc[:,idx_query_features].values
                # Original Implementation
                if func_groupby is None:
                    cross_validation_results = model_selection.cross_val_score(model, X=Ar_X, y=Ar_y, cv=cv, n_jobs=n_jobs, scoring=scoring)
                    accuracy = np.mean(cross_validation_results)
                    sem = stats.sem(cross_validation_results)
                 # Extended to PO1 Project [DEPRACATED]
                else:
                    raise Exception("Scoring must be `accuracy` and `func_groupby` is not None")
                    # if func_groupby in ["compound", "replicate"]:
                        # func_groupby = {"compound": self._parse_compound, "replicate":self._parse_replicate}[func_groupby]
                    # # Future: Cleaner way to remove duplicate indices
                    # y_groupby = pd.Series(pd.Series(Ar_y, index=self.y.index.map(func_groupby)).to_dict(), name="y_groupby")
                    # args = dict(X=self.X, y=self.y, Ar_X=Ar_X, Ar_y=Ar_y, y_groupby=y_groupby, func_groupby=func_groupby, class_labels_encoding=self.class_labels_encoding)
                    # parallel_results = np.asarray(Parallel(n_jobs=n_jobs)(delayed(self._cross_val_score_parallel)(model=clone(model),idx_tr=idx_tr, idx_te=idx_te, **args) for idx_tr, idx_te in cv))
                    # accuracy = np.mean(parallel_results)

                # Update scores
                if i > 0:
                    condition_A = accuracy > max(scores)
                    condition_B = accuracy > target_score
                    if all([condition_A, condition_B]):

                        if log_file is not None:
                            entry = "\t".join([str(log_prefix),
                                              "Baseline={}".format(to_precision(baseline_score)),
                                              "iteration={}".format(i),
                                              "index={}".format(pos),
                                              "{}={}".format(label_features_included, idx_query_features.size),
                                              "Accuracy={}".format(to_precision(accuracy)),
                                              "∆={}".format(to_precision(accuracy - baseline_score)),
                                              "SEM={}".format(to_precision(sem)),
                            ])
                            if type(log_file) == logging.Logger:
                                log_file.info(entry)
                            if self.verbose:
                                if hasattr(log_file, 'write'):
                                    print(entry, file=log_file)
                        if path_save is not None:
                            save_data = dict(model=model, X=Ar_X, y=Ar_y, cv=cv, idx_features=list(idx_query_features), idx_samples=list(self.X.index), n_jobs=n_jobs, scoring=scoring, random_state=self.random_state)
                            write_object(save_data, path_save, compression="infer")
                        # Reset the Counter
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                # Organize results for dataframe
                cv_summary[pos][scoring] = accuracy
                cv_summary[pos]["sem"] = sem
                cv_summary[pos][label_features_included] = idx_query_features.size
                cv_summary[pos]["feature_set"] = list(idx_query_features)

                # Individual cross-validations
                cv_results[pos] = pd.Series(cross_validation_results, index=list(map(lambda x:"cv={}".format(x), range(1, len(cross_validation_results)+1))))

                # Append scores
                scores.append(accuracy)

                if early_stopping_counter >= early_stopping:
                    if log_file is not None:
                        entry = "Early stopping at iteration={} with maximum score= {}".format(i, max(scores))
                        if type(log_file) == logging.Logger:
                            log_file.info(entry)
                        if hasattr(log_file, 'write'):
                            if self.verbose:
                                print(entry, file=log_file)
                    break
        # DataFrame for number of features included
        self.performance_ = pd.DataFrame(cv_summary).T
        # print("TESTING", type(self.performance_), self.performance_)

        self.performance_[label_features_included] = self.performance_[label_features_included].astype(int)
        self.performance_[label_features_excluded] = (self.X.shape[1] - self.performance_[label_features_included]).astype(int)
        self.performance_ = self.performance_.loc[:,[scoring, "sem", label_features_included, label_features_excluded, "feature_set"]]

        # DataFrame for individual cross-validation results
        df_cvresults = pd.DataFrame(cv_results).T

        if cv_labels is not None:
            df_cvresults.columns = cv_labels
        self.performance_ = pd.concat([self.performance_, df_cvresults], axis=1)

        # Best features
        self.best_features_ = self.performance_.sort_values(["accuracy", "sem"], ascending=[False, True]).iloc[0,:]["feature_set"]

        return self.performance_

    # =================================================================================
    # Plotting
    # =================================================================================
    def plot_scores(self, min_threshold=None, style="seaborn-white", figsize=(21,5), ax=None, size_scatter=50, ylabel=None,  title=None, title_kws=dict(), legend_kws=dict(), ylim=None, baseline_score=None):

        with plt.style.context(style):
            df_performance_wrt_included = self.performance_.set_index( "number_of_features_included", drop=True)
            weights_ = self.extract_weights(min_threshold=min_threshold)
            scores = df_performance_wrt_included["accuracy"]
            df_cvresults = df_performance_wrt_included.iloc[:,4:]
            scores = df_cvresults.mean(axis=1)
            sems = df_cvresults.sem(axis=1)

            # Keywords
            _title_kws = dict(fontsize=15, fontweight="bold")
            _title_kws.update(title_kws)

            legend_fontsize = 8 if df_cvresults.shape[1] > 20 else 10
            _legend_kws = {'bbox_to_anchor': (1, 0.5), 'edgecolor': 'black', 'facecolor': 'white', 'fontsize': legend_fontsize, 'frameon': True, 'loc': 'center left'}
            _legend_kws.update(legend_kws)

            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = plt.gcf()


            # Get maximum indicies
            idx_maximums= list()
            values_maximums = list()

            current_maximum = 0
            for i,v in scores.iteritems():
                if v > current_maximum:
                    idx_maximums.append(i)
                    values_maximums.append(v)
                    current_maximum = v


            # Plot each cv accuracy
            df_cvresults.plot(color=sns.color_palette(palette="hls", n_colors=df_cvresults.shape[1]), ax=ax, alpha=0.382)
            # Plot mean accuracy
            ax.plot(scores.index, scores.values, color="black", label="", linewidth=1.618, alpha=0.618)
            # Plot progressive steps
            ax.scatter(idx_maximums, values_maximums, s=size_scatter, color="darkslategray", marker="o", label="", edgecolor="red", linewidth=1.618, alpha=1)
            # Accuracy is the only scoring metric available for the time being
            score_metric = "Accuracy" #str(self.performance_.columns[0]).capitalize()

            # Index of best score
            df_stats = df_performance_wrt_included.loc[:,["accuracy", "sem"]].sort_values(["accuracy", "sem"], ascending=[False, True])
            idx_bestscore = df_stats.index[0]
            best_score = df_stats.loc[idx_bestscore,"accuracy"]
            sem_at_best_score = df_stats.loc[idx_bestscore, "sem"]

            # Lines
            ax.axhline(best_score, linestyle=":", color="black", label= "%s: %0.3f ± %0.3f"%(score_metric, best_score, sem_at_best_score))
            ax.axvline(idx_bestscore, linestyle=":", color="black", label="n= {} features".format(idx_bestscore))

            if baseline_score is not None:
                ax.axhline(baseline_score, color="maroon", linestyle=":", label="Baseline: {}".format(baseline_score))

            # Labels
            if ylabel is None:
                ylabel = score_metric
            ax.set_xlabel("Number of features included")
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            ax.legend(**_legend_kws)
            if title is not None:
                ax.set_title(title, _title_kws)
            if ylim:
                ax.set_ylim(ylim)
            pad = 0.1
            ax.set_xlim((scores.index.min()-pad, scores.index.max()+pad))
            return fig, ax


    def plot_weights(self,  ax=None, style="seaborn-white",  title=None, title_kws=dict(),  axis_kws=dict(), tick_kws=dict(), figsize=(21,5), palette="hls", *args):
        """
        Plot the weights
        """
        _title_kws = {"fontsize":15, "fontweight":"bold"}
        _title_kws.update(title_kws)

        _axis_kws = {"fontsize":15}
        _axis_kws.update(axis_kws)

        _tick_kws = {"fontsize":12}
        _tick_kws.update(tick_kws)
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = plt.gcf()

            df_w = self.extract_weights(mode=1, *args)
            color = sns.color_palette(palette=palette, n_colors=df_w.shape[1])
            df_w.plot(color=color, ax=ax)
            ax.legend_.remove()
            ax.set_xlabel("Threshold", **_axis_kws)
            ax.set_ylabel("$W$", **_axis_kws)
            ax.set_xticklabels(ax.get_xticks(), **_tick_kws)
            ax.set_yticklabels(ax.get_yticks(), **_tick_kws)
            ax.set_xlim((df_w.index.min(), df_w.index.max()))
            if title is not None:
                ax.set_title(title, _title_kws)
            return fig, ax

    # Writing serialized object
    def save_model(self, path, compression="infer"):
        """
        Extensions:
        pickle ==> .pkl
        dill ==> .dill
        gzipped-pickle ==> .pgz
        bzip2-pickle ==> .pbz2
        """
        write_object(self, path)

# Controller
def main(argv=None):
    parser = argparse.ArgumentParser(
    """
    soothsayer:clairvoyance {}
    """.format(__version_clairvoyance__)
    )
    # Input
    parser.add_argument('-X', '--feature_matrix', type=str, help = 'Input: Path/to/Tab-separated-value.tsv of feature matrix (rows=samples, cols=features)')
    parser.add_argument('-y', '--target_vector', type=str, help = 'Input: Path/to/Tab-separated-value.tsv of target vector (rows=samples, column=integer target)')
    parser.add_argument("-e", "--encoding", default=None, help="Input: Path/to/Tab-separated-value.tsv of encoding.  Column 1 has enocded integer and Column 2 has string representation.  No header! [Default: None]")

    # Transformation
    parser.add_argument("--transformation", type=str, default=None, help="Transform the feature matrix.  Valid Arguments: {'tss', 'clr'}")
    parser.add_argument("--multiplicative_replacement", type=float, default=None, help="Multiplicative replacement used for CLR [Default: 1/m^2]")

    # Hyperparameters
    parser.add_argument("-m", "--model_type", type=str, default="logistic,tree", help="Model types in ['logistic','tree'] represented by a comma-separated list [Default: 'logistic,tree']")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of iterations to split data and fit models. [Default: 10]")
    parser.add_argument("--random_state", type=int, default=0, help="[Default: 0]")
    parser.add_argument("--n_jobs", type=int, default=1, help="[Default: 1]")
    parser.add_argument("--min_threshold", type=str, default="0.0,None", help="Minimum accuracy for models when considering coeficients for weights. Can be a single float or list of floats separated by `,` ranging from 0-1. If None it takes an aggregate weight.  [Default: 0.0,None]")
    parser.add_argument("--random_mode", type=int, default=0, help="Random mode 0,1, or 2.  0 uses the same random_state each iteration.  1 uses a different random_state for each parameter set.  2 uses a different random_state each time. [Default: 0]")
    parser.add_argument("--percentiles", type=str, default="0.0,0.3,0.5,0.75,0.9,0.95,0.96,0.97,0.98,0.99", help="Iterative mode takes in a comma-separated list of floats from 0-1 that dictate how many of the features will be used [Default: 0.0,0.3,0.5,0.75,0.9,0.95,0.96,0.97,0.98,0.99]")
    parser.add_argument("--solver", type=str, default="liblinear", help="LogisticRegression solver {‘liblinear’, ‘sag’, ‘saga’} [Default: liblinear]")


    # Labels & Directories
    parser.add_argument("-n", "--name", type=str, default=time.strftime("%Y-%m-%d"), help="Name for data [Default: Date]")
    parser.add_argument("-o", "--out_dir", type=str, default = os.getcwd(), help = "Output: Path/to/existing-directory-for-output.  Do not use `~` to specify /home/[user]/ [Default: cwd]" )
    parser.add_argument("--feature_type", type=str, default="feature", help="Name of feature type (e.g. gene, orf, otu, etc.) [Default: 'feature']")
    parser.add_argument("--class_type", type=str, default="class", help="Name of class type (e.g. phenotype, MOA, etc.) [Default: 'class']")

    # Cross Validation
    parser.add_argument("--method", type=str, default="bruteforce", help = "{'adaptive', 'bruteforce'} [Default: 'bruteforce']")
    parser.add_argument("--adaptive_range", type=str, default="0.0,0.5,1.0", help="Adaptive range in the form of a string of floats separated by commas [Default:'0.0,0.5,1.0']")
    parser.add_argument("--adaptive_steps", type=str, default="1,10", help="Adaptive stepsize in the form of a string of ints separated by commas. len(adaptive_range) - 1 [Default:'1,10']")
    parser.add_argument("--cv", default=5, type=str, help="Number of steps for percentiles when cross validating and dropping features.  If cv=0 then it skips cross-validation step. [Default: 5]")
    parser.add_argument("--min_bruteforce", type=int, default=150, help="Minimum number of features to adjust to bruteforce. If value is 0 then it will not change method [Default: 150]")
    parser.add_argument("--early_stopping", type=int, default=100, help="Stopping the algorthm if a certain number of iterations do not increase the accuracy. Use -1 if you don't want to use `early_stopping` [Default: 100]")


    # Reading in dataframes
    parser.add_argument("--compression", type=str, default="infer")
    parser.add_argument("--pickled", type=str, default="infer")

    # I/O Model
    parser.add_argument("--save_kernel", default=False, help="Save the kernel [Default: False]")
    parser.add_argument("--save_model", default=False, help="Save the model [Default: False]")
    parser.add_argument("--save_data", default=True, help="Save the model [Default: True]")




    opts = parser.parse_args(argv)

    opts.force_overwrite = True

    # opts.force_overwrite = True
    start_time = time.time()



    # Create working directory
    os.makedirs(opts.out_dir, exist_ok=True)
    path_synopsis = "{}/synopsis".format(opts.out_dir)
    os.makedirs(path_synopsis, exist_ok=True)
    log_info = create_logfile("information",
                                  "{}/clairvoyance.log".format(opts.out_dir),
                                  mode=determine_mode_for_logfiles("{}/clairvoyance.log".format(opts.out_dir), opts.force_overwrite)
    )

    # Initiating variables
    # ====================
    # Default Random State
    np.random.seed(opts.random_state)
    # Parallel
    if opts.n_jobs == -1:
        opts.n_jobs = multiprocessing.cpu_count()

    # File Paths
    if opts.out_dir.endswith("/"):
        opts.out_dir = opts.out_dir[:-1]

    # Adaptive ranges
    if opts.method == "adaptive":
        opts.adaptive_range = list(map(lambda x:float(x.strip()), opts.adaptive_range.split(",")))
        opts.adaptive_steps = list(map(lambda x:int(x.strip()), opts.adaptive_steps.split(",")))
        assert np.all(np.asarray(opts.adaptive_range) <= 1.0), "All values in `adaptive_range` should be: 0.0 <= x <= 1.0"
    # Early stopping
    if opts.early_stopping == -1:
        opts.early_stopping = np.inf

    # Minimum threshold
    if "," in opts.min_threshold:
        opts.min_threshold = opts.min_threshold.lower()
        special_values = list()
        if "none" in opts.min_threshold:
            opts.min_threshold = opts.min_threshold.replace("none","")
            special_values.append(None)
        opts.min_threshold = sorted(map(float, filter(bool, opts.min_threshold.split(",")))) + special_values
    else:
        if opts.min_threshold == "none":
            opts.min_threshold = [None]
        else:
            opts.min_threshold = [float(opts.min_threshold)]

    # Percentiles
    if "," in opts.percentiles:
        opts.percentiles = sorted(map(float, filter(bool, opts.percentiles.split(","))))
    else:
        if opts.percentiles.lower() in ["none", "false"]:
            opts.percentiles = [0.0]
        else:
            opts.percentiles = [float(opts.percentiles)]
    if 0.0 not in opts.percentiles:
        opts.percentiles = [0.0] + opts.percentiles
    assert all(map(lambda x: 0.0 <= x < 1.0, opts.percentiles)), "All percentiles must meet the following criteria: 0.0 <= x < 1.0"

    # Model types
    if "," in opts.model_type:
        opts.model_type = list(filter(bool, opts.model_type.split(",")))
    else:
        opts.model_type = [opts.model_type]
    assert set(opts.model_type) <= set(["logistic", "tree"]), "model_type should only be a combination of ['logistic','tree']. Current input: {}".format(opts.model_type)


    # Serialization
    if opts.pickled != "infer":
        if opts.pickled.lower() in ["false", "none"]:
            opts.pickled = False
        if opts.pickled.lower() == "true":
            opts.pickled = True
    # Cross Validation Configure
    cv_repr = opts.cv
    if str(opts.cv).isnumeric():
        opts.cv = int(opts.cv)
        cv_labels = list(map(lambda x: "cv={}".format(x+1), range(opts.cv)))
    else:
        with open(opts.cv,"r") as f:
            _tmp_ncols_cv = len(f.read().split("\n")[0].split("\t"))
        # Is it just 2 columns (train, test) with no header?
        if  _tmp_ncols_cv == 2:
            opts.cv = pd.read_csv(opts.cv, header=None, sep="\t").applymap(eval).values.tolist()
            cv_labels = list(map(lambda x: "cv={}".format(x+1), range(len(opts.cv))))
        # Are there 3 columns with a header with the first column being the name of cross-validation set?
        if _tmp_ncols_cv == 3:
            opts.cv = pd.read_csv(opts.cv, sep="\t", index_col=0).applymap(eval).loc[:,["Training", "Testing"]]
            cv_labels = opts.cv.index
            opts.cv = opts.cv.values.tolist()

    # Saving model
    if type(opts.save_model) == str:
        if opts.save_model.lower() in ["false", "none", "no"]:
            opts.save_model = None
        else:
            opts.save_model = True

    # Saving model data
    if type(opts.save_data) == str:
        if opts.save_data.lower() in ["false", "none", "no"]:
            opts.save_data = None
        else:
            opts.save_data = True



    # Static data
    X_original = read_dataframe(opts.feature_matrix, compression=opts.compression, pickled=opts.pickled,  func_index=str, func_columns=str, verbose=False)


    y = read_dataframe(opts.target_vector, compression=opts.compression, pickled=opts.pickled, func_index=str, func_columns=str, verbose=False)
    if isinstance(y, pd.Series):
        y = y.to_frame()
    y = y.iloc[:,0]

    # Encoding
    if opts.encoding:
        opts.encoding = pd.read_csv(opts.encoding, sep="\t", header=None, index_col=0).iloc[:,0].to_dict()

    # Iterate through model types
    best_configuration = None
    baseline_scores_ = dict()
    summary_table = list()

    # f_full_synopsis = open("{}/{}__synopsis.tsv".format(path_synopsis,opts.name), "w")
    for m in opts.model_type:
        print("= == === ===== ======= ============ =====================", file=sys.stderr)
        print(m, file=sys.stderr)
        print("= == === ===== ======= ============ =====================".replace("=","."), file=sys.stderr)


        estimator = {"logistic":LogisticRegression(random_state=opts.random_state, multi_class='ovr', solver=opts.solver), "tree":DecisionTreeClassifier(random_state=opts.random_state)}[m]
        # Placeholders for dynamic data
        path_features_current = opts.feature_matrix
        best_configuration_for_modeltype = None # (percentile, min_threshold, hyperparameters)
        best_accuracy_for_modeltype = 0
        best_hyperparameters_for_modeltype = None
        model = None
        best_features_for_modeltype = None

        log_crossvalidation = create_logfile(m,
                                                 "{}/{}__cross-validation.log".format(path_synopsis,m),
                                                 mode=determine_mode_for_logfiles("{}/{}__cross-validation.log".format(path_synopsis,m), opts.force_overwrite)
        )
        print("Opening {} synopsis:".format(m), "{}/{}__synopsis.tsv".format(path_synopsis,m))
        f_model_synopsis = open("{}/{}__synopsis.tsv".format(path_synopsis,m), "w", buffering=1)
        print("model_type", "hyperparameters", "percentile",  "min_threshold", "baseline_score", "baseline_score_of_current_percentile", "accuracy", "sem", "delta", "random_state", "number_of_features_included", "feature_set", "clairvoyance_weights", sep="\t", file=f_model_synopsis)

        # Iterate through percentiles of weighted features
        baseline_percentile_0 = None
        for current_iteration, i_percentile in enumerate(opts.percentiles, start=0):

            # Determine whether or not to continue the analysis
            continue_analysis = True
            path_summary = "{}/{}/percentile_{}/summary.txt".format(opts.out_dir,m,i_percentile)
            if opts.force_overwrite == False:
                if os.path.exists(path_summary):
                    if os.path.getsize(path_summary) > 0:
                        continue_analysis = False
            # ==================================================================
            # Load previous analysis
            # ==================================================================
            # Check if directory exists, make one if not
            log_info.info("\t\t{} | Percentile={} | Creating temporary directories in {}".format(m,i_percentile,opts.out_dir))
            os.makedirs("{}/{}/percentile_{}".format(opts.out_dir,m,i_percentile), exist_ok=True)

            f_summary = open(path_summary, "w")
            # Load features
            log_info.info("\t\t{} | Percentile={} | Loading features datasets".format(m,i_percentile ))

            # Terrible hack but this will be addressed in future versions
            features_for_current_iteration = read_dataframe(path_features_current, compression=opts.compression, pickled=opts.pickled,  func_index=str, func_columns=str, verbose=False).columns
            X = X_original[features_for_current_iteration]
            if X.shape[1] < 2:
                print("Skipping current iteration {} because there is/are only {} feature[s]".format(i_percentile, X.shape[1]), file=sys.stderr)
            else:
                # transform
                if opts.transformation is not None:
                    if opts.transformation == "clr":
                        if opts.multiplicative_replacement is None:
                            opts.multiplicative_replacement = 1/X.shape[1]**2
                        X = X + opts.multiplicative_replacement
                    X = transform(X, method=opts.transformation.lower(), axis=1) # COULD PROBABLY MOVE THIS UP AND COMBINE

                # Common Identifiers
                try:
                    assert np.all([X.index == y.index]), "X.index should be the same ordering as y.index"
                except ValueError:
                    A = set(X.index)
                    B = set(y.index)

                    print("set(X) - set(y) : len(A - B) ==>  {}".format(A - B),
                          "set(y) - set(X) : len(B - A) ==>  {}".format(B - A),
                          sep=2*"\n",
                          file=sys.stderr
                    )
                    sys.exit(1)



                params = pd.Series(opts.__dict__)
                if not continue_analysis:
                    log_info.info("\t\t{} | Percentile={} | Skipping analysis and loading data from: {}/{}/percentile_{}".format(m,i_percentile,opts.out_dir,m,i_percentile))

                    # Load model
                    model = read_object(path="{}/{}/percentile_{}/model.pkl".format(opts.out_dir,m,i_percentile), compression=None)

                    # Set parameters for estimator and calculate baseline score
                    estimator.set_params(**model.best_hyperparameters_)
                    baseline_score = model_selection.cross_val_score(estimator, X=X.values, y=y.values, cv=opts.cv).mean()
                    baseline_scores_[(m,i_percentile)] = baseline_score

                    # Set original feature size and baseline score
                    if i_percentile == 0.0:
                        n_features = X.shape[1]
                        baseline_percentile_0 = baseline_score
                    print("{} | Percentile={} | X.shape = {} | Baseline score({}) = {}".format(m, i_percentile, X.shape,path_features_current,to_precision(baseline_score)), file=sys.stderr)


                    # Determine if weights are different between any of the other runs
                    tol_difference = 1
                    if len(opts.min_threshold) > 1:
                        _current_order = np.asarray(model.extract_weights(min_threshold=opts.min_threshold[0], ascending=False).index)
                        for x in opts.min_threshold[1:]:
                            _query_order = np.asarray(model.extract_weights(min_threshold=x, ascending=False).index)
                            num_differences = np.sum(_current_order != _query_order)
                            if num_differences < tol_difference*2:  # Times 2 because differences come in pairs
                                opts.min_threshold.remove(x)
                                log_info.info("\t\t{} | Percentile={} | Removing min_accuracy={} from computation because {} ordering is already present from another threshold".format(m,i_percentile,x,model.feature_type))
                            else:
                                _current_order = _query_order

                    # Placeholders
                    best_accuracy_for_percentile = 0
                    best_min_threshold_for_percentile = None
                    best_weights_for_percentile = None
                    # best_results_for_percentile = None

                    for x in opts.min_threshold:
                        path_query = "{}/{}/percentile_{}/min_threshold_{}".format(opts.out_dir,m,i_percentile,x)
                        w_ = model.extract_weights(min_threshold=x, name=opts.name, ascending=False)

                        # Are there weights for this threshold?  If there's not then that means there are no models that predicted with this threshold or higher
                        if not w_.isnull().any():
                            scores_ = read_dataframe("{}/scores.tsv.gz".format(path_query), sep="\t", compression="gzip", verbose=False)

                            # Update the scores
                            idx_best_accuracy_for_minthreshold = scores_["accuracy"].sort_values(ascending=False).index[0]
                            query_accuracy = scores_.loc[idx_best_accuracy_for_minthreshold,"accuracy"]
                            query_sem = scores_.loc[idx_best_accuracy_for_minthreshold,"sem"]
                            feature_set = scores_.loc[idx_best_accuracy_for_minthreshold, "feature_set"]

                            # Update accuracies for current percentile
                            if query_accuracy > best_accuracy_for_percentile:
                                best_min_threshold_for_percentile = x
                                best_accuracy_for_percentile = query_accuracy
                                best_weights_for_percentile = w_.copy()

                                # Update accuracies for current model
                                if query_accuracy > best_accuracy_for_modeltype:
                                    best_configuration_for_modeltype = (i_percentile, x)
                                    best_hyperparameters_for_modeltype = model.best_hyperparameters_
                                    best_accuracy_for_modeltype = query_accuracy
                                    best_features_for_modeltype = feature_set

                            # Synthesize summary table
                            Se_query = pd.Series([m, model.best_hyperparameters_, i_percentile, x, baseline_percentile_0, baseline_score, query_accuracy,  query_sem, query_accuracy - baseline_percentile_0, opts.random_state, len(feature_set), feature_set, best_weights_for_percentile[feature_set].tolist()],
                                           index=["model_type", "hyperparameters", "percentile",  "min_threshold", "baseline_score", "baseline_score_of_current_percentile", "accuracy", "sem", "delta", "random_state", "number_of_features_included", "feature_set", "clairvoyance_weights"])
                            print(*Se_query.values, sep="\t", file=f_model_synopsis)
                            summary_table.append(Se_query)
                # ==================================================================
                # Continue analysis
                # ==================================================================
                else:
                    if X.shape[1] <= opts.min_bruteforce:
                        method_crossvalidation = "bruteforce"
                        if opts.method != "bruteforce":
                            log_info.info("\t\t{} | Percentile={} | Adjusting `method` to `bruteforce` because X.shape[1] <= {}".format(m,i_percentile,opts.min_bruteforce))
                    else:
                        method_crossvalidation = opts.method
                    # Run model
                    model = Clairvoyant(
                                model_type=m,
                                n_iter=opts.n_iter,
                                map_encoding=opts.encoding,
                                feature_type=opts.feature_type,
                                class_type=opts.class_type,
                                n_jobs=opts.n_jobs,
                                verbose=True,
                                random_state=opts.random_state,
                                random_mode=opts.random_mode,
                    )
                    log_info.info("\t\tFitting data")

                    model.fit(X, y, desc="{} | Percentile={} | Permuting samples and fitting models".format(m, i_percentile))
                    log_info.info("\t\t{} | Percentile={} | Best hyperparameters from fitting: {}".format(m,i_percentile,model.best_hyperparameters_))
                    log_info.info("\t\t{} | Percentile={} | Saving model: {}/{}/percentile_{}/model.pkl".format(m,i_percentile,opts.out_dir,m,i_percentile))
                    if opts.save_model:
                        model.save_model(path="{}/{}/percentile_{}/model.pkl".format(opts.out_dir,m,i_percentile), compression=None)


                    # Get kernel and weights
                    kernel_ = model.extract_kernel(into="xarray", name=opts.name)
                    acu_ = model.extract_accuracies(into="pandas", name=opts.name)
                    hyperparameters_ = model.extract_hyperparameters()
                    W_ = model.extract_weights(min_threshold=None, mode=1) # Future: Incldue any other min_threshold values that aren't in the linspace

                    log_info.info("\t\t{} | Percentile={} | Calculating weights".format(m,i_percentile))

                    # Set parameters for estimator and calculate baseline score
                    estimator.set_params(**model.best_hyperparameters_)
                    baseline_score = model_selection.cross_val_score(estimator, X=X.values, y=y.values, cv=opts.cv).mean()
                    baseline_scores_[(m,i_percentile)] = baseline_score

                    # Set original feature size and baseline score
                    if i_percentile == 0.0:
                        n_features = X.shape[1]
                        baseline_percentile_0 = baseline_score
                    print("{} | Percentile={} | X.shape = {} | Baseline score({}) = {}".format(m,i_percentile,X.shape,path_features_current,to_precision(baseline_score)), file=sys.stderr)

                    acu_.to_frame("Accuracy").to_csv("{}/{}/percentile_{}/acu.tsv.gz".format(opts.out_dir,m,i_percentile), sep="\t", compression="gzip")
                    hyperparameters_.to_csv("{}/{}/percentile_{}/hyperparameters.tsv.gz".format(opts.out_dir,m,i_percentile), sep="\t", compression="gzip")
                    W_.to_csv("{}/{}/percentile_{}/weights.tsv.gz".format(opts.out_dir,m,i_percentile ), sep="\t", compression="gzip")



                    if opts.save_kernel:
                        kernel_.to_netcdf("{}/{}/percentile_{}/kernel.nc".format(opts.out_dir, m, i_percentile))
                    # Determine if weights are different
                    tol_difference = 1
                    if len(opts.min_threshold) > 1:
                        _current_order = np.asarray(model.extract_weights(min_threshold=opts.min_threshold[0], ascending=False).index)
                        for x in opts.min_threshold[1:]:
                            _query_order = np.asarray(model.extract_weights(min_threshold=x, ascending=False).index)
                            num_differences = np.sum(_current_order != _query_order)
                            if num_differences < tol_difference*2:  # Times 2 because differences come in pairs
                                opts.min_threshold.remove(x)
                                log_info.info("\t\t{} | Percentile={} | Removing min_accuracy={} from computation because {} ordering is already present from another threshold".format(m,i_percentile,x,model.feature_type))
                            else:
                                _current_order = _query_order
                    # Soothsayer | feature Finder
                    print("================================\nsoothsayer:clairvoyance {}\n================================".format(__version_clairvoyance__), file=f_summary)
                    print("Name:", opts.name, sep="\t", file=f_summary)
                    print("X:", path_features_current, sep="\t", file=f_summary)
                    print("y:", opts.target_vector, sep="\t", file=f_summary)
                    print("Encoding:", opts.encoding, sep="\t", file=f_summary)
                    print("Path:", opts.out_dir, sep="\t", file=f_summary)
                    print("Shape:", X.shape, sep="\t", file=f_summary)
                    print("Model Type:", m, sep="\t", file=f_summary)
                    print("Percentile:", i_percentile, sep="\t", file=f_summary)
                    # if opts.load_model:
                    #     print("Loaded Model:", opts.load_model, sep="\t", file=f_summary)
                    print("================\n Baseline \n================", file=f_summary)

                    print("Percentile = 0: ", baseline_percentile_0, file=f_summary)
                    print("Current percentile = {}:".format(i_percentile), baseline_score, file=f_summary)


                    print("================\n Hyperparameters \n================", file=f_summary)
                    print("\n".join(str(params[["n_iter", "random_state", "random_mode", "n_jobs", "min_threshold"]]).split("\n")[:-1]), file=f_summary)
                    print("================\n Labels \n================", file=f_summary)
                    print("\n".join(str(params[["feature_type", "class_type"]]).split("\n")[:-1]), file=f_summary)
                    print("================\n Cross Validation \n================", file=f_summary)
                    if opts.method == "adaptive":
                        print("\n".join(str(params[["method","adaptive_range","adaptive_steps", "early_stopping"]]).split("\n")[:-1]), file=f_summary)
                    if opts.method == "bruteforce":
                        print("\n".join(str(params[["method", "min_bruteforce", "early_stopping"]]).split("\n")[:-1]), file=f_summary)


                    print("cv", cv_repr, sep="\t", file=f_summary)
                    print("================\n Data Type \n================", file=f_summary)
                    print("\n".join(str(params[["compression", "pickled"]]).split("\n")[:-1]), file=f_summary)

                    # Placeholders
                    best_accuracy_for_percentile = 0
                    best_min_threshold_for_percentile = None
                    best_results_for_percentile = None
                    best_weights_for_percentile = None
                    # Run crossvalidation
                    run_crossvalidation = opts.cv != 0
                    if run_crossvalidation:
                        # Cross Validation
                        for x in opts.min_threshold:
                            # Default to run the analysis.  First check if the path exists if does say if you want to overwrite.  If not then make the directory and do the analysis
                            run_analysis = True
                            path_query = "{}/{}/percentile_{}/min_threshold_{}".format(opts.out_dir,m,i_percentile,x)
                            if os.path.exists(path_query):
                                # Is there directory empty?
                                if len(os.listdir(path_query)) > 0: # or os.path.getsize > ?
                                    if opts.force_overwrite == False:
                                        run_analysis = False
                            else:
                                os.makedirs(path_query, exist_ok=True)

                            if run_analysis:
                                # Calculate weights
                                w_ = model.extract_weights(min_threshold=x, name=opts.name, ascending=False)

                                # Are there weights for this threshold?
                                if not w_.isnull().any():
                                    weights_ = w_.to_frame("weights")
                                    weights_.to_csv("{}/weights.tsv.gz".format(path_query), sep="\t", compression="gzip")

                                    # NOTE: This is a hack for outputing the weights for each class but only works for logistic regression.  Once this tree methods are converted to OneVsRest then this can be properly integrated
                                    if m == "logistic":
                                        if len(model.class_labels) > 2:
                                            weights_class_specific_ = model.extract_weights(min_threshold=x, name=opts.name, mode=3)
                                            weights_class_specific_.to_csv("{}/weights.class-specific.tsv.gz".format(path_query), sep="\t", compression="gzip")

                                    log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Cross validation".format(m,i_percentile,x))
                                    # Synthesize Datasets

                                    if opts.save_data:
                                        path_save = "{}/model-data.pkl".format(path_query)
                                    else:
                                        path_save = None

                                    scores_ = model.cross_validate(model=clone(estimator),
                                                                    min_threshold=x,
                                                                    cv=opts.cv,
                                                                    cv_labels=cv_labels,
                                                                    method=opts.method,
                                                                    early_stopping=opts.early_stopping,
                                                                    adaptive_range = opts.adaptive_range,
                                                                    adaptive_steps = opts.adaptive_steps,
                                                                    n_jobs=opts.n_jobs,
                                                                    verbose=True,
                                                                    func_groupby=None,
                                                                    path_save=path_save,
                                                                    target_score=best_accuracy_for_modeltype,
                                                                    desc="{} | CV | Percentile={} | Minimum threshold={}".format(m,i_percentile,x),
                                                                    log_file=log_crossvalidation,
                                                                    log_prefix = "{} | pctl={} | t={}".format(m,i_percentile,x)
                                                                    )

                                    log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Writing scores".format(m,i_percentile,x))
                                    scores_.to_csv("{}/scores.tsv.gz".format(path_query), sep="\t", compression="gzip")

                                    # Update the scores
                                    idx_best_accuracy_for_minthreshold = scores_["accuracy"].sort_values(ascending=False).index[0]
                                    query_accuracy = scores_.loc[idx_best_accuracy_for_minthreshold,"accuracy"]
                                    query_sem = scores_.loc[idx_best_accuracy_for_minthreshold,"sem"]
                                    feature_set = scores_.loc[idx_best_accuracy_for_minthreshold, "feature_set"]

                                    # Update accuracies for current percentile
                                    if query_accuracy > best_accuracy_for_percentile:
                                        best_min_threshold_for_percentile = x
                                        if scores_.shape[0] >= 10:
                                            best_results_for_percentile = scores_.sort_values(["accuracy", "sem", "number_of_features_included"], ascending=[False,True, False]).iloc[:10,:3]
                                        else:
                                            best_results_for_percentile = scores_.sort_values(["accuracy", "sem", "number_of_features_included"], ascending=[False,True, False]).iloc[:scores_.shape[0],:3]


                                        best_accuracy_for_percentile = query_accuracy
                                        best_weights_for_percentile = w_.copy()
                                        log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Updating the model object with accuracy={}".format(m,i_percentile,x,query_accuracy))

                                        # May be slower than necessary to put this here but it's safer
                                        if opts.save_model:
                                            model.save_model(path="{}/{}/percentile_{}/model.pkl".format(opts.out_dir,m,i_percentile), compression=None)

                                        # Update accuracies for current model
                                        if query_accuracy > best_accuracy_for_modeltype:
                                            best_configuration_for_modeltype = (i_percentile, x)
                                            best_hyperparameters_for_modeltype = model.best_hyperparameters_
                                            best_accuracy_for_modeltype = query_accuracy
                                            best_features_for_modeltype = feature_set

                                    log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Creating plot".format(m,i_percentile,x))

                                    fig, ax = model.plot_scores(title="{} | {} | percentile={} | min_threshold={}".format(opts.name,m,i_percentile,x), ylim=(0,1), min_threshold=x, baseline_score=baseline_score)
                                    fig.savefig("{}/scores.png".format(path_query), dpi=300, format="png", bbox_inches="tight")
                                    plt.close()

                                    # Synthesize summary table
                                    Se_query = pd.Series([m, model.best_hyperparameters_, i_percentile, x, baseline_percentile_0, baseline_score, query_accuracy,  query_sem, query_accuracy - baseline_percentile_0, opts.random_state, len(feature_set), feature_set, best_weights_for_percentile[feature_set].tolist()],
                                                   index=["model_type", "hyperparameters", "percentile",  "min_threshold", "baseline_score", "baseline_score_of_current_percentile", "accuracy", "sem", "delta", "random_state", "number_of_features_included", "feature_set", "clairvoyance_weights"])
                                    print(*Se_query.values, sep="\t", file=f_model_synopsis)
                                    summary_table.append(Se_query)

                                else:

                                    log_info.error("\t\t{} | Percentile={} | Minimum threshold={} | No models had an accuracy for this threshold so the weights were NaN".format(m,i_percentile,x))
                                    if os.path.exists(path_query):
                                        if len(os.listdir(path_query)) == 0:
                                            os.rmdir(path_query)
                                            log_info.error("\t\t{} | Percentile={} | Minimum threshold={} | Removing {}".format(m,i_percentile,x,path_query))
                                    break



                    # Plot weights
                    fig, ax = model.plot_weights(title="{} | {} | percentile={}".format(opts.name,m,i_percentile))
                    fig.savefig("{}/{}/percentile_{}/weights.png".format(opts.out_dir,m,i_percentile), dpi=300, format="png", bbox_inches="tight")
                    plt.close()

                    log_info.info("\t\t{} | Percentile={} | Complete".format(m,i_percentile))

                    print("================\n Results \n================", file=f_summary)
                    print("Best hyperparameters:", model.best_hyperparameters_, sep="\t", file=f_summary)
                    print("Best weights from minimum threshold:", best_min_threshold_for_percentile, 2*"\n", sep="\t", file=f_summary)

                    try:
                        print(best_results.to_string(), file=f_summary)
                    except:
                        if best_results_for_percentile is not None:
                            print("\n".join(str(best_results_for_percentile.iloc[0,:]).split("\n")[:-1]), file=f_summary)
                    print(2*"\n", file=f_summary)


                    # Close summary files
                    f_summary.close() # Should this be where the synopsis is?

            # Create next dataset
            if current_iteration+1 < len(opts.percentiles):
                next_percentile = opts.percentiles[current_iteration+1]

                # Get positions
                number_of_features_for_next_iteration = int(round((1.0-next_percentile)*n_features))

                print(m, current_iteration, i_percentile, next_percentile, opts.percentiles, number_of_features_for_next_iteration, opts.min_threshold,  sep="\t")
                # Get feature set
                assert best_weights_for_percentile is not None, "Try this again with --force_overwrite because certain files are being read incorrectly"
                idx_next_feature_set = best_weights_for_percentile.nlargest(number_of_features_for_next_iteration).index

                # Create filepath
                os.makedirs("{}/{}/percentile_{}".format(opts.out_dir, m,next_percentile), exist_ok=True)
                path_next_feature_matrix = "{}/{}/percentile_{}/X.subset.pkl".format(opts.out_dir,m,next_percentile) #Current iteration and not current_itation + 1 because it starts at 1.  It is already offset
                # Check if the file exists already and overwrite if necessary
                update_next_feature_matrix = True
                if os.path.exists(path_next_feature_matrix):
                    if opts.force_overwrite == False:
                        update_next_feature_matrix = False
                if update_next_feature_matrix:
                    X.loc[:,idx_next_feature_set].to_pickle(path_next_feature_matrix, compression=None)
                # Update the features for next iteration
                path_features_current = path_next_feature_matrix

        # Save best configuration
        print("= == === ===== ======= ============ =====================".replace("=","."), file=sys.stderr)
        print("{} | Percentile={} | Minimum threshold = {} | Baseline score = {} | Best score = {} | ∆ = {} | n_features = {}".format(m,best_configuration_for_modeltype[0],best_configuration_for_modeltype[1],to_precision(baseline_percentile_0),to_precision(best_accuracy_for_modeltype),to_precision(best_accuracy_for_modeltype - baseline_percentile_0),len(best_features_for_modeltype)), file=sys.stderr)
        print("= == === ===== ======= ============ =====================".replace("=","."), file=sys.stderr)

        # Copy best configuration
        path_src = "{}/{}/percentile_{}".format(opts.out_dir,m,best_configuration_for_modeltype[0])
        path_dst = "{}/{}__pctl_{}__t_{}".format(path_synopsis,m,best_configuration_for_modeltype[0],best_configuration_for_modeltype[1])

        for file_suffix in ["weights.tsv.gz", "scores.tsv.gz", "scores.png", "weights.class-specific.tsv.gz"]:
            src = "{}/min_threshold_{}/{}".format(path_src,best_configuration_for_modeltype[1],file_suffix)
            if os.path.exists(src):
                shutil.copy2(src="{}/min_threshold_{}/{}".format(path_src,best_configuration_for_modeltype[1],file_suffix),
                             dst="{}__{}".format(path_dst,file_suffix),
                )
        # Copy summary file
        for file_suffix in ["acu.tsv.gz", "hyperparameters.tsv.gz", "summary.txt", "weights.png", "weights.tsv.gz"]:
            src = "{}/{}/percentile_{}/{}".format(opts.out_dir,m,best_configuration_for_modeltype[0],file_suffix)
            if os.path.exists(src):
                shutil.copy2(src=src,
                             dst="{}/{}__pctl_{}__{}".format(path_synopsis,m,best_configuration_for_modeltype[0],file_suffix)
                )

        # Close files
        f_model_synopsis.flush()
        f_model_synopsis.close()

        print("", file=sys.stderr)

    # Analysis summary
    df_summarytable = pd.DataFrame(summary_table)
    df_summarytable.index.name = "iteration"
    df_summarytable.to_csv("{}/output.tsv".format(path_synopsis), sep="\t")
    log_info.info("\t\tTotal time:\t {}".format(format_duration(start_time)))
    for m, df in df_summarytable.groupby("model_type"):
        best_configuration = df.sort_values(["accuracy", "sem", "number_of_features_included"], ascending=[False, True, True]).iloc[0]
        best_configuration.name = " "
        best_configuration.to_csv(sys.stdout, sep="\t")
    # df_summarytable.to_csv(sys.stdout, sep="\t")

# Initialize
if __name__ == "__main__":
    # Check version
    python_version = sys.version.split(" ")[0]
    condition_1 = int(python_version.split(".")[0]) == 3
    condition_2 = int(python_version.split(".")[1]) >= 6
    assert all([condition_1, condition_2]), "Python version must be >= 3.6.  You are running: {}\n{}".format(python_version,sys.executable)
    main()
