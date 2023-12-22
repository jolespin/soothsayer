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

# Version
__version_clairvoyance__ = "v2022.11.28"

# Built-ins
import os, sys, re, multiprocessing, itertools, argparse, subprocess, time, logging, datetime, shutil, copy, warnings
from collections import *
from io import StringIO, TextIOWrapper, BytesIO
import pickle, gzip, bz2


warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
# Blocks the ConvergenceWarning...it's really annoying
# ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.

os.environ['KMP_DUPLICATE_LIB_OK']='True' # In case you have 2 versions of OMP installed

# PyData
import pandas as pd
import numpy as np
import xarray as xr

# ## Plotting
# from matplotlib import use as mpl_backend
# mpl_backend("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns

# Machine learning
from scipy import stats
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import clone

# Parallel
from joblib import Parallel, delayed

# Utility
from tqdm import tqdm

# Soothsayer Utils
from soothsayer_utils import (
    format_duration,
    read_dataframe,
    read_object,
    write_object,
)

pd.set_option('max_colwidth',300)

# Functions
legend_kws = dict(fontsize=10, frameon=True, facecolor="white", edgecolor="black", loc='center left', bbox_to_anchor=(1, 0.5))
to_precision = lambda x, precision=5, return_type=float: return_type(("{0:.%ie}" % (precision-1)).format(x))

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


# ==============================================================================
# Normalization
# ==============================================================================
# Total sum scaling
def transform_tss(X):
    if np.any(X.values < 0):
        raise ValueError("Cannot have negative proportions")
    if X.ndim != 2:
        raise ValueError("Input matrix must have two dimensions")
    if np.all(X == 0, axis=1).sum() > 0:
        raise ValueError("Input matrix cannot have rows with all zeros")
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

# 
# class ClairvoyanceClassification(object):
#     pass 

# class ClairvoyanceRegression(object):
#     pass

# Clairvoyant
class Clairvoyant(object):
    """
    Clairvoyant is an algorithm that weights features that can optimize the predictive capacity of a dataset
    """
    def __init__(
        self,
        # Base
        name=None,
        # Iteration
        n_iter=10,
        # Modeling
        random_state=0,
        algorithm="logistic",
        n_jobs=-1,
        random_mode=0,
        # Logistic Regression
        lr_space={
            "C":np.linspace(1e-10,1.0,11),
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
        self.algorithm = algorithm.lower()
        self.models = self._placeholder_models()
        self.n_jobs = n_jobs
        self.feature_type = feature_type
        self.class_type = class_type
        self.verbose = verbose
        # Metadata
        self.metadata = {
            "name":self.name,
            "n_models":self.n_models,
            "algorithm":self.algorithm
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
        if self.algorithm == "logistic":
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

        if self.algorithm == "tree":
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

        # Iterate through n_iter
        weights = list()
        accuracies = list()

        for i in tqdm(range(self.random_state_value, self.random_state_value + self.n_iter), desc=desc):
            rs = np.random.RandomState(i)
            X_A, X_B, y_A, y_B = model_selection.train_test_split(X,y,test_size=subset,stratify=stratify, random_state=rs)
            params = {"X_A":X_A, "X_B":X_B, "y_A":y_A, "y_B":y_B}
            parallel_results = Parallel(n_jobs=self.n_jobs)(delayed(self._fit_partitions)(**params,
                                                                                          model=model,
                                                                                          weight_type={"logistic":"coef_","tree":"feature_importances_"}[self.algorithm]) for model in self.models)

            # Extract weights and accuracy
            # Convert to array and collapse (X_A and X_B)
            # Store the weights and accuracies

            # Weights
            if self.algorithm == "logistic":
                # Might be able to condense the w for both
                w = np.concatenate(list(map(lambda x: x[:-1], parallel_results)), axis=0) #w.shape ==> (n_space_per_iteration, 2_splits, n_classifications, n_features)
                w = np.concatenate([w[:,0,:,:], w[:,1,:,:]], axis=0) # w shape:  (2*n_space_per_iteration, n_classifications, n_features)
            if self.algorithm == "tree":
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

        if stratify:
            stratify = self.y
        (self.kernel_, self.accuracies_) = self._merge_partitions(X=self.X, y=self.y, stratify=stratify, subset=subset, desc=desc)

        self.shape_ = self.kernel_.shape
        max_accuracy = self.extract_hyperparameters()["accuracy"].max()
        idx_maxacu = self.extract_hyperparameters()["accuracy"][lambda x: x == max_accuracy].index
        if len(idx_maxacu) > 1:
            if self.algorithm == "logistic":
                # Prefer l2 with higher C
                self.best_hyperparameters_ = pd.DataFrame(list(map(dict,idx_maxacu))).sort_values(["penalty", "C"], ascending=[False,False]).iloc[0,:].to_dict()
            elif self.algorithm == "tree":
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
    def extract_kernel(self, into="numpy",  minimum_threshold=None):
        """
        Extract coefficients from training
        if penalty is specific, an array for l1 or l2 loss specifically is output.  Does not support xarray
        """
        minimum_threshold = 0 if minimum_threshold in [False, None, "baseline"] else minimum_threshold


        W = self.kernel_[self.accuracies_ > minimum_threshold]

        if into == "numpy":
            return W

        if self.algorithm == "logistic":
            if into == "xarray":
                if W.shape[1] > 1:
                    return xr.DataArray(W, dims=["index_model", self.class_type, self.feature_type], coords=[range(W.shape[0]), self.class_labels, self.feature_ids])
                else:
                    return xr.DataArray(np.squeeze(W), dims=["index_model",  self.feature_type], coords=[range(W.shape[0]), self.feature_ids])
            if into == "pandas":
                panel = pd.Panel(W,
                             major_axis=pd.Index(range(W.shape[1]), level=0, name=self.class_type),
                             minor_axis=pd.Index(self.feature_ids, level=1, name=self.feature_type))
                df = panel.to_frame().T
                df.index.name = "index"
                return df
        elif self.algorithm == "tree":
            if into == "xarray":
                return xr.DataArray(W, dims=["index_model", self.feature_type], coords=[range(W.shape[0]),  self.feature_ids])
            if into == "pandas":
                return pd.DataFrame(W, index=pd.Index(range(W.shape[0]), name="index_model"), columns=pd.Index(self.feature_ids, name=self.feature_type))

    def extract_accuracies(self, into="numpy"):
        """
        Extract coefficients from training
        if penalty is specific, an array for l1 or l2 loss specifically is output.  Does not support xarray
        """
        if into == "numpy":
            return self.accuracies_
        if into == "pandas":
            return pd.Series(self.accuracies_)

    def extract_weights(self, minimum_threshold=None, ascending=False, func_reduce=np.sum, func_weights=None,  mode=0, threshold_range=(0,0.9)):
        """
        Extract weights
        """
        if all([(mode == 0), (minimum_threshold is None)]):
            mode = 2
        if mode == 0:
            # Extract weights from the kernel
            W = np.abs(self.extract_kernel(into="numpy", minimum_threshold=minimum_threshold))

            if self.algorithm == "logistic":
                weights_ = pd.Series(func_reduce(func_reduce(W, axis=0), axis=0), index=self.feature_ids)
            if self.algorithm == "tree":
                weights_ = pd.Series(func_reduce(W, axis=0), index=self.feature_ids)
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
            if minimum_threshold is not None:
                threshold_values = sorted(set(threshold_values) | set([minimum_threshold]))
            for i in threshold_values:
                w = self.extract_weights(minimum_threshold=i, ascending=None, mode=0)
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

            A = np.abs(self.extract_kernel(into="numpy", minimum_threshold=minimum_threshold))
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


    def cross_validate(self, model="auto", cv=5, cv_labels=None, n_jobs=-1,  early_stopping=100, verbose=True,  minimum_threshold=None, path_save=None, target_score=0, desc="Cross validating model subsets", log_file=sys.stderr, log_prefix=""):
        """
        Notes:
            * Automatically called extract weights?
        """

        self.performance_ = None
        # Hardcoded placeholder
        scoring="accuracy"
        # Model
        if model == "auto":
            if self.algorithm == "logistic":
                model = LogisticRegression(random_state=self.random_state, multi_class="ovr",  **self.best_hyperparameters_, **self.lr_kws)
            elif self.algorithm == "tree":
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

        # Weights
        weights_ = self.extract_weights(minimum_threshold=minimum_threshold, ascending=False )

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
        for i,pos in enumerate(tqdm(range(1, self.X.shape[1]), desc=desc)):
            idx_query_features = weights_.index[:pos]
            if idx_query_features.size > 0:
                Ar_X = self.X.loc[:,idx_query_features].values
                # Original Implementation
                cross_validation_results = model_selection.cross_val_score(model, X=Ar_X, y=Ar_y, cv=cv, n_jobs=n_jobs, scoring=scoring)
                accuracy = np.mean(cross_validation_results)
                sem = stats.sem(cross_validation_results)

                # Update scores
                if i > 0:
                    condition_A = accuracy > max(scores)
                    condition_B = accuracy > target_score
                    condition_C = accuracy > baseline_score #! 2022.11.28 | Handle cases where accuracy is lower than baseline 
                    if all([condition_A, condition_B, condition_C]): #! 2022.11.28

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
                        data = dict(model=model, X=Ar_X, y=Ar_y, cv=cv, idx_features=list(idx_query_features), idx_samples=list(self.X.index), n_jobs=n_jobs, scoring=scoring, random_state=self.random_state)
                        write_object(data, path_save, compression="infer")
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
    # def plot_scores(self, minimum_threshold=None, style="seaborn-white", figsize=(21,5), ax=None, size_scatter=50, ylabel=None,  title=None, title_kws=dict(), legend_kws=dict(), ylim=None, baseline_score=None):

    #     with plt.style.context(style):
    #         df_performance_wrt_included = self.performance_.set_index( "number_of_features_included", drop=True)
    #         weights_ = self.extract_weights(minimum_threshold=minimum_threshold)
    #         scores = df_performance_wrt_included["accuracy"]
    #         df_cvresults = df_performance_wrt_included.iloc[:,4:]
    #         scores = df_cvresults.mean(axis=1)
    #         sems = df_cvresults.sem(axis=1)

    #         # Keywords
    #         _title_kws = dict(fontsize=15, fontweight="bold")
    #         _title_kws.update(title_kws)

    #         legend_fontsize = 8 if df_cvresults.shape[1] > 20 else 10
    #         _legend_kws = {'bbox_to_anchor': (1, 0.5), 'edgecolor': 'black', 'facecolor': 'white', 'fontsize': legend_fontsize, 'frameon': True, 'loc': 'center left'}
    #         _legend_kws.update(legend_kws)

    #         if ax is None:
    #             fig, ax = plt.subplots(figsize=figsize)
    #         else:
    #             fig = plt.gcf()


    #         # Get maximum indicies
    #         idx_maximums= list()
    #         values_maximums = list()

    #         current_maximum = 0
    #         for i,v in scores.iteritems():
    #             if v > current_maximum:
    #                 idx_maximums.append(i)
    #                 values_maximums.append(v)
    #                 current_maximum = v


    #         # Plot each cv accuracy
    #         df_cvresults.plot(color=sns.color_palette(palette="hls", n_colors=df_cvresults.shape[1]), ax=ax, alpha=0.382)
    #         # Plot mean accuracy
    #         ax.plot(scores.index, scores.values, color="black", label="", linewidth=1.618, alpha=0.618)
    #         # Plot progressive steps
    #         ax.scatter(idx_maximums, values_maximums, s=size_scatter, color="darkslategray", marker="o", label="", edgecolor="red", linewidth=1.618, alpha=1)
    #         # Accuracy is the only scoring metric available for the time being
    #         score_metric = "Accuracy" #str(self.performance_.columns[0]).capitalize()

    #         # Index of best score
    #         df_stats = df_performance_wrt_included.loc[:,["accuracy", "sem"]].sort_values(["accuracy", "sem"], ascending=[False, True])
    #         idx_bestscore = df_stats.index[0]
    #         best_score = df_stats.loc[idx_bestscore,"accuracy"]
    #         sem_at_best_score = df_stats.loc[idx_bestscore, "sem"]

    #         # Lines
    #         ax.axhline(best_score, linestyle=":", color="black", label= "%s: %0.3f ± %0.3f"%(score_metric, best_score, sem_at_best_score))
    #         ax.axvline(idx_bestscore, linestyle=":", color="black", label="n= {} features".format(idx_bestscore))

    #         if baseline_score is not None:
    #             ax.axhline(baseline_score, color="maroon", linestyle=":", label="Baseline: {}".format(baseline_score))

    #         # Labels
    #         if ylabel is None:
    #             ylabel = score_metric
    #         ax.set_xlabel("Number of features included")
    #         ax.set_ylabel(ylabel, fontsize=12)
    #         ax.set_xlabel(ax.get_xlabel(), fontsize=12)
    #         ax.legend(**_legend_kws)
    #         if title is not None:
    #             ax.set_title(title, _title_kws)
    #         if ylim:
    #             ax.set_ylim(ylim)
    #         pad = 0.1
    #         ax.set_xlim((scores.index.min()-pad, scores.index.max()+pad))
    #         return fig, ax


    # def plot_weights(self,  ax=None, style="seaborn-white",  title=None, title_kws=dict(),  axis_kws=dict(), tick_kws=dict(), figsize=(21,5), palette="hls", *args):
    #     """
    #     Plot the weights
    #     """
    #     _title_kws = {"fontsize":15, "fontweight":"bold"}
    #     _title_kws.update(title_kws)

    #     _axis_kws = {"fontsize":15}
    #     _axis_kws.update(axis_kws)

    #     _tick_kws = {"fontsize":12}
    #     _tick_kws.update(tick_kws)
    #     with plt.style.context(style):
    #         if ax is None:
    #             fig, ax = plt.subplots(figsize=figsize)
    #         else:
    #             fig = plt.gcf()

    #         df_w = self.extract_weights(mode=1, *args)
    #         color = sns.color_palette(palette=palette, n_colors=df_w.shape[1])
    #         df_w.plot(color=color, ax=ax)
    #         ax.legend_.remove()
    #         ax.set_xlabel("Threshold", **_axis_kws)
    #         ax.set_ylabel("$W$", **_axis_kws)
    #         ax.set_xticklabels(ax.get_xticks(), **_tick_kws)
    #         ax.set_yticklabels(ax.get_yticks(), **_tick_kws)
    #         ax.set_xlim((df_w.index.min(), df_w.index.max()))
    #         if title is not None:
    #             ax.set_title(title, _title_kws)
    #         return fig, ax

    # Writing serialized object
    def export_model(self, path, compression="infer"):
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
    parser.add_argument('-X', '--feature_table', type=str, help = 'Input: Path/to/Tab-separated-value.tsv of feature matrix (rows=samples, cols=features)')
    parser.add_argument('-y', '--target_vector', type=str, help = 'Input: Path/to/Tab-separated-value.tsv of target vector (rows=samples, column=integer target)')

    # Transformation
    parser.add_argument("--transformation", type=str, default=None, help="Transform the feature matrix.  Valid Arguments: {'tss', 'clr'}")
    parser.add_argument("--multiplicative_replacement", type=float, default=None, help="Multiplicative replacement used for CLR [Default: 1/m^2]")

    # Hyperparameters
    parser.add_argument("-a", "--algorithm", type=str, default="logistic,tree", help="Model types in ['logistic','tree'] represented by a comma-separated list [Default: 'logistic,tree']")
    parser.add_argument("-n", "--n_iter", type=int, default=10, help="Number of iterations to split data and fit models. [Default: 10]")
    parser.add_argument("--random_state", type=int, default=0, help="[Default: 0]")
    parser.add_argument("-p", "--n_jobs", type=int, default=1, help="[Default: 1]")
    parser.add_argument("-T", "--minimum_threshold", type=str, default="0.0,None", help="Minimum accuracy for models when considering coeficients for weights. Can be a single float or list of floats separated by `,` ranging from 0-1. If None it takes an aggregate weight.  [Default: 0.0,None]")
    parser.add_argument("--random_mode", type=int, default=0, help="Random mode 0,1, or 2.  0 uses the same random_state each iteration.  1 uses a different random_state for each parameter set.  2 uses a different random_state each time. [Default: 0]")
    parser.add_argument("-P", "--percentiles", type=str, default="0.0,0.3,0.5,0.75,0.9,0.95,0.96,0.97,0.98,0.99", help="Iterative mode takes in a comma-separated list of floats from 0-1 that dictate how many of the features will be used [Default: 0.0,0.3,0.5,0.75,0.9,0.95,0.96,0.97,0.98,0.99]")
    parser.add_argument("--solver", type=str, default="liblinear", help="LogisticRegression solver {‘liblinear’, ‘sag’, ‘saga’} [Default: liblinear]")


    # Labels & Directories
    parser.add_argument("-o", "--out_dir", type=str, default = os.getcwd(), help = "Output: Path/to/existing-directory-for-output.  Do not use `~` to specify /home/[user]/ [Default: cwd]" )
    parser.add_argument("--feature_type", type=str, default="feature", help="Name of feature type (e.g. gene, orf, otu, etc.) [Default: 'feature']")
    parser.add_argument("--class_type", type=str, default="class", help="Name of class type (e.g. phenotype, MOA, etc.) [Default: 'class']")

    # Validation
    parser.add_argument("--cv", default=5, type=str, help="Number of steps for percentiles when cross validating and dropping features.  If cv=0 then it skips cross-validation step. [Default: 5]")
    parser.add_argument("-e", "--early_stopping", type=int, default=100, help="Stopping the algorthm if a certain number of iterations do not increase the accuracy. Use -1 if you don't want to use `early_stopping` [Default: 100]")
    parser.add_argument("-m", "--metric", type=str, default="accuracy", help="Performance metric {accuracy, precision, recall, f1, f_beta} [Default: accuracy]")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta used for F-beta score [Default: accuracy]")
    parser.add_argument("--average", type=str, default=None, help="This parameter is required for multiclass/multilabel targets.  [Default: binary if 2 classes else macro]")
    parser.add_argument("-l", "--positive_label", type=str,  help="The class to report if average='binary' and the data is binary. If the data are multiclass or multilabel, this will be ignored.  Required if binary data.")

    # I/O Model
    parser.add_argument("--export_netcdf", action="store_true", help="Export the kernel as netcdf [Default does not]")
    parser.add_argument("--export_model", action="store_true", help="Export the model as pickle [Default does not]")

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

    # Early stopping
    if opts.early_stopping == -1:
        opts.early_stopping = np.inf

    # Minimum threshold
    if "," in opts.minimum_threshold:
        opts.minimum_threshold = opts.minimum_threshold.lower()
        special_values = list()
        if "none" in opts.minimum_threshold:
            opts.minimum_threshold = opts.minimum_threshold.replace("none","")
            special_values.append(None)
        opts.minimum_threshold = sorted(map(float, filter(bool, opts.minimum_threshold.split(",")))) + special_values
    else:
        if opts.minimum_threshold == "none":
            opts.minimum_threshold = [None]
        else:
            opts.minimum_threshold = [float(opts.minimum_threshold)]

    # Percentiles
    if "," in opts.percentiles:
        opts.percentiles = sorted(map(float, filter(bool, opts.percentiles.split(","))))
    else:
        if opts.percentiles == "":
            opts.percentiles = [0.0]
        else:
            opts.percentiles = [float(opts.percentiles)]
    if 0.0 not in opts.percentiles:
        opts.percentiles = [0.0] + opts.percentiles
    assert all(map(lambda x: 0.0 <= x < 1.0, opts.percentiles)), "All percentiles must meet the following criteria: 0.0 <= x < 1.0"

    # Model types
    if "," in opts.algorithm:
        opts.algorithm = list(filter(bool, opts.algorithm.split(",")))
    else:
        opts.algorithm = [opts.algorithm]
    assert set(opts.algorithm) <= set(["logistic", "tree"]), "algorithm should only be a combination of ['logistic','tree']. Current input: {}".format(opts.algorithm)



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


    # Static data
    X_original = read_dataframe(opts.feature_table,  func_index=str, func_columns=str, verbose=False)

    y = read_dataframe(opts.target_vector,  func_index=str, func_columns=str, verbose=False)
    if isinstance(y, pd.Series):
        y = y.to_frame()
    y = y.iloc[:,0]

    # Scoring
    number_of_classes = y.nunique()
    if number_of_classes == 2:
        assert opts.positive_label is not None, "Must provide --positive_label if -y has only 2 classes"
        assert opts.positive_label in y.unique(), "--positive_label must be in in -y"

    acceptable_arguments = {"accuracy", ""}

    # Iterate through model types
    best_configuration = None
    baseline_scores_ = dict()
    summary_table = list()

    for m in opts.algorithm:
        print("= == === ===== ======= ============ =====================", file=sys.stderr)
        print(m, file=sys.stderr)
        print("= == === ===== ======= ============ =====================".replace("=","."), file=sys.stderr)


        estimator = {"logistic":LogisticRegression(random_state=opts.random_state, multi_class='ovr', solver=opts.solver), "tree":DecisionTreeClassifier(random_state=opts.random_state)}[m]
        # Placeholders for dynamic data
        path_features_current = opts.feature_table
        best_configuration_for_algorithm = None # (percentile, minimum_threshold, hyperparameters)
        best_accuracy_for_algorithm = 0
        best_hyperparameters_for_algorithm = None
        model = None
        best_features_for_algorithm = None

        log_crossvalidation = create_logfile(m,
                                                 "{}/{}__cross-validation.log".format(path_synopsis,m),
                                                 mode=determine_mode_for_logfiles("{}/{}__cross-validation.log".format(path_synopsis,m), opts.force_overwrite)
        )
        print("Opening {} synopsis:".format(m), "{}/{}__synopsis.tsv".format(path_synopsis,m))
        f_model_synopsis = open("{}/{}__synopsis.tsv".format(path_synopsis,m), "w", buffering=1)
        print("algorithm", "hyperparameters", "percentile",  "minimum_threshold", "baseline_score", "baseline_score_of_current_percentile", "accuracy", "sem", "delta", "random_state", "number_of_features_included", "feature_set", "clairvoyance_weights", sep="\t", file=f_model_synopsis)

        # Iterate through percentiles of weighted features
        baseline_percentile_0 = None
        for current_iteration, i_percentile in enumerate(opts.percentiles, start=0):

            # ==================================================================
            # Load previous analysis
            # ==================================================================
            # Check if directory exists, make one if not
            log_info.info("\t\t{} | Percentile={} | Creating temporary directories in {}".format(m,i_percentile,opts.out_dir))
            os.makedirs("{}/{}/percentile_{}".format(opts.out_dir,m,i_percentile), exist_ok=True)

            f_summary = open("{}/{}/percentile_{}/summary.txt".format(opts.out_dir,m,i_percentile), "w")
            # Load features
            log_info.info("\t\t{} | Percentile={} | Loading features datasets".format(m,i_percentile ))

            # Terrible hack but this will be addressed in future versions
            features_for_current_iteration = read_dataframe(path_features_current,   func_index=str, func_columns=str, verbose=False).columns
            X = X_original[features_for_current_iteration]
            if X.shape[1] < 2:
                print("Skipping current iteration {} because there is/are only {} feature[s]".format(i_percentile, X.shape[1]), file=sys.stderr)
            else:
                # transform
                if opts.transformation is not None:
                    if opts.transformation == "clr":
                        if opts.multiplicative_replacement is None:
                            if np.any(X.values == 0):
                                log_info.info("Detected zeros in -X. Using default multiplicative replacement 1/m^2")
                                opts.multiplicative_replacement = 1/X.shape[1]**2
                            else:
                                log_info.info("No zeros detected in -X. Setting multiplicative replacement to 0")
                                opts.multiplicative_replacement = 0
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

                # ==================================================================
                # Analysis
                # ==================================================================

                # Run model
                model = Clairvoyant(
                            algorithm=m,
                            n_iter=opts.n_iter,
                            feature_type=opts.feature_type,
                            class_type=opts.class_type,
                            n_jobs=opts.n_jobs,
                            verbose=True,
                            random_state=opts.random_state,
                            random_mode=opts.random_mode,
                            lr_kws=dict(max_iter=10000, solver=opts.solver),
                            tree_kws={},
                )
                log_info.info("\t\tFitting data")

                model.fit(X, y, desc="{} | Percentile={} | Permuting samples and fitting models".format(m, i_percentile))
                log_info.info("\t\t{} | Percentile={} | Best hyperparameters from fitting: {}".format(m,i_percentile,model.best_hyperparameters_))
                log_info.info("\t\t{} | Percentile={} | Saving model: {}/{}/percentile_{}/model.pkl".format(m,i_percentile,opts.out_dir,m,i_percentile))
                if opts.export_model:
                    model.export_model(path="{}/{}/percentile_{}/model.pkl".format(opts.out_dir,m,i_percentile), compression=None)


                # Get kernel and weights
                kernel_ = model.extract_kernel(into="xarray")
                acu_ = model.extract_accuracies(into="pandas")
                hyperparameters_ = model.extract_hyperparameters()
                W_ = model.extract_weights(minimum_threshold=None, mode=1) # Future: Incldue any other minimum_threshold values that aren't in the linspace

                log_info.info("\t\t{} | Percentile={} | Calculating weights".format(m,i_percentile))

                # Set parameters for estimator and calculate baseline score
                estimator.set_params(**model.best_hyperparameters_)
                baseline_score = model_selection.cross_val_score(estimator, X=X.values, y=y.values, cv=opts.cv).mean()
                baseline_scores_[(m,i_percentile)] = baseline_score
                features_baseline = X.columns.tolist()

                # Set original feature size and baseline score
                if i_percentile == 0.0:
                    n_features = X.shape[1]
                    baseline_percentile_0 = baseline_score
                print("{} | Percentile={} | X.shape = {} | Baseline score({}) = {}".format(m,i_percentile,X.shape,path_features_current,to_precision(baseline_score)), file=sys.stderr)

                acu_.to_frame("Accuracy").to_csv("{}/{}/percentile_{}/acu.tsv.gz".format(opts.out_dir,m,i_percentile), sep="\t", compression="gzip")
                hyperparameters_.to_csv("{}/{}/percentile_{}/hyperparameters.tsv.gz".format(opts.out_dir,m,i_percentile), sep="\t", compression="gzip")
                W_.to_csv("{}/{}/percentile_{}/weights.tsv.gz".format(opts.out_dir,m,i_percentile ), sep="\t", compression="gzip")



                if opts.export_netcdf:
                    kernel_.to_netcdf("{}/{}/percentile_{}/kernel.nc".format(opts.out_dir, m, i_percentile))

                # Determine if weights are different
                tol_difference = 1
                if len(opts.minimum_threshold) > 1:
                    _current_order = np.asarray(model.extract_weights(minimum_threshold=opts.minimum_threshold[0], ascending=False).index)
                    for x in opts.minimum_threshold[1:]:
                        _query_order = np.asarray(model.extract_weights(minimum_threshold=x, ascending=False).index)
                        num_differences = np.sum(_current_order != _query_order)
                        if num_differences < tol_difference*2:  # Times 2 because differences come in pairs
                            opts.minimum_threshold.remove(x)
                            log_info.info("\t\t{} | Percentile={} | Removing min_accuracy={} from computation because {} ordering is already present from another threshold".format(m,i_percentile,x,model.feature_type))
                        else:
                            _current_order = _query_order
                # Soothsayer | feature Finder
                print("================================\nsoothsayer:clairvoyance {}\n================================".format(__version_clairvoyance__), file=f_summary)
                print("X:", path_features_current, sep="\t", file=f_summary)
                print("y:", opts.target_vector, sep="\t", file=f_summary)
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
                print("\n".join(str(params[["n_iter", "random_state", "random_mode", "n_jobs", "minimum_threshold"]]).split("\n")[:-1]), file=f_summary)
                print("================\n Labels \n================", file=f_summary)
                print("\n".join(str(params[["feature_type", "class_type"]]).split("\n")[:-1]), file=f_summary)
                print("================\n Cross Validation \n================", file=f_summary)

                print("\n".join(str(params[[  "early_stopping"]]).split("\n")[:-1]), file=f_summary)


                print("cv", cv_repr, sep="\t", file=f_summary)


                # Placeholders
                best_accuracy_for_percentile = 0
                best_minimum_threshold_for_percentile = None
                best_results_for_percentile = None
                best_weights_for_percentile = None
                # Run crossvalidation
                run_crossvalidation = opts.cv != 0
                if run_crossvalidation:
                    # Cross Validation
                    for x in opts.minimum_threshold:
                        # Default to run the analysis.  First check if the path exists if does say if you want to overwrite.  If not then make the directory and do the analysis
                        run_analysis = True
                        path_query = "{}/{}/percentile_{}/minimum_threshold_{}".format(opts.out_dir,m,i_percentile,x)
                        if os.path.exists(path_query):
                            # Is there directory empty?
                            if len(os.listdir(path_query)) > 0: # or os.path.getsize > ?
                                if opts.force_overwrite == False:
                                    run_analysis = False
                        else:
                            os.makedirs(path_query, exist_ok=True)

                        if run_analysis:
                            # Calculate weights
                            w_ = model.extract_weights(minimum_threshold=x,  ascending=False)

                            # Are there weights for this threshold?
                            if not w_.isnull().any():
                                weights_ = w_.to_frame("weights")
                                weights_.to_csv("{}/weights.tsv.gz".format(path_query), sep="\t", compression="gzip")

                                # NOTE: This is a hack for outputing the weights for each class but only works for logistic regression.  Once this tree methods are converted to OneVsRest then this can be properly integrated
                                if m == "logistic":
                                    if len(model.class_labels) > 2:
                                        weights_class_specific_ = model.extract_weights(minimum_threshold=x,  mode=3)
                                        weights_class_specific_.to_csv("{}/weights.class-specific.tsv.gz".format(path_query), sep="\t", compression="gzip")

                                log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Cross validation".format(m,i_percentile,x))
                                # Synthesize Datasets

                                path_save = "{}/model-data.pkl".format(path_query)

                                scores_ = model.cross_validate(model=clone(estimator),
                                                                minimum_threshold=x,
                                                                cv=opts.cv,
                                                                cv_labels=cv_labels,
                                                                early_stopping=opts.early_stopping,
                                                                n_jobs=opts.n_jobs,
                                                                verbose=True,
                                                                path_save=path_save,
                                                                target_score=best_accuracy_for_algorithm,
                                                                desc="{} | CV | Percentile={} | Minimum threshold={}".format(m,i_percentile,x),
                                                                log_file=log_crossvalidation,
                                                                log_prefix = "{} | pctl={} | t={}".format(m,i_percentile,x)
                                                                )

                                log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Writing scores".format(m,i_percentile,x))
                                scores_.to_csv("{}/scores.tsv.gz".format(path_query), sep="\t", compression="gzip")

                                # Update the scores
                                idx_best_accuracy_for_minimum_threshold = scores_["accuracy"].sort_values(ascending=False).index[0]
                                query_accuracy = scores_.loc[idx_best_accuracy_for_minimum_threshold,"accuracy"]
                                query_sem = scores_.loc[idx_best_accuracy_for_minimum_threshold,"sem"]
                                feature_set = scores_.loc[idx_best_accuracy_for_minimum_threshold, "feature_set"]

                                # Update accuracies for current percentile
                                if query_accuracy > baseline_score:
                                    if query_accuracy > best_accuracy_for_percentile:
                                        best_minimum_threshold_for_percentile = x
                                        #! Remove
                                        # ----------
                                        if scores_.shape[0] >= 10: 
                                            best_results_for_percentile = scores_.sort_values(["accuracy", "sem", "number_of_features_included"], ascending=[False,True, False]).iloc[:10,:3]
                                        else:
                                            best_results_for_percentile = scores_.sort_values(["accuracy", "sem", "number_of_features_included"], ascending=[False,True, False]).iloc[:scores_.shape[0],:3]
                                        # -----------

                                        best_accuracy_for_percentile = query_accuracy
                                        best_weights_for_percentile = w_.copy()
                                        log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Updating the model object with accuracy={}".format(m,i_percentile,x,query_accuracy))

                                        # May be slower than necessary to put this here but it's safer
                                        if opts.export_model:
                                            model.export_model(path="{}/{}/percentile_{}/model.pkl".format(opts.out_dir,m,i_percentile), compression=None)

                                        # Update accuracies for current model
                                        if query_accuracy > best_accuracy_for_algorithm:
                                            best_configuration_for_algorithm = (i_percentile, x)
                                            best_hyperparameters_for_algorithm = model.best_hyperparameters_
                                            best_accuracy_for_algorithm = query_accuracy
                                            best_features_for_algorithm = feature_set
                                else:
                                    print("{} | Percentile={} | X.shape = {} | Baseline score({}) = {} was higher than all subsets at minimum threshold {}".format(m, i_percentile, X.shape,path_features_current,to_precision(baseline_score), x), file=sys.stderr)
                                    x = "baseline"
                                    best_configuration_for_algorithm = (i_percentile, x)
                                    best_hyperparameters_for_algorithm = model.best_hyperparameters_
                                    best_accuracy_for_algorithm = baseline_score
                                    best_features_for_algorithm = features_baseline
                                    best_accuracy_for_percentile = baseline_score
                                    best_weights_for_percentile = w_.copy()

                                log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Creating plot".format(m,i_percentile,x))

                                # fig, ax = model.plot_scores(title="{} | percentile={} | minimum_threshold={}".format(m,i_percentile,x), ylim=(0,1), minimum_threshold=x, baseline_score=baseline_score)
                                # fig.savefig("{}/scores.png".format(path_query), dpi=300, format="png", bbox_inches="tight")
                                # plt.close()

                                # Synthesize summary table
                                Se_query = pd.Series([m, model.best_hyperparameters_, i_percentile, x, baseline_percentile_0, baseline_score, query_accuracy,  query_sem, query_accuracy - baseline_percentile_0, opts.random_state, len(feature_set), feature_set, best_weights_for_percentile[feature_set].tolist()],
                                                index=["algorithm", "hyperparameters", "percentile",  "minimum_threshold", "baseline_score", "baseline_score_of_current_percentile", "accuracy", "sem", "delta", "random_state", "number_of_features_included", "feature_set", "clairvoyance_weights"])
                                print(*Se_query.values, sep="\t", file=f_model_synopsis)
                                summary_table.append(Se_query)

                            else:

                                log_info.error("\t\t{} | Percentile={} | Minimum threshold={} | No models had an accuracy for this threshold so the weights were NaN".format(m,i_percentile,x))
                                if os.path.exists(path_query):
                                    if len(os.listdir(path_query)) == 0:
                                        os.rmdir(path_query)
                                        log_info.error("\t\t{} | Percentile={} | Minimum threshold={} | Removing {}".format(m,i_percentile,x,path_query))
                                break



                # # Plot weights
                # fig, ax = model.plot_weights(title="{} | percentile={}".format(m,i_percentile))
                # fig.savefig("{}/{}/percentile_{}/weights.png".format(opts.out_dir,m,i_percentile), dpi=300, format="png", bbox_inches="tight")
                # plt.close()

                log_info.info("\t\t{} | Percentile={} | Complete".format(m,i_percentile))

                print("================\n Results \n================", file=f_summary)
                print("Best hyperparameters:", model.best_hyperparameters_, sep="\t", file=f_summary)
                print("Best weights from minimum threshold:", best_minimum_threshold_for_percentile, 2*"\n", sep="\t", file=f_summary)

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

                print(m, current_iteration, i_percentile, next_percentile, opts.percentiles, number_of_features_for_next_iteration, opts.minimum_threshold,  sep="\t")
                # Get feature set
                assert best_weights_for_percentile is not None, "Try this again with --force_overwrite because certain files are being read incorrectly"
                idx_next_feature_set = best_weights_for_percentile.nlargest(number_of_features_for_next_iteration).index

                # Create filepath
                os.makedirs("{}/{}/percentile_{}".format(opts.out_dir, m,next_percentile), exist_ok=True)
                path_next_feature_table = "{}/{}/percentile_{}/X.subset.pkl".format(opts.out_dir,m,next_percentile) #Current iteration and not current_itation + 1 because it starts at 1.  It is already offset
                # Check if the file exists already and overwrite if necessary
                update_next_feature_table = True
                if os.path.exists(path_next_feature_table):
                    if opts.force_overwrite == False:
                        update_next_feature_table = False
                if update_next_feature_table:
                    X.loc[:,idx_next_feature_set].to_pickle(path_next_feature_table, compression=None)
                # Update the features for next iteration
                path_features_current = path_next_feature_table

        # Save best configuration
        print("= == === ===== ======= ============ =====================".replace("=","."), file=sys.stderr)
        print("{} | Percentile={} | Minimum threshold = {} | Baseline score = {} | Best score = {} | ∆ = {} | n_features = {}".format(m,best_configuration_for_algorithm[0],best_configuration_for_algorithm[1],to_precision(baseline_percentile_0),to_precision(best_accuracy_for_algorithm),to_precision(best_accuracy_for_algorithm - baseline_percentile_0),len(best_features_for_algorithm)), file=sys.stderr)
        print("= == === ===== ======= ============ =====================".replace("=","."), file=sys.stderr)

        # Copy best configuration
        path_src = "{}/{}/percentile_{}".format(opts.out_dir,m,best_configuration_for_algorithm[0])
        path_dst = "{}/{}__pctl_{}__t_{}".format(path_synopsis,m,best_configuration_for_algorithm[0],best_configuration_for_algorithm[1])

        for file_suffix in ["weights.tsv.gz", "scores.tsv.gz", "scores.png", "weights.class-specific.tsv.gz"]:
            src = "{}/minimum_threshold_{}/{}".format(path_src,best_configuration_for_algorithm[1],file_suffix)
            if os.path.exists(src):
                shutil.copy2(src="{}/minimum_threshold_{}/{}".format(path_src,best_configuration_for_algorithm[1],file_suffix),
                             dst="{}__{}".format(path_dst,file_suffix),
                )
        # Copy summary file
        for file_suffix in ["acu.tsv.gz", "hyperparameters.tsv.gz", "summary.txt", "weights.png", "weights.tsv.gz"]:
            src = "{}/{}/percentile_{}/{}".format(opts.out_dir,m,best_configuration_for_algorithm[0],file_suffix)
            if os.path.exists(src):
                shutil.copy2(src=src,
                             dst="{}/{}__pctl_{}__{}".format(path_synopsis,m,best_configuration_for_algorithm[0],file_suffix)
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
    for m, df in df_summarytable.groupby("algorithm"):
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
