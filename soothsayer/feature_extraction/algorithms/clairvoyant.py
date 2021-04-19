# Built-ins
import os, sys, itertools, time, copy, logging
from collections import OrderedDict, defaultdict, Mapping

# PyData
import pandas as pd
import numpy as np
import xarray as xr

## Plotting
import matplotlib.pyplot as plt
import seaborn as sns

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

# Soothsayer
from soothsayer.utils import *
from soothsayer.io import read_object, write_object

# Clairvoyant
class Clairvoyant(object):
    """
    Clairvoyant is an algorithm that weights attributes that can optimize the predictive capacity of a dataset

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
    (10) Added attribute set to scores output
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
    Randomly sample attributes

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
        attr_type="attr",
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
        self.attr_type = attr_type
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
        return f"{class_name}({str(self.metadata)[1:-1]})"
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
            assert all([*map(lambda x: x in self.lr_space, ["penalty", "C"])]), "`lr_kws` must contain `penalty` and `C`"
            param_set = [*map(lambda x:dict(zip(self.lr_space.keys(), x)), itertools.product(*self.lr_space.values()))]
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
            assert all([*map(lambda x: x in self.tree_space, ["criterion", "max_features", "min_samples_leaf"])]), "`tree_kws` must contain `criterion`, `max_features`, and `min_samples_leaf`"
            param_set = [*map(lambda x:dict(zip(self.tree_space.keys(), x)), itertools.product(*self.tree_space.values()))]
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
        # (n_space_per_iteration, 2_splits, n_classifications, n_attributes)
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
                w = np.concatenate([*map(lambda x: x[:-1], parallel_results)], axis=0) #w.shape ==> (n_space_per_iteration, 2_splits, n_classifications, n_attributes)
                w = np.concatenate([w[:,0,:,:], w[:,1,:,:]], axis=0) # w shape:  (2*n_space_per_iteration, n_classifications, n_attributes)
            if self.model_type == "tree":
                w = np.asarray([*map(lambda x: x[0], parallel_results)])
                w = np.concatenate([w[:,0], w[:,1]], axis=0)
            weights.append(w)
            # Accuracies
            acu = np.asarray([*map(lambda x: x[-1], parallel_results)]) #acu shape ==> (n_space_per_iteration, 2_splits)
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
        self.obsv_ids = self.X.index
        self.attr_ids = self.X.columns
        self.shape_input_ = self.X.shape
        self.class_labels = sorted(y.unique())
        self.class_labels_encoding = self.class_labels
        if self.map_encoding is not None:
            self.class_labels = [*map(lambda x:self.map_encoding[x], self.class_labels)]
        if stratify:
            stratify = self.y
        (self.kernel_, self.accuracies_) = self._merge_partitions(X=self.X, y=self.y, stratify=stratify, subset=subset, desc=desc)

        self.shape_ = self.kernel_.shape
        max_accuracy = self.extract_hyperparameters()["accuracy"].max()
        idx_maxacu = self.extract_hyperparameters()["accuracy"][lambda x: x == max_accuracy].index
        if len(idx_maxacu) > 1:
            if self.model_type == "logistic":
                # Prefer l2 with higher C
                self.best_hyperparameters_ = pd.DataFrame([*map(dict,idx_maxacu)]).sort_values(["penalty", "C"], ascending=[False,False]).iloc[0,:].to_dict()
            elif self.model_type == "tree":
                # Prefer lower min_samples_leaf, gini, and sqrt
                self.best_hyperparameters_ = pd.DataFrame([*map(dict,idx_maxacu)]).sort_values(["min_samples_leaf", "criterion", "max_features"], ascending=[True,False, False]).iloc[0,:].to_dict()
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
                    return xr.DataArray(W, dims=["index_model", self.class_type, self.attr_type], coords=[range(W.shape[0]), self.class_labels, self.attr_ids], name=name)
                else:
                    return xr.DataArray(np.squeeze(W), dims=["index_model",  self.attr_type], coords=[range(W.shape[0]), self.attr_ids], name=name)
            if into == "pandas":
                panel = pd.Panel(W,
                             major_axis=pd.Index(range(W.shape[1]), level=0, name=self.class_type),
                             minor_axis=pd.Index(self.attr_ids, level=1, name=self.attr_type))
                df = panel.to_frame().T
                df.index.name = "index"
                return df
        elif self.model_type == "tree":
            if into == "xarray":
                return xr.DataArray(W, dims=["index_model", self.attr_type], coords=[range(W.shape[0]),  self.attr_ids], name=name)
            if into == "pandas":
                return pd.DataFrame(W, index=pd.Index(range(W.shape[0]), name="index_model"), columns=pd.Index(self.attr_ids, name=self.attr_type))

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
                weights_ = pd.Series(func_reduce(func_reduce(W, axis=0), axis=0), index=self.attr_ids, name=name)
            if self.model_type == "tree":
                weights_ = pd.Series(func_reduce(W, axis=0), index=self.attr_ids, name=name)
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
                                columns=pd.Index(self.attr_ids, name=self.attr_type)).T



    def extract_hyperparameters(self, ascending=None):
        index = pd.Index([*map(frozenset, self.param_index.keys())], dtype="category")
        data = [*map(np.asarray, self.param_index.values())]
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

        # Attribute iterables
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
        label_attrs_included = f"num_{str(self.attr_type)}_included"
        label_attrs_excluded = f"num_{str(self.attr_type)}_excluded"

        # Cross validation
        scores = list()
        Ar_y = self.y.values

        # Baseline
        baseline_score  = np.mean(model_selection.cross_val_score(model, X=self.X.values, y=Ar_y, cv=cv, n_jobs=n_jobs, scoring=scoring))
        early_stopping_counter = 0
        for i,pos in enumerate(iterable):
            idx_query_attrs = weights_.index[:pos]
            if idx_query_attrs.size > 0:
                Ar_X = self.X.loc[:,idx_query_attrs].values
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
                            entry = "\t".join([str(log_prefix), f"Baseline={to_precision(baseline_score)}", f"iteration={i}", f"index={pos}", f"{label_attrs_included}={idx_query_attrs.size}", f"Accuracy={to_precision(accuracy)}", f"∆={to_precision(accuracy - baseline_score)}", f"SEM={to_precision(sem)}"])
                            if type(log_file) == logging.Logger:
                                log_file.info(entry)
                            if self.verbose:
                                if hasattr(log_file, 'write'):
                                    print(entry, file=log_file)
                        if path_save is not None:
                            save_data = dict(model=model, X=Ar_X, y=Ar_y, cv=cv, idx_attrs=list(idx_query_attrs), idx_samples=list(self.X.index), n_jobs=n_jobs, scoring=scoring, random_state=self.random_state)
                            write_object(save_data, path_save, compression="infer")
                        # Reset the Counter
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1
                # Organize results for dataframe
                cv_summary[pos][scoring] = accuracy
                cv_summary[pos]["sem"] = sem
                cv_summary[pos][label_attrs_included] = idx_query_attrs.size
                cv_summary[pos][f"{self.attr_type}_set"] = list(idx_query_attrs)

                # Individual cross-validations
                cv_results[pos] = pd.Series(cross_validation_results, index=[*map(lambda x:f"cv={x}", range(1, len(cross_validation_results)+1))])

                # Append scores
                scores.append(accuracy)

                if early_stopping_counter >= early_stopping:
                    if log_file is not None:
                        entry = f"Early stopping at iteration={i} with maximum score= {max(scores)}"
                        if type(log_file) == logging.Logger:
                            log_file.info(entry)
                        if hasattr(log_file, 'write'):
                            if self.verbose:
                                print(entry, file=log_file)
                    break
        # DataFrame for number of attributes included
        self.performance_ = pd.DataFrame(cv_summary).T
        self.performance_[label_attrs_included] = self.performance_[label_attrs_included].astype(int)
        self.performance_[label_attrs_excluded] = (self.X.shape[1] - self.performance_[label_attrs_included]).astype(int)
        self.performance_ = self.performance_.loc[:,[scoring, "sem", label_attrs_included, label_attrs_excluded, f"{self.attr_type}_set"]]

        # DataFrame for individual cross-validation results
        df_cvresults = pd.DataFrame(cv_results).T

        if cv_labels is not None:
            df_cvresults.columns = cv_labels
        self.performance_ = pd.concat([self.performance_, df_cvresults], axis=1)

        # Best attributes
        self.best_attrs_ = self.performance_.sort_values(["accuracy", "sem"], ascending=[False, True]).iloc[0,:][f"{self.attr_type}_set"]

        return self.performance_

    # =================================================================================
    # Plotting
    # =================================================================================
    def plot_scores(self, min_threshold=None, style="seaborn-white", figsize=(21,5), ax=None, size_scatter=50, ylabel=None,  title=None, title_kws=dict(), legend_kws=dict(), ylim=None, baseline_score=None):

        with plt.style.context(style):
            df_performance_wrt_included = self.performance_.set_index( f"num_{str(self.attr_type)}_included", drop=True)
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
            ax.axvline(idx_bestscore, linestyle=":", color="black", label=f"n= {idx_bestscore} {self.attr_type}s")

            if baseline_score is not None:
                ax.axhline(baseline_score, color="maroon", linestyle=":", label=f"Baseline: {baseline_score}")

            # Labels
            if ylabel is None:
                ylabel = score_metric
            ax.set_xlabel(f"Number of {self.attr_type}s included")
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
