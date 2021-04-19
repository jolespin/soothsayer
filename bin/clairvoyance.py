#!/usr/bin/env python
# =============================
# Creator: Josh L. Espinoza (J. Craig Venter Institute)
# Date[0]:Init: 2018-April-02
# Date[-1]:Current: 2018-August-28
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
__version_clairvoyance__ = "v0.3_2018-08-28"

# Built-ins
import os, sys, multiprocessing, argparse, time, shutil, warnings
from ast import literal_eval
# warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning: Series.compress(condition) is deprecated. Use 'Series[condition]' or 'np.asarray(series).compress(condition)' instead.
# warnings.simplefilter(action='ignore', category=UserWarning) # UserWarning: Attempting to set identical left==right results in singular transformations; automatically expanding
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK']='True' # In case you have 2 versions of OMP installed

# PyData
import pandas as pd
import numpy as np

## Plotting
from matplotlib import use as mpl_backend
mpl_backend("Agg")
import matplotlib.pyplot as plt

# Machine learning
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import clone

# Utility
from tqdm import tqdm, trange

# Soothsayer
from soothsayer.utils import format_duration, create_logfile, determine_mode_for_logfiles, to_precision
from soothsayer.io import read_dataframe, read_object, write_object
from soothsayer.transmute.normalization import normalize
from soothsayer.feature_extraction import Clairvoyant

# Config
pd.set_option('max_colwidth',300)

# Functions
legend_kws = dict(fontsize=10, frameon=True, facecolor="white", edgecolor="black", loc='center left', bbox_to_anchor=(1, 0.5))

# Controller
def main(argv=None):
    parser = argparse.ArgumentParser(
    """
    soothsayer:clairvoyance {}
    """.format(__version_clairvoyance__)
    )
    # Input
    parser.add_argument('-X', '--attribute_matrix', type=str, help = 'Input: Path/to/Tab-separated-value.tsv of attribute matrix (rows=samples, cols=attributes)')
    parser.add_argument('-y', '--target_vector', type=str, help = 'Input: Path/to/Tab-separated-value.tsv of target vector (rows=samples, column=integer target)')
    parser.add_argument("-e", "--encoding", default=None, help="Input: Path/to/Tab-separated-value.tsv of encoding.  Column 1 has enocded integer and Column 2 has string representation.  No header! [Default: None]")

    # Normalization
    parser.add_argument("--normalize", type=str, default="None", help="Normalize the attribute matrix.  Valid Arguments: {'tss', 'zscore', 'quantile', None}: Warning, apply with intelligence.  For example, don't use total-sum-scaling for fold-change values. [Experimental: 2018-May-03]")

    # Hyperparameters
    parser.add_argument("-m", "--model_type", type=str, default="logistic,tree", help="Model types in ['logistic','tree'] represented by a comma-separated list [Default: 'logistic,tree']")
    parser.add_argument("--n_iter", type=int, default=10, help="Number of iterations to split data and fit models. [Default: 10]")
    parser.add_argument("--random_state", type=int, default=0, help="[Default: 0]")
    parser.add_argument("--n_jobs", type=int, default=1, help="[Default: 1]")
    parser.add_argument("--min_threshold", type=str, default="0.0,None", help="Minimum accuracy for models when considering coeficients for weights. Can be a single float or list of floats separated by `,` ranging from 0-1. If None it takes an aggregate weight.  [Default: 0.0,None]")
    parser.add_argument("--random_mode", type=int, default=0, help="Random mode 0,1, or 2.  0 uses the same random_state each iteration.  1 uses a different random_state for each parameter set.  2 uses a different random_state each time. [Default: 0]")
    parser.add_argument("--percentiles", type=str, default="0.0,0.3,0.5,0.75,0.9,0.95,0.96,0.97,0.98,0.99", help="Iterative mode takes in a comma-separated list of floats from 0-1 that dictate how many of the attributes will be used [Default: 0.0,0.3,0.5,0.75,0.9,0.95,0.96,0.97,0.98,0.99]")

    # Labels & Directories
    parser.add_argument("-n", "--name", type=str, default=time.strftime("%Y-%m-%d"), help="Name for data [Default: Date]")
    parser.add_argument("-o", "--out_dir", type=str, default = os.getcwd(), help = "Output: Path/to/existing-directory-for-output.  Do not use `~` to specify /home/[user]/ [Default: cwd]" )
    parser.add_argument("--attr_type", type=str, default="attr", help="Name of attribute type (e.g. gene, orf, otu, etc.) [Default: 'attr']")
    parser.add_argument("--class_type", type=str, default="class", help="Name of class type (e.g. phenotype, MOA, etc.) [Default: 'class']")

    # Cross Validation
    parser.add_argument("--method", type=str, default="bruteforce", help = "{'adaptive', 'bruteforce'} [Default: 'bruteforce']")
    parser.add_argument("--adaptive_range", type=str, default="0.0,0.5,1.0", help="Adaptive range in the form of a string of floats separated by commas [Default:'0.0,0.5,1.0']")
    parser.add_argument("--adaptive_steps", type=str, default="1,10", help="Adaptive stepsize in the form of a string of ints separated by commas. len(adaptive_range) - 1 [Default:'1,10']")
    parser.add_argument("--cv", default=5, type=str, help="Number of steps for percentiles when cross validating and dropping attributes.  If cv=0 then it skips cross-validation step. [Default: 5]")
    parser.add_argument("--min_bruteforce", type=int, default=150, help="Minimum number of attributes to adjust to bruteforce. If value is 0 then it will not change method [Default: 150]")
    parser.add_argument("--early_stopping", type=int, default=100, help="Stopping the algorthm if a certain number of iterations do not increase the accuracy. Use -1 if you don't want to use `early_stopping` [Default: 100]")


    # Reading in dataframes
    parser.add_argument("--compression", type=str, default="infer")
    parser.add_argument("--pickled", type=str, default="infer")

    # I/O Model
    parser.add_argument("--save_kernel", default=True, help="Save the kernel [Default: True]")
    parser.add_argument("--save_model", default=False, help="Save the model [Default: False]")
    parser.add_argument("--save_data", default=True, help="Save the model [Default: True]")


    # Broken parameters
    # parser.add_argument("--scoring", type=str, default="accuracy", help="Scoring metric [Default: 'accuracy'] NOTE ONLY USE ACCURACY FOR NOW")
    # parser.add_argument("--load_model", default="infer", help="path/to/model.pbz2.  Default: 'infer' which looks for model in the working directory")
    # parser.add_argument("--force_overwrite", default=False, help="Force overwrite data [Default: False]") # Note to fix this, all percentiles above the first change need to be recomputed



    opts = parser.parse_args(argv)

    opts.force_overwrite = True

    # opts.force_overwrite = True
    start_time = time.time()



    # Create working directory
    os.makedirs(opts.out_dir, exist_ok=True)
    path_synopsis = "{}/{}__synopsis".format(opts.out_dir,opts.name)
    os.makedirs(path_synopsis, exist_ok=True)
    log_info = create_logfile("information",
                                  "{}/{}.log".format(opts.out_dir,opts.name),
                                  mode=determine_mode_for_logfiles("{}/{}.log".format(opts.out_dir,opts.name), opts.force_overwrite)
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


    # Normalization
    if opts.normalize.lower() in ["false","none"]:
        opts.normalize = None

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
            opts.cv = pd.read_csv(opts.cv, header=None, sep="\t", dtype=str).applymap(literal_eval).values.tolist()
            cv_labels = list(map(lambda x: "cv={}".format(x+1), range(len(opts.cv))))
        # Are there 3 columns with a header with the first column being the name of cross-validation set?
        if _tmp_ncols_cv == 3:
            opts.cv = pd.read_csv(opts.cv, sep="\t", index_col=0, dtype=str).applymap(literal_eval).loc[:,["Training", "Testing"]]
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
    y = read_dataframe(opts.target_vector, compression=opts.compression, pickled=opts.pickled, func_index=str, func_columns=str, verbose=False)
    if isinstance(y, pd.Series):
        y = y.to_frame()
    y = y.iloc[:,0]

    # Encoding
    if opts.encoding:
        opts.encoding = pd.read_table(opts.encoding, sep="\t", header=None, index_col=0).iloc[:,0].to_dict()

    # Iterate through model types
    best_configuration = None
    baseline_scores_ = dict()
    summary_table = list()
    for m in opts.model_type:
        print("= == === ===== ======= ============ =====================", file=sys.stderr)
        print(m, file=sys.stderr)
        print("= == === ===== ======= ============ =====================".replace("=","."), file=sys.stderr)


        estimator = {"logistic":LogisticRegression(random_state=opts.random_state, multi_class="ovr", solver='liblinear'), "tree":DecisionTreeClassifier(random_state=opts.random_state)}[m]
        # Placeholders for dynamic data
        path_attributes_current = opts.attribute_matrix
        best_configuration_for_modeltype = None # (percentile, min_threshold, hyperparameters)
        best_accuracy_for_modeltype = 0
        best_hyperparameters_for_modeltype = None
        model = None
        best_attributes_for_modeltype = None

        log_crossvalidation = create_logfile(m,
                                                 "{}/{}__cross-validation.log".format(path_synopsis,m),
                                                 mode=determine_mode_for_logfiles("{}/{}__cross-validation.log".format(path_synopsis,m), opts.force_overwrite)
        )


        # Iterate through percentiles of weighted attributes
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
            # Load attributes
            log_info.info("\t\t{} | Percentile={} | Loading attributes datasets".format(m,i_percentile ))
            X = read_dataframe(path_attributes_current, compression=opts.compression, pickled=opts.pickled,  func_index=str, func_columns=str, verbose=False)
            if X.shape[1] < 2:
                print("Skipping current iteration {} because there is/are only {} attribute[s]".format(i_percentile, X.shape[1]), file=sys.stderr)
            else:
                # Normalize
                if opts.normalize is not None:
                    X = normalize(X, method=opts.normalize.lower(), axis=1) # COULD PROBABLY MOVE THIS UP AND COMBINE

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
                    model = read_object(path="{}/{}/percentile_{}/model.pbz2".format(opts.out_dir,m,i_percentile), compression="bz2")

                    # Set parameters for estimator and calculate baseline score
                    estimator.set_params(**model.best_hyperparameters_)
                    baseline_score = model_selection.cross_val_score(estimator, X=X.values, y=y.values, cv=opts.cv).mean()
                    baseline_scores_[(m,i_percentile)] = baseline_score

                    # Set original attribute size and baseline score
                    if i_percentile == 0.0:
                        n_attributes = X.shape[1]
                        baseline_percentile_0 = baseline_score
                    print("{} | Percentile={} | X.shape = {} | Baseline score({}) = {}".format(m, i_percentile, X.shape,path_attributes_current,to_precision(baseline_score)), file=sys.stderr)


                    # Determine if weights are different between any of the other runs
                    tol_difference = 1
                    if len(opts.min_threshold) > 1:
                        _current_order = np.asarray(model.extract_weights(min_threshold=opts.min_threshold[0], ascending=False).index)
                        for x in opts.min_threshold[1:]:
                            _query_order = np.asarray(model.extract_weights(min_threshold=x, ascending=False).index)
                            num_differences = np.sum(_current_order != _query_order)
                            if num_differences < tol_difference*2:  # Times 2 because differences come in pairs
                                opts.min_threshold.remove(x)
                                log_info.info("\t\t{} | Percentile={} | Removing min_accuracy={} from computation because {} ordering is already present from another threshold".format(m,i_percentile,x,model.attr_type))
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
                            attribute_set = scores_.loc[idx_best_accuracy_for_minthreshold, "{}_set".format(opts.attr_type)]

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
                                    best_attributes_for_modeltype = attribute_set

                            # Synthesize summary table
                            Se_query = pd.Series([m, model.best_hyperparameters_, i_percentile, x, baseline_percentile_0, baseline_score, query_accuracy,  query_sem, query_accuracy - baseline_percentile_0, opts.random_state, len(attribute_set), attribute_set],
                                           index=["model_type", "hyperparameters", "percentile",  "min_threshold", "baseline_score", "baseline_score_of_current_percentile", "accuracy", "sem", "delta", "random_state", "num_{}_included".format(opts.attr_type), "{}_set".format(opts.attr_type)])
                            summary_table.append(Se_query)
                # ==================================================================
                # Continue analysis
                # ==================================================================
                else:
                    if X.shape[1] <= opts.min_bruteforce:
                        method_crossvalidation = "bruteforce"
                        if opts.method is not "bruteforce":
                            log_info.info("\t\t{} | Percentile={} | Adjusting `method` to `bruteforce` because X.shape[1] <= {}".format(m,i_percentile,opts.min_bruteforce))
                    else:
                        method_crossvalidation = opts.method
                    # Run model
                    model = Clairvoyant(
                                model_type=m,
                                n_iter=opts.n_iter,
                                map_encoding=opts.encoding,
                                attr_type=opts.attr_type,
                                class_type=opts.class_type,
                                n_jobs=opts.n_jobs,
                                verbose=True,
                                random_state=opts.random_state,
                                random_mode=opts.random_mode,
                    )
                    log_info.info("\t\tFitting data")
                    model.fit(X, y, desc="{} | Percentile={} | Permuting samples and fitting models".format(m, i_percentile))
                    log_info.info("\t\t{} | Percentile={} | Best hyperparameters from fitting: {}".format(m,i_percentile,model.best_hyperparameters_))
                    log_info.info("\t\t{} | Percentile={} | Saving model: {}/{}/percentile_{}/model.pbz2".format(m,i_percentile,opts.out_dir,m,i_percentile))
                    if opts.save_model:
                        model.save_model(path="{}/{}/percentile_{}/model.pbz2".format(opts.out_dir,m,i_percentile), compression="bz2")


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

                    # Set original attribute size and baseline score
                    if i_percentile == 0.0:
                        n_attributes = X.shape[1]
                        baseline_percentile_0 = baseline_score
                    print("{} | Percentile={} | X.shape = {} | Baseline score({}) = {}".format(m,i_percentile,X.shape,path_attributes_current,to_precision(baseline_score)), file=sys.stderr)

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
                                log_info.info("\t\t{} | Percentile={} | Removing min_accuracy={} from computation because {} ordering is already present from another threshold".format(m,i_percentile,x,model.attr_type))
                            else:
                                _current_order = _query_order
                    # Soothsayer | Attribute Finder
                    print("================================\nsoothsayer:clairvoyance {}\n================================".format(__version_clairvoyance__), file=f_summary)
                    print("Name:", opts.name, sep="\t", file=f_summary)
                    print("X:", path_attributes_current, sep="\t", file=f_summary)
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
                    print("\n".join(str(params[["attr_type", "class_type"]]).split("\n")[:-1]), file=f_summary)
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
                    run_crossvalidation = opts.cv is not 0
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
                                        path_save = "{}/model-data.pbz2".format(path_query)
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
                                    attribute_set = scores_.loc[idx_best_accuracy_for_minthreshold, "{}_set".format(opts.attr_type)]

                                    # Update accuracies for current percentile
                                    if query_accuracy > best_accuracy_for_percentile:
                                        best_min_threshold_for_percentile = x
                                        if scores_.shape[0] >= 10:
                                            best_results_for_percentile = scores_.sort_values(["accuracy", "sem", "num_{}_included".format(opts.attr_type)], ascending=[False,True, False]).iloc[:10,:3]
                                        else:
                                            best_results_for_percentile = scores_.sort_values(["accuracy", "sem", "num_{}_included".format(opts.attr_type)], ascending=[False,True, False]).iloc[:scores_.shape[0],:3]


                                        best_accuracy_for_percentile = query_accuracy
                                        best_weights_for_percentile = w_.copy()
                                        log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Updating the model object with accuracy={}".format(m,i_percentile,x,query_accuracy))

                                        # May be slower than necessary to put this here but it's safer
                                        if opts.save_model:
                                            model.save_model(path="{}/{}/percentile_{}/model.pbz2".format(opts.out_dir,m,i_percentile), compression="bz2")

                                        # Update accuracies for current model
                                        if query_accuracy > best_accuracy_for_modeltype:
                                            best_configuration_for_modeltype = (i_percentile, x)
                                            best_hyperparameters_for_modeltype = model.best_hyperparameters_
                                            best_accuracy_for_modeltype = query_accuracy
                                            best_attributes_for_modeltype = attribute_set

                                    log_info.info("\t\t{} | Percentile={} | Minimum threshold={} | Creating plot".format(m,i_percentile,x))

                                    fig, ax = model.plot_scores(title="{} | {} | percentile={} | min_threshold={}".format(opts.name,m,i_percentile,x), ylim=(0,1), min_threshold=x, baseline_score=baseline_score)
                                    fig.savefig("{}/scores.png".format(path_query), dpi=300, format="png", bbox_inches="tight")
                                    plt.close()

                                    # Synthesize summary table
                                    Se_query = pd.Series([m, model.best_hyperparameters_, i_percentile, x, baseline_percentile_0, baseline_score, query_accuracy,  query_sem, query_accuracy - baseline_percentile_0, opts.random_state, len(attribute_set), attribute_set],
                                                   index=["model_type", "hyperparameters", "percentile",  "min_threshold", "baseline_score", "baseline_score_of_current_percentile", "accuracy", "sem", "delta", "random_state", "num_{}_included".format(opts.attr_type), "{}_set".format(opts.attr_type)])
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


                    # Close summary file
                    f_summary.close()

            # Create next dataset
            if current_iteration+1 < len(opts.percentiles):
                next_percentile = opts.percentiles[current_iteration+1]

                # Get positions
                number_of_attributes_for_next_iteration = int(round((1.0-next_percentile)*n_attributes))

                print(m, current_iteration, i_percentile, next_percentile, opts.percentiles, number_of_attributes_for_next_iteration, opts.min_threshold,  sep="\t")
                # Get attribute set
                assert best_weights_for_percentile is not None, "Try this again with --force_overwrite because certain files are being read incorrectly"
                idx_next_attribute_set = best_weights_for_percentile.nlargest(number_of_attributes_for_next_iteration).index

                # Create filepath
                os.makedirs("{}/{}/percentile_{}".format(opts.out_dir, m,next_percentile), exist_ok=True)
                path_next_attribute_matrix = "{}/{}/percentile_{}/X.subset.pbz2".format(opts.out_dir,m,next_percentile) #Current iteration and not current_itation + 1 because it starts at 1.  It is already offset
                # Check if the file exists already and overwrite if necessary
                update_next_attribute_matrix = True
                if os.path.exists(path_next_attribute_matrix):
                    if opts.force_overwrite == False:
                        update_next_attribute_matrix = False
                if update_next_attribute_matrix:
                    X.loc[:,idx_next_attribute_set].to_pickle(path_next_attribute_matrix, compression="bz2")
                # Update the attributes for next iteration
                path_attributes_current = path_next_attribute_matrix

        # Save best configuration
        print("= == === ===== ======= ============ =====================".replace("=","."), file=sys.stderr)
        print("{} | Percentile={} | Minimum threshold = {} | Baseline score = {} | Best score = {} | ∆ = {} | n_attrs = {}".format(m,best_configuration_for_modeltype[0],best_configuration_for_modeltype[1],to_precision(baseline_percentile_0),to_precision(best_accuracy_for_modeltype),to_precision(best_accuracy_for_modeltype - baseline_percentile_0),len(best_attributes_for_modeltype)), file=sys.stderr)
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
        print("", file=sys.stderr)

    # Analysis summary
    df_summarytable = pd.DataFrame(summary_table)
    df_summarytable.index.name = "Iteration"
    df_summarytable.to_csv("{}/{}__synopsis.tsv".format(path_synopsis,opts.name), sep="\t")
    log_info.info("\t\tTotal time:\t {}".format(format_duration(start_time)))
    df_summarytable.to_csv(sys.stdout, sep="\t")

# Initialize
if __name__ == "__main__":
    # Check version
    python_version = sys.version.split(" ")[0]
    condition_1 = int(python_version.split(".")[0]) == 3
    condition_2 = int(python_version.split(".")[1]) >= 6
    assert all([condition_1, condition_2]), "Python version must be >= 3.6.  You are running: {}\n{}".format(python_version,sys.executable)
    main(sys.argv)
