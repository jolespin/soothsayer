from .algorithms.clairvoyant import *
from ..io import read_textfile, read_dataframe
from ..utils import is_path_like, assert_acceptable_arguments
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
import pandas as pd

_algorithms = ["Clairvoyant"]
_utils = ["get_trace_from_algorithm_log", "get_best_model_from_algorithm"]
__all__ = _algorithms + _utils
__all__ = sorted(__all__)


def get_trace_from_algorithm_log(path:str, algorithm:str="Clairvoyance"):
    """
    Get the progress states from a feature extraction algorithm
    """
    accepted_algorithms = {"clairvoyance"}
    algorithm = algorithm.lower()
    assert algorithm in accepted_algorithms, f"`algorithm` must be one of the following: {accepted_algorithms}"
    if algorithm == "clairvoyance":
        baseline = None
        trace = list()
        for i,line in read_textfile(path, enum=True, generator=True, mode="r"):
            if "Accuracy=" in line:
                if baseline is None:
                    baseline = float(line.split("Baseline=")[1].split("\t")[0])
                accuracy = float(line.split("Accuracy=")[1].split("\t")[0])
                trace.append(accuracy)
    return {"baseline":baseline, "trace":trace}



# Get best model from feature selection
def get_best_model_from_algorithm(synopsis:pd.DataFrame, model_type="infer", prefer="logistic", less_features_is_better=True, copy_synopsis=True, into=pd.Series, algorithm="clairvoyance"):
    multiple_hits_message = "Multiple instances with best accuracy, lowest sem, and number of features.  Choosing first option."
    assert algorithm == "clairvoyance", "Currently, `clairvoyance` is the only supported algorithm"
    if is_path_like(synopsis):
        df_synopsis = read_dataframe(synopsis, evaluate_columns=["hyperparameters"] )
        for id_feature_field in filter(lambda x:x.endswith("_set"), df_synopsis.columns):
            df_synopsis[id_feature_field] = df_synopsis[id_feature_field].map(eval)
            try:
                df_synopsis[id_feature_field] = df_synopsis[id_feature_field].map(lambda x:list(map(eval, x)))
            except NameError:
                pass
    else:
        df_synopsis = synopsis
        id_feature_field = list(filter(lambda x:x.endswith("_set"), df_synopsis.columns))[0]

    # Sort the synopsis
    feature_type = "_".join(id_feature_field.split("_")[:-1])
    df_synopsis = df_synopsis.sort_values(["accuracy", "sem", "num_{}_included".format(feature_type)], ascending=[False, True, less_features_is_better])

    # Infer best model_type
    if model_type == "infer":
        max_accuracy = df_synopsis["accuracy"].max()
        idx_with_max_accuracy = df_synopsis["accuracy"][lambda x: x == max_accuracy].index
        if len(idx_with_max_accuracy) == 1:
            model_type = df_synopsis.loc[idx_with_max_accuracy[0],"model_type"]
        else:
            model_types_with_best_accuracy = df_synopsis.loc[idx_with_max_accuracy,"model_type"].unique()
            if len(model_types_with_best_accuracy) == 2:
                model_type = prefer
            else:
                warnings.warn(multiple_hits_message)
                model_type = model_types_with_best_accuracy[0]

    # Subset model of interest
    df_synopsis = df_synopsis.query("model_type == '{}'".format(model_type))
    max_accuracy = df_synopsis["accuracy"].max()
    idx_with_max_accuracy = df_synopsis["accuracy"][lambda x: x == max_accuracy].index
    if len(idx_with_max_accuracy) > 1:
        warnings.warn(multiple_hits_message)
    # Best model
    best_model_info = df_synopsis.loc[idx_with_max_accuracy[0]]
    ModelClass = {
        "logistic":LogisticRegression,
        "tree":DecisionTreeClassifier}[model_type]
    output_info = {
        "clf":ModelClass(**best_model_info["hyperparameters"],
                           random_state=best_model_info["random_state"]),
        "hyperparameters":best_model_info["hyperparameters"],
        "features":best_model_info[id_feature_field],
        **best_model_info[["accuracy", "sem", "delta"]],
    }
    if copy_synopsis:
        output_info["synopsis"] = df_synopsis
    return into(output_info)
