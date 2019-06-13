from .algorithms.clairvoyant import *
from ..io import read_textfile

_algorithms = ["Clairvoyant"]
_utils = ["get_trace_from_algorithm_log"]
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
