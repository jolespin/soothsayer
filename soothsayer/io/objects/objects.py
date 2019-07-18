# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, copy, pathlib, importlib
from collections import *

# Compression & Serialization
import pickle, gzip, bz2, zipfile

# Soothsayer
from soothsayer.utils import format_path, infer_compression

# ======
# Future
# ======
# (1) Add ability for `path` arguments to be `pathlib.Path`
# (2) Add ability to process ~ in path

# =============
# Serialization
# =============
# Writing serial object
def write_object(obj, path:str, compression="infer", serialization_module=pickle, protocol=pickle.HIGHEST_PROTOCOL, *args):
    """
    Extensions:
    pickle ==> .pkl
    dill ==> .dill
    gzipped-pickle ==> .pgz
    bzip2-pickle ==> .pbz2
    """
    assert obj is not None, "Warning: `obj` is NoneType"
    path = format_path(path, str)

    # Use infer_compression here
    if compression == "infer":
        _ , ext = os.path.splitext(path)
        if (ext == ".pkl") or (ext == ".dill"):
            compression = None
        if ext in {".pgz", ".gz"}:
            compression = "gzip"
        if ext in {".pbz2", ".bz2"}:
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

# Reading serial object
def read_object(path:str, compression="infer", serialization_module=pickle):
    path = format_path(path, str)

    if compression == "infer":
        _ , ext = os.path.splitext(path)
        if (ext == ".pkl") or (ext == ".dill"):
            compression = None
        if ext in {".pgz", ".gz"}:
            compression = "gzip"
        if ext in {".pbz2", ".bz2"}:
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

# Importing a functions from a module
def read_script_as_module(name_module, path):
    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    path = format_path(path, str)
    spec = importlib.util.spec_from_file_location(name_module, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
