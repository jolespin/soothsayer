# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, datetime, pathlib
from collections import OrderedDict, defaultdict
from io import StringIO, BytesIO, TextIOWrapper
from ast import literal_eval

# Compression & Serialization
import pickle, gzip, bz2, zipfile

# PyData
import pandas as pd

# Soothsayer
from soothsayer.utils import format_path, infer_compression, is_file_like

# Create file object
def get_file_object(path:str, mode="infer", compression="infer", safe_mode="infer", verbose=True):
    """
    # Should this be in io?
    with get_file_object("./test.txt.zip", mode="infer", verbose=False) as f_read:
    with get_file_object("./test_write.txt.bz2", mode="write", verbose=False) as f_write:
        for line in f_read.readlines():
            line = str(line.strip())
            print(line, file=f_write)
    """
    # Init
    f = None
    file_will_be_written = False

    # Paths
    path = format_path(path)
    path_exists = os.path.exists(path)

    if verbose:
        header = " ".join(["Processing file:", path])
        print("="*len(header), header, "="*len(header), sep="\n", file=sys.stderr)
    if compression == "infer":
        compression = infer_compression(path)
        if verbose:
            print("Inferring compression:", compression, file=sys.stderr)

    # Inferring mode
    if mode == "infer": # Create new function for this? infer_filemode?
        if path_exists:
            mode = "read"
        else:
            mode = "write"
    assert mode != "infer", "The mode should be inferred by this point.  Please specify mode manually."
    assert compression != "infer", "The compression should be inferred by this point.  Please specify compression manually."

    # Generic read write
    if mode in ["write", "read"]:
        if mode == "write":
            mode = "w"
        if mode == "read":
            mode = "r"
        if compression in ["gzip", "bz2"]:
            mode = mode + "b"
        if verbose:
            print("Inferring mode:", mode, file=sys.stderr)

    # Will a file be written?
    if "w" in mode:
        file_will_be_written = True

    # Ensure zip is not being written
    if file_will_be_written:
        assert compression != "zip", "Currently cannot handle writing zip files.  Please use gzip, bz2, or None."
        # Future do this:
        # https://docs.python.org/3/library/zipfile.html#zipfile.ZipFile.open

    # Safe mode
    if safe_mode == "infer":
        if file_will_be_written:
            safe_mode = True
        else:
            safe_mode = False
    assert safe_mode in {True,False}, "Please choose either True or False for `safe_mode`"
    if safe_mode:
        if all([file_will_be_written, path_exists]):
            raise Exception(f"Safe Mode: Please explicitly provide a writeable mode ['w', 'wb', or 'write'] because `{path}` already exists and will be rewritten.")

    # GZIP compression
    if compression == "gzip":
        f = gzip.open(path, mode)
    # BZ2 compression
    if compression == "bz2":
        f = bz2.open(path, mode)
    if compression == "zip":
        filename, ext = os.path.splitext(os.path.split(path)[-1])
        f = zipfile.ZipFile(path,mode).open(filename)
    # No compression
    if f is None:
        return open(path, mode)

    # Reading and writing compressed files
    else:
        return TextIOWrapper(f, encoding="utf-8")


# Text reading wrapper
def read_textfile(path, enum=True, generator=True, mode="read", compression="infer", into=pd.Series):
    """
    2018-May-29
    Edits: 2018-December-27: Added `get_file_object` dependency
    """
    assert mode not in ["w", "wb", "a"], "`mode` should not be in {w, wb, a} because it will overwrite"
    assert compression  in ["infer", "gzip", "bz2", "zip", None], "Valid `compression` types are 'infer', 'gzip', 'bz2', 'zip'"
    # Format file path
    path = format_path(path, str)
    # Get file object
    handle = get_file_object(path=path, mode=mode, compression=compression, safe_mode=False, verbose=False)
    # Subfunctions
    def run_return_list(handle):
        data = handle.read().split("\n")
        handle.close()
        if into == pd.core.series.Series:
            return pd.Series(data, name=path)
        else:
            if enum:
                return into([*enumerate(data)])
            if not enum:
                return into(data)

    def run_return_iterable(handle):
        if enum:
            for i,line in enumerate(handle.readlines()):
                yield i, line.strip()
        if not enum:
            for line in handle.readlines():
                yield line.strip()
        handle.close()

    # Controller
    if generator:
        return run_return_iterable(handle=handle)
    if not generator:
        return run_return_list(handle=handle)
