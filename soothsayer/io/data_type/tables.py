# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, datetime, copy, uuid, pathlib
from collections import *
from io import StringIO, BytesIO
from ast import literal_eval

# Compression & Serialization
import pickle, gzip, bz2, zipfile

# PyData
import pandas as pd

# ======
# Future
# ======
# (1) Add ability for `path` arguments to be `pathlib.Path`
# (2) Add ability to process ~ in path
# ======
# Tables
# ======
# Read dataframe
def read_dataframe(path:str, sep="infer", index_col=0, header=0, compression="infer", pickled="infer", func_index=None, func_columns=None, engine="c", verbose=False, excel="infer", sheet_name=0, **args):
    start_time = time.time()
    if type(path) == pathlib.PosixPath:
        path = str(path.absolute())
    _ , ext = os.path.splitext(path)
    ext = ext.lower()

    if excel == "infer":
        if ext in {".xlsx", ".xls"}:
            excel = True
        else:
            excel = False

    if excel:
        if "sheetname" in args:
            sheet_name = args.pop("sheetname")
            print("DEPRECATED: Use `sheet_name` instead of `sheetname`", file=sys.stderr)
        df = pd.read_excel(path, sheet_name=sheet_name, index_col=index_col, header=header, **args)

    else:
        # Seperator
        if any(list(map(lambda x:ext.endswith(x),[".csv", "csv.gz", "csv.zip"]))):
            sep = ","
        else:
            sep = "\t"

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
                if ext == ".gz":
                    compression = "gzip"
                if ext == ".bz2":
                    compression = "bz2"
                if ext == ".zip":
                    compression = "zip"


        if pickled:
            df = pd.read_pickle(path, compression=compression)
        else:
            df = pd.read_table(path, sep=sep, index_col=index_col, header=header, compression=compression, engine=engine, **args)

    condition_A = any([(excel == False), (sheet_name is not None)])
    condition_B = all([(excel == True), (sheet_name is None)])

    if condition_A:
        # Map indices
        if func_index is not None:
            df.index = df.index.map(func_index)
        if func_columns is not None:
            df.columns = df.columns.map(func_columns)
        if verbose:
            print(f"{path.split('/')[-1]} | Dimensions: {df.shape} | Time: {to_precision(time.time() - start_time)} seconds", file=sys.stderr)
    if condition_B:
        if verbose:
            print(f"{path.split('/')[-1]} | Sheets: {len(df)} | Time: {to_precision(time.time() - start_time)} seconds", file=sys.stderr)
    return df

# Write dataframe
def write_dataframe(data, path:str, sep="\t", compression="infer", pickled="infer",  excel="infer", **args):
    start_time = time.time()
    if type(path) == pathlib.PosixPath:
        path = str(path.absolute())
    _ , ext = os.path.splitext(path)
    # Excel
    if excel == "infer":
        if ext in {".xlsx", ".xls"}:
            excel = True
        else:
            excel = False

    if excel:
        if not isinstance(data, Mapping):
            data = {"Sheet1":data}
        writer = pd.ExcelWriter(path)
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=str(sheet_name))
        writer.save()
    else:
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
                compression = None
                if ext == ".gz":
                    compression = "gzip"
                if ext == ".bz2":
                    compression = "bz2"

        if pickled:
            data.to_pickle(path, compression=compression, **args)
        else:
            data.to_csv(path, sep=sep, compression=compression, **args)
