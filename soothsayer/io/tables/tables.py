# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, datetime, copy, pathlib
from collections import OrderedDict, defaultdict, Mapping
from io import StringIO, BytesIO
# from ast import literal_eval

# Compression & Serialization
import pickle, gzip, bz2, zipfile

# PyData
import pandas as pd

# Soothsayer
from soothsayer.utils import format_path, to_precision

# ======
# Future
# ======
# (1) Add ability for `path` arguments to be `pathlib.Path`
# (2) Add ability to process ~ in path
# ======
# Tables
# ======
# Read dataframe
def read_dataframe(path:str, sep="infer", index_col=0, header=0, compression="infer", pickled="infer", func_index=None, func_columns=None, evaluate_columns=None, engine="c", verbose=False, excel="infer", infer_series=False, sheet_name=None,  **args):

    start_time = time.time()
    path = format_path(path, str)
    dir , ext = os.path.splitext(path)
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
            df = pd.read_csv(path, sep=sep, index_col=index_col, header=header, compression=compression, engine=engine, **args)

    condition_A = any([(excel == False), (sheet_name is not None)])
    condition_B = all([(excel == True), (sheet_name is None)])

    if condition_A:
        # Map indices
        if func_index is not None:
            df.index = df.index.map(func_index)
        if func_columns is not None:
            df.columns = df.columns.map(func_columns)

        if evaluate_columns is not None:
            for field_column in evaluate_columns:
                try:
                    df[field_column] = df[field_column].map(eval)
                except ValueError:
                    if verbose:
                        print(f"Could not use `eval` on column=`{field_column}`", file=sys.stderr)
        if verbose:
            print(f"{path.split('/')[-1]} | Dimensions: {df.shape} | Time: {to_precision(time.time() - start_time)} seconds", file=sys.stderr)
    if condition_B:
        if verbose:
            print(f"{path.split('/')[-1]} | Sheets: {len(df)} | Time: {to_precision(time.time() - start_time)} seconds", file=sys.stderr)
    if infer_series:
        if df.shape[1] == 1:
            df = df.iloc[:,0]
    return df

# Write dataframe
def write_dataframe(data, path:str, sep="\t", compression="infer", pickled="infer",  excel="infer", create_directory_if_necessary=False, **args):
    start_time = time.time()
    path = format_path(path, str)
    _ , ext = os.path.splitext(path)
    dir, filename = os.path.split(path)
    if not os.path.exists(dir):
        dir = str(pathlib.Path(dir).absolute())
        os.makedirs(dir, exist_ok=True)

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

# Read results from feature selection
def read_multiple_synopsis(dir:str, from_algorithm:str="clairvoyance"):
    """
    Assumes the following directory structure:
    __________________________________
    | project_directory
    | | synopsis_dir_for_all_submodels
    | | | {name}_synopsis
    | | | | {name}__synopsis.tsv
    ----------------------------------
    """
    accepted_args = ["clairvoyance"]
    def _read_clairvoyance(dir:str):
        # Get synopsis
        _data = list()
        for file in os.scandir(dir):
            if file.is_dir():
                submodel = file.name.split("_")[1]
                for file2 in os.scandir(file):
                    if file2.name.endswith("__synopsis.tsv"):
                        _df = read_dataframe(file2)
                        _df.insert(loc=0, column="submodel", value=submodel)
                        _data.append(_df)
        df_synopsis = pd.concat(_data, axis=0).reset_index()

        # Evaluate fields
        fields_to_evaluate = ["hyperparameters"] + [*filter(lambda field:field.endswith("set"), df_synopsis.columns)]
        for field in fields_to_evaluate:
            df_synopsis[field] = df_synopsis[field].map(eval)
        return df_synopsis

    assert from_algorithm in accepted_args, f"{from_algorithm} is not in {accepted_args}"
    dir = format_path(dir, str)
    if from_algorithm == "clairvoyance":
        return _read_clairvoyance(dir)
