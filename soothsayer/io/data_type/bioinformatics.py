# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, pathlib
from collections import *
from io import StringIO, BytesIO
from ast import literal_eval

# Compression & Serialization
import pickle, gzip, bz2, zipfile

# PyData
import pandas as pd

# Biology
from Bio import SeqIO

# ======
# Future
# ======
# (1) Add ability for `path` arguments to be `pathlib.Path`
# (2) Add ability to process ~ in path

# ==============
# Bioinformatics
# ==============
# Clustalo Distance Matrix
def read_clustalo_distmat(path:str):
    df_tmp = pd.read_table(path, sep="\t", index_col=0, header=None, skiprows=1)
    tmp = df_tmp.index.map(lambda line:[x for x in line.split(" ") if len(x) > 0])

    data = OrderedDict()

    for line in tqdm(tmp):
        data[line[0]] = pd.Series([*map(float,line[1:])], name=line[0])
    df_dism = pd.DataFrame(data)
    df_dism.index = df_dism.columns
    return df_dism

# Read Fasta File
def read_fasta(path, description=True, case="upper", func_header=None, into=pd.Series, compression="infer"):
    """
    Reads in a single fasta file or a directory of fasta files into a dictionary.
    """
    # Compression
    if compression == "infer":
        if path.endswith(".gz"):
            compression = "gzip"
        elif path.endswith(".zip"):
            compression = "zip"
        else:
            compression = None

    # Open file object
    if compression == "gzip":
        handle = StringIO(gzip.GzipFile(path).read().decode("utf-8"))
    if compression == "zip":
        handle = StringIO(zipfile.ZipFile(path,"r").read(path.split("/")[-1].split(".zip")[0]).decode("utf-8"))
    if compression == None:
        handle = open(path, "r")

    # Read in fasta
    d_id_seq = OrderedDict()

    # Verbose but faster
    if case == "lower":
        for record in SeqIO.FastaIO.SimpleFastaParser(handle):
            seq = record[1].lower()
            if description == True:
                header = record[0]
            if description == False:
                header = record[0].split(" ")[0]
            d_id_seq[header] = seq
    if case == "upper":
        for record in SeqIO.FastaIO.SimpleFastaParser(handle):
            seq = record[1].upper()
            if description == True:
                header = record[0]
            if description == False:
                header = record[0].split(" ")[0]
            d_id_seq[header] = seq
    if case is None:
        for record in SeqIO.FastaIO.SimpleFastaParser(handle):
            seq = record[1]
            if description == True:
                header = record[0]
            if description == False:
                header = record[0].split(" ")[0]
            d_id_seq[header] = seq

    # Close File
    handle.close()

    # Transform header
    if func_header is not None:
        d_id_seq = OrderedDict( [(func_header(id),seq) for id, seq in d_id_seq.items()])
    return into(d_id_seq)


# Writing fasta files
def write_fasta(sequences, path:str):
    if type(sequences) == pd.Series:
        sequences = sequences.to_dict(OrderedDict)
    with open(path, "w") as f:
        for id, seq in sequences.items():
            print(">%s\n%s"%(id,seq), file=f)
