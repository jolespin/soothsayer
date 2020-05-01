# ==============================================================================
# Modules
# ==============================================================================
# Built-ins
import os, sys, time, pathlib, warnings, requests
from collections import *
from io import StringIO, BytesIO
from ast import literal_eval
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Compression & Serialization
import pickle, gzip, bz2, zipfile

# PyData
import pandas as pd
import numpy as np

# Biology
from Bio import SeqIO

# Soothsayer
from soothsayer.utils import format_path, infer_compression, is_path_like, is_query_class, assert_acceptable_arguments
from ..text import read_textfile, get_file_object

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
    path = format_path(path)
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            conditions = [
                line.startswith("#"),
            ]
            try:
                int(line)
                conditions.append(True)
            except ValueError:
                pass
            if not any(conditions):
                break
    df_tmp = pd.read_csv(path, sep="\t", index_col=0, header=None, skiprows=i)
    tmp = df_tmp.index.map(lambda line:[x for x in line.split(" ") if len(x) > 0])
    if ":" in tmp[0][0]:
        tmp = tmp.map(lambda x:x[1:])
    data = OrderedDict()
    try:
        for line in tmp:
            data[line[0]] = pd.Series([*map(float,line[1:])], name=line[0])
        df_dism = pd.DataFrame(data)
        df_dism.index = df_dism.columns
        return df_dism
    except ValueError:
        print("Could not accurately parse clustal distance matrix.  Returning raw data instead.", file=sys.stderr)
        return tmp

# VizBin corrdinates
def read_vizbin_coords(xy_path:str, fasta_path:str=None, xlabel="x", ylabel="y", write_coords=True):
    """
    VizBin: http://claczny.github.io/VizBin/
    xy_path: VizBin coordinates file path
    fasta_path: Input fasta file path

    Usage:
    xy_path = "/var/folders/6z/5vbtz_gmkr76ftgc3149dvtr0003c0/T/map1723096415078571924/points.txt"
    fasta_path = "./assembly.fa")
    df_xy = read_vizbin_coords(xy_path, fasta_path)
    """
    xy_path = format_path(xy_path)

    df_xy = pd.read_csv(xy_path, sep=",", header=None).astype(float)
    path_write = None
    if fasta_path is not None:
        fasta_path = format_path(fasta_path)
        df_xy.index = [*map(lambda record:str(record.id), SeqIO.parse(fasta_path, "fasta"))]
        if write_coords == True:
            path_write = f"{fasta_path}.xy"
    df_xy.columns = [xlabel,ylabel]
    if path_write is not None:
        df_xy.to_csv(path_write, sep="\t")
    return df_xy

# CheckM QA
def read_checkm_qa(path:str, format=8, verbose=True, rename_columns=True, func_orf_to_contig=lambda id_orf:"_".join(id_orf.split("_")[:-1])):
    """
    Tested for v1.0.7
    Read CheckM qa output into pd.DataFrame
    accepted_formats = {2,8}
    """
    accepted_formats = {2,8}
    assert format in accepted_formats, f"Currently only supporting formats: `{accepted_formats}`"
    path = format_path(path)

    # QA: Format = 2
    def _read_format_qa_2(path, rename_columns, verbose):
        d_rename = {
            "Bin Id":"id_bin",
            "Marker lineage":"marker_lineage",
            "# genomes":"number_of_genomes",
            "# markers":"number_of_markers",
            "# marker sets":"number_of_marker-sets",
            "Completeness":"completeness",
            "Contamination":"contamination",
            "Strain heterogeneity":"strain_heterogeneity",
            "Genome size (bp)":"genome_size[bp]",
            "# ambiguous bases":"number_of_ambiguous-bases",
            "# scaffolds":"number_of_scaffolds",
            "# contigs":"number_of_contigs",
            "N50 (scaffolds)":"N50[scaffolds]",
            "N50 (contigs)":"N50[contigs]",
            "Mean scaffold length (bp)":"Mean_scaffold_length[bp]",
            "Mean contig length (bp)":"Mean_contig_length[bp]",
            "Longest scaffold (bp)":"Longest_scaffold[bp]",
            "Longest contig (bp)":"Longest_contig[bp]",
            "GC std (scaffolds > 1kbp)":"GC_std[scaffolds>1kbp]",
            "Coding density":"coding_density",
            "Translation table":"trans_table",
            "# predicted genes":"number_of_predicted-genes",
        }
        tab_table = False
        for i, line in read_textfile(path):
            if i == 0:
                if not line.startswith("-"):
                    tab_table = True
                    break
            if i == 1:
                columns = pd.Index([*map(lambda x: x.strip(), filter(bool, line.split("  ")))])
                break
        if tab_table:
            df_table = pd.read_csv(path, sep="\t", index_col=None)
            columns = df_table.columns
        else:
            table = pd.read_csv(path, sep="\t", skiprows=[0,1,2],skipfooter=1, header=None).values.ravel()
            df_table = pd.DataFrame([*map(lambda line: [*filter(bool, line.split(" "))], table)])
            df_table.columns = columns
        if verbose:
            print("Total bins:", df_table.shape[0], file=sys.stderr)
        if rename_columns:
            df_table.columns = columns.map(lambda c: d_rename[c] if c in d_rename else c)
            return df_table.set_index("id_bin", drop=True).sort_values(["completeness", "contamination"], ascending=[False, True])
        else:
            return df_table.set_index("Bin Id", drop=True).sort_values(["Completeness", "Contamination"], ascending=[False, True])
    # QA: Format = 8
    def _read_format_qa_8(path, verbose):
        table = list()
        for i,line in read_textfile(path):
            if i > 0:
                data = line.replace("\n","").split("\t")
                # Get bin ID
                ids = data[:2]
                # Get HMM IDs
                hmm_ids = set([x.split(".")[0].split(",")[0] for x in data[2:]  ])
                # Concat positions
                if i == 0:
                    hmm_pos = data[2:]
                else:
                    hmm_pos = set([*map(lambda organized_hit: (organized_hit[0], int(organized_hit[1]), int(organized_hit[2])), map(lambda hit: tuple(hit.split(",")), data[2:]))])
                # Create table
                line_out = ids + [hmm_ids] + [hmm_pos]
                table.append(line_out)
        # Build table
        df_table = pd.DataFrame(table)
        df_table.columns = ["id_bin", "id_orf", "markers", "marker_positions"]

        # Split ORFs
        table = list()
        for i, row in df_table.iterrows():
            if "&&" in row["id_orf"]:
                for id_orf in row["id_orf"].split("&&"):
                    sub_row = row.copy()
                    sub_row["id_orf"] = id_orf
                    table.append(sub_row)
            else:
                table.append(row)
        df_table = pd.DataFrame(table)

        if func_orf_to_contig is not None:
            df_table.insert(loc=1, column="id_contig", value=df_table["id_orf"].map(func_orf_to_contig).tolist())
        if verbose:
            print("Total markers:",len(set().union(*df_table["markers"])), sep="\t", file=sys.stderr)
        return df_table

    # Controller
    if format == 2:
        return _read_format_qa_2(path, rename_columns, verbose)
    if format == 8:
        return _read_format_qa_8(path, verbose)

# Read Fasta File
def read_fasta(path:str, description:bool=True, case="upper", func_header=None, into=pd.Series, compression="infer", name=None, verbose=True):
    """
    Reads in a single fasta file or a directory of fasta files into a dictionary.
    """
    # Get path
    path = format_path(path)

    # Assign pathname as name if there isn't one
    if name is None:
        name = path

    # Open file object
    f = get_file_object(path, mode="read", compression=compression, safe_mode=False, verbose=False)

    # Read in fasta
    d_id_seq = OrderedDict()

    if verbose:
        seq_records = tqdm(SeqIO.FastaIO.SimpleFastaParser(f), f"Reading sequence file: {path}")
    else:
        seq_records = SeqIO.FastaIO.SimpleFastaParser(f)

    # Verbose but faster
    if case == "lower":
        for header, seq in seq_records:
            seq = seq.lower()
            if not description:
                header = header.split(" ")[0]
            d_id_seq[header] = seq
    if case == "upper":
        for header, seq in seq_records:
            seq = seq.upper()
            if not description:
                header = header.split(" ")[0]
            d_id_seq[header] = seq
    if case is None:
        for header, seq in seq_records:
            if not description:
                header = header.split(" ")[0]
            d_id_seq[header] = seq

    # Close File
    f.close()

    # Transform header
    if func_header is not None:
        d_id_seq = OrderedDict( [(func_header(id),seq) for id, seq in d_id_seq.items()])
    sequences = into(d_id_seq)
    if hasattr(sequences, "name"):
        sequences.name = name
    return sequences


# Writing sequence files
def write_fasta(sequences, path:str, compression="infer"):
    """
    Sequence stats:
    count    29999.000000
    mean       310.621754
    std       1339.422833
    min         56.000000
    25%         75.000000
    50%        111.000000
    75%        219.000000
    max      54446.000000

    Benchmarks:
    No compression: 47.2 ms ± 616 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    Gzip: 9.85 s ± 261 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    Bzip2: 864 ms ± 16.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """
    # path = format_path(path)
    # if compression == "infer":
    #     compression = infer_compression(path)
    if is_query_class(path, ["stdout", "stderr", "streamwrapper"]):
        path.writelines(f">{id}\n{seq}\n" for id, seq in sequences.items())
    else:
        with get_file_object(path, mode="write", compression=compression, safe_mode=False, verbose=False) as f:
            f.writelines(f">{id}\n{seq}\n" for id, seq in sequences.items())

# Read blast output
def read_blast(path:str, length_query=None, length_subject=None, sort_by="bitscore"):
    """
    if 12: ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
    if 15 assumes: -outfmt '6 std qlen slen stitle': ["std", "qlen", "slen", "stitle"]
    ####################################################
    Column    NCBI name    Description
    1    qseqid    Query Seq-id (ID of your sequence)
    2    sseqid    Subject Seq-id (ID of the database hit)
    3    pident    Percentage of identical matches
    4    length    Alignment length
    5    mismatch    Number of mismatches
    6    gapopen    Number of gap openings
    7    qstart    Start of alignment in query
    8    qend    End of alignment in query
    9    sstart    Start of alignment in subject (database hit)
    10    send    End of alignment in subject (database hit)
    11    evalue    Expectation value (E-value)
    12    bitscore    Bit score
    13    sallseqid    All subject Seq-id(s), separated by a ';'
    14    score    Raw score
    15    nident    Number of identical matches
    16    positive    Number of positive-scoring matches
    17    gaps    Total number of gaps
    18    ppos    Percentage of positive-scoring matches
    19    qframe    Query frame
    20    sframe    Subject frame
    21    qseq    Aligned part of query sequence
    22    sseq    Aligned part of subject sequence
    23    qlen    Query sequence length
    24    slen    Subject sequence length
    25    salltitles    All subject title(s), separated by a '<>'

    Example inputs:
        * blat -prot Yeast/Saccharomyces_cerevisiae.R64-1-1.pep.all.processed.fa  Phaeodactylum_tricornutum.ASM15095v2.pep.all.processed.fa -out=blast8 yeast-pt.blast8
        * diamond blastp -d ../../../../reference_data/references/gracilibacteria/reference_proteins.nmnd -q ./prodigal_output/orfs.faa -f 6 -o ./diamond_output/output.blast6

    """
    path = format_path(path)
    idx_default_fields = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
    # ["query_id", "subject_id", "identity", "alignment_length", "mismatches", "gap_openings", "query_start", "query_end", "subject_start", "subject_end", "e-value", "bit_score"]
    df_blast = pd.read_csv(path, header=None, sep="\t")
    if df_blast.shape[1] == 12:
        df_blast.columns = idx_default_fields
    if df_blast.shape[1] == 15:
        df_blast.columns = idx_default_fields + ["qlen", "slen", "stitle"]
    # Length of query
    if length_query is not None:
        if is_path_like(length_query):
            length_query = read_fasta(length_query, description=False, verbose=False)
        if isinstance(length_query[0], str):
            length_query = length_query.map(len)
        df_blast["qlength"] = df_blast["qseqid"].map(lambda id: length_query[id])
        df_blast["qratio"] = (df_blast["qend"] - df_blast["qstart"])/df_blast["qlength"]
    # Length of subject
    if length_subject is not None:
        if is_path_like(length_subject):
            length_subject = read_fasta(length_subject, description=False, verbose=False)
        if isinstance(length_subject[0], str):
            length_subject = length_subject.map(len)
        df_blast["slength"] = df_blast["sseqid"].map(lambda id: length_subject[id])
        df_blast["sratio"] = (df_blast["send"] - df_blast["sstart"])/df_blast["slength"]
    if sort_by is not None:
        df_blast = df_blast.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

    return df_blast
# Helper function for reading gtf and gff3
def _read_gtf_gff_base(path, compression, record_type, verbose):
    # Read the gff3 file
    with get_file_object(path, mode="read", compression=compression, safe_mode=False, verbose=False) as f:
        if verbose:
            iterable_lines = tqdm(f.readlines(), "Processing lines")
        else:
            iterable_lines = f.readlines()
        data= list()
        if record_type is None:
            for line in iterable_lines:
                if not line.startswith("#"):
                    line = line.strip("\n")
                    if bool(line):
                        base_fields = line.split("\t")
                        data.append(base_fields)
        else:
            for line in iterable_lines:
                if not line.startswith("#"):
                    if f"{record_type}_id" in line:
                        line = line.strip("\n")
                        base_fields = line.split("\t")
                        data.append(base_fields)
    # Generate table
    df_base = pd.DataFrame(data)
    df_base.columns = ["seq_record", "source", "seq_type", "pos_start", "pos_end", ".1", "sense", ".2", "data_fields"]
    return df_base

def read_gff3(path:str, compression="infer", record_type=None, verbose = True, reset_index=False, name=True):
    def f(x):
        fields = x.split(";")
        data = dict()
        for item in fields:
            k, v = item.split("=")
            data[k] = v
        return data
    path = format_path(path)
    if verbose:
        print("Reading gff3 file:",path,sep="\t", file=sys.stderr)
    accepted_recordtypes = {"exon", "gene", "transcript", "protein", None}
    assert record_type in accepted_recordtypes, "Unrecognized record_type.  Please choose from the following: {}".format(accepted_recordtypes)

    # Read the gff3 file
    df_base = _read_gtf_gff_base(path, compression, record_type, verbose)

    try:
        df_fields = pd.DataFrame(df_base["data_fields"].map(f).to_dict()).T
        df_gff3 = pd.concat([df_base[["seq_record", "source", "seq_type", "pos_start", "pos_end", ".1", "sense", ".2"]], df_fields], axis=1)
         # Contains identifier
        if "ID" in df_gff3.columns:
            df_gff3["seq_id"] = df_gff3["ID"].map(lambda x: "-".join(x.split("-")[1:]) if not pd.isnull(x) else x)
        if reset_index:
            df_gff3 = df_gff3.reset_index(drop=True)
    except IndexError:
        warnings.warn("Could not successfully parse file: {}".format(path))
        df_gff3 = df_base

    # Path index
    if name is not None:
        if name == True:
            name = path
        df_gff3.index.name = name

    return df_gff3

# GTF
def read_gtf(path:str, compression="infer", record_type=None, verbose = True, reset_index=False, name=True):
    path = format_path(path)
    if verbose:
        print(f"Reading gtf file:",path,sep="\t", file=sys.stderr)
    accepted_recordtypes = {"exon", "gene", "transcript", "protein", None}
    assert record_type in accepted_recordtypes, f"Unrecognized record_type.  Please choose from the following: {accepted_recordtypes}"

    # Read the gtf file
    df_base = _read_gtf_gff_base(path, compression, record_type, verbose)

    # Splitting fields
    iterable_fields =  df_base.iloc[:,-1].iteritems()

    # Parse fields
    dataframe_build = dict()
    for i, gtf_data in iterable_fields:
        gtf_data = gtf_data.replace("''","").replace('"',"")
        fields = filter(bool, map(lambda field:field.strip(), gtf_data.split(";")))
        fields = map(lambda field: field.split(" "), fields)
        dataframe_build[i] = dict(fields)
    df_fields = pd.DataFrame(dataframe_build).T
    df_gtf = pd.concat([df_base.iloc[:,:7], df_fields], axis=1)

    # Reset index
    if reset_index:
        df_gtf = df_gtf.reset_index(drop=True)

    # Path index
    if name is not None:
        if name == True:
            name = path
        df_gtf.index.name = name
    return df_gtf

# NCBI XML
def read_ncbi_xml(path:str, index_name=None):
    # Format path
    path = format_path(path)
    # Parse the XML tree
    tree = ET.parse(path)
    root = tree.getroot()
    # Extract the attributes
    data = defaultdict(dict)
    for record in tqdm(root.getchildren(), "Reading NCBI XML: {}".format(path)):
        id_record = record.attrib["accession"]
        for attribute in record.findall("Attributes/*"):
            data[id_record][attribute.attrib["attribute_name"]] = attribute.text
    # Create pd.DataFrame
    df = pd.DataFrame(data).T
    df.index.name = index_name
    return df

# Read EMBL-EBI sample metadata
def read_ebi_sample_metadata(query, base_url="https://www.ebi.ac.uk/metagenomics/api/v1/samples/{}/metadata", mode="infer"):
    """
    query:
        id_sample: "ERS488919"
        url: "https://www.ebi.ac.uk/metagenomics/api/v1/samples/ERS488919/metadata
        file: "/path/to/ERS488919.json"

    JSON file can be retrieved via:
        curl -X GET "https://www.ebi.ac.uk/metagenomics/api/v1/samples/ERS488919/metadata" -H "accept: application/json" > ERS488919.json
    """
    # Acceptable arguments for mode
    assert_acceptable_arguments(mode, {"infer", "cloud", "local"})

    # Functions
    def _get_url(query, base_url):
        # Is a URL provided?
        if any(map(lambda x: x in query, ["http", "www"])):
            url = query
        # Is a sample identifier provided
        else:
            url = base_url.format(query)
        return url

    # Infer mode
    url = None
    response = None
    if mode == "infer":
        try:
            url = _get_url(query, base_url)
            response = requests.get(url)
            mode = "cloud"
        except requests.exceptions.MissingSchema:
            mode = "local"

    # Cloud
    if mode == "cloud":
        if response is None:
            # Get cloud data
            url = _get_url(query, base_url)
            response = requests.get(url)
        data = response.json()["data"]
    if mode == "local":
        with open(query, "r") as f:
            data = json.loads(f.read())["data"]
    # Determine sample identifier
    id_sample = data[0]["relationships"]["sample"]["data"]["id"]

    return pd.Series(
            data=dict(map(lambda item: (item["attributes"]["key"], item["attributes"]["value"]), data)),
            name=id_sample,
    )
