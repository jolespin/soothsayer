import os, sys, datetime, site, pathlib, operator
from collections import OrderedDict, defaultdict, namedtuple
import pandas as pd
import numpy as np
import ete3
import skbio
# import mmh3
from scipy import stats
from scipy.spatial.distance import squareform
from tqdm import tqdm
from ..symmetry import Symmetric
from ..io import write_object, read_fasta, read_textfile
from ..utils import pd_dataframe_matmul, format_path, pd_series_collapse, is_dict_like, is_query_class, is_symmetrical, fragment, assert_acceptable_arguments, hash_kmer
from ..transmute.normalization import normalize
from Bio.SeqIO.FastaIO import SimpleFastaParser

__all__ = ["PhylogenomicFunctionalComponents",  "phylogenomically_binned_functional_potential", "mcr_from_directory", "get_taxonomy_lineage_from_identifier", "get_taxonomy_identifier_from_name", "infer_taxonomy", "alpha_diversity", "beta_diversity", "otu_to_level", "reverse_complement","hash_kmer", "count_kmers", "prevalence", "get_intergenic_sequences"]
__all__ = sorted(__all__)

# Extract intergenic sequences
def get_intergenic_sequences(path_annotations:str, path_sequences:str, seq_type="gene", include_peripheral_info=True, peripheral_tag="locus_tag", into=pd.Series):
    """
    Currently tested on GFF3 but it should work for GTF too
    """
    # Get sequences
    sequences = read_fasta(path_sequences, description=False, into=dict, verbose=False)

    # Get gene positions
    d_contig_gene_info = defaultdict(list)

    for i, line in read_textfile(path_annotations):
        if bool(line):
            fields = line.split("\t")
            id_contig = fields[0]
            if fields[2] == "gene":
                pos_start = int(fields[3])-1
                pos_end = int(fields[4])
                data = dict(map(lambda item:item.split("="), filter(bool, fields[-1].split(";"))))
                d_contig_gene_info[id_contig].append((pos_start, pos_end, fields[6], data[peripheral_tag]))

    # Get intergenic sequences
    d_id_seqs= OrderedDict()
    for id_contig, info in d_contig_gene_info.items():
        seq_reference = sequences[id_contig]
        for i in range(0, len(info)-1):
            pos_start = info[i][1]
            pos_end = info[i+1][0]
            id = "{}|intergenic_{}-{}".format(id_contig,pos_start+1, pos_end)
            if include_peripheral_info:
                id = "{}|({}){}|({}){}".format(id,info[i][2], info[i][3], info[i+1][2], info[i+1][3])
            seq_intergenic = seq_reference[pos_start:pos_end]
            if len(seq_intergenic) > 0:
                d_id_seqs[id] = seq_intergenic
    return into(d_id_seqs)


# Phylogeneomically binned functional potential
def phylogenomically_binned_functional_potential(X:pd.DataFrame,MCR:pd.DataFrame):
    """
    X: DataFrame of scaled compositional data that preserves proportionality (e.g. TPM or relative-abundance)
    MCR: DataFrame

    X.shape (n,m)
    MCR.shape (m,p)
    Output: (n,p)
    """
    assert X.shape[1] == MCR.shape[0], "Shapes are not compatible"
    return pd_dataframe_matmul(X,MCR)

# Parse the files from KEGG Maple
def mcr_from_directory(directory:str, name=None, end_tag="module.xls", mcr_field="MCR % (ITR)", index_name_bins = "id_bin", index_name_module = "id_module", store_raw_data=False):
    """
    directory:
    Path to kegg maple results which should have the following structure:
    | directory
    | | bin_1
    | | | xyz1.module.xls
    | | bin_2
    | | | xyz2.module.xls

    Returns:
    namedtuple with the following attributes:
        .ratios = MCR ratio
        .metadata = module metadata
        .name = user-provided name of experiment
        .raw_data = {id_bin:raw_kegg_maple_}

    Assumes subdirectories are bin_ids and
    each directory has a file that has the
    following format: 8206709269.module.xls

    https://www.genome.jp/maple-bin/mapleSubmit.cgi?aType=sDirect
    """
    MCR = namedtuple("MCR", ["ratios", "metadata", "name", "raw_data"])
    # Format the path
    directory = format_path(directory)
    # Convert the path to a Path object
    directory = pathlib.Path(directory)

    # Check directory
    assert directory.exists(), f"{str(directory.absolute())} does not exist"
    assert directory.is_dir(), f"{str(directory.absolute())} is not a directory"

    # Iterate through each KEGG Maple result dataset
    d_bin_mcr = OrderedDict()
    module_metadata = list()
    d_bin_raw = dict()
    for subdir in tqdm(directory.iterdir(), f"Reading subdirectories in {str(directory)}"):
        sheets = list()
        if subdir.is_dir():
            id_bin = subdir.name
            for file in subdir.iterdir():
                if file.name.endswith(end_tag):
                    for sheetname in ["module_pathway", "module_complex", "module_function", "module_signature"]:
                        df = pd.read_excel(file, index_col=0, sheetname=sheetname).set_index("ID", drop=False)
                        sheets.append(df)
            df_bin = pd.concat(sheets, axis=0)
            module_metadata.append(df_bin.loc[:,["Type", "Small category", "Name(abbreviation)"]])
            d_bin_mcr[id_bin] = df_bin[mcr_field]
            # Store raw data
            if store_raw_data:
                d_bin_raw[id_bin] = df_bin.drop("ID", axis=1)
    # Module Completion Ratios
    df_mcr = pd.DataFrame(d_bin_mcr).T.fillna(0.0)
    df_mcr.index.name = index_name_bins
    df_mcr.columns.name = index_name_module

    # Metadata
    df_metadata = pd.concat(module_metadata, axis=0).drop_duplicates() # May not be the most efficient but w/e

    return MCR(ratios=df_mcr, metadata=df_metadata, name=name, raw_data=d_bin_raw)


# Phylogenomic Functional Components
class PhylogenomicFunctionalComponents(object):
    __doc__ = """
    PhylogenomicFunctionalComponents
    ================================
    This includes methods to create Phylogenomic Functional Components.
    For `components_taxonomy` check out the databases available with Soothsayer:
    %s/soothsayer/db/

    Formatting should resemble the following:

    components_taxonomy.head()
    NODE_100000_length_1286_cov_38.26_32_868_-            UNBINNED__Streptococcus mutans
    NODE_100000_length_1286_cov_38.26_996_1285_-           UNBINNED__Streptococcus mitis
    NODE_100001_length_1286_cov_31.4249_1_152_+               UNBINNED__Neisseria mucosa
    NODE_100001_length_1286_cov_31.4249_257_925_+             UNBINNED__Neisseria mucosa
    NODE_100004_length_1286_cov_2.62551_1_686_-      UNBINNED__Leptotrichia goodfellowii
    Name: taxonomy, dtype: object

    components_functional.head() # Note this one will most likely have multiple functions for each ORF which is why it's a set and not a string
    NODE_100009_length_1286_cov_12.6458_1_870_+         {M00432__Leucine biosynthesis, 2-oxoisovalerate => 2-oxoisocaproate}
    NODE_395196_length_426_cov_39.5364_1_426_+       {M00051__Uridine monophosphate biosynthesis, glutamine (+ PRPP) => UMP}
    NODE_395187_length_426_cov_2.6442_1_426_-                         {M00120__Coenzyme A biosynthesis, pantothenate => CoA}
    NODE_39517_length_2534_cov_7.92457_605_1108_+                                             {M00063__CMP-KDO biosynthesis}
    NODE_39517_length_2534_cov_7.92457_1_572_+        {M00018__Threonine biosynthesis, aspartate => homoserine => threonine}
    """%site.getsitepackages()[0]
    def __init__(self,
                 # Input data
                 lengths:pd.Series,
                 components_taxonomy:pd.Series,
                 components_functional:pd.Series,
                 # Labels
                 label_unknown:str="unknown_function_collective",
                 attr_type="ORF",
                 obsv_type=None,
                 functional_type=None,
                 name=None,
                 description=None,
                 # Misc
                 delimiter = "|",
                ):
        # Input data
        self.lengths = lengths
        self.components_taxonomy = components_taxonomy
        self.components_functional = components_functional
        # Description
        self.name = name
        self.description = description
        # Data Types
        self.attr_type = attr_type
        self.obsv_type = obsv_type
        self.functional_type = functional_type
        # Miscellaneous
        self.label_unknown = label_unknown
        self.delimiter = delimiter
        self.__synthesized__ = datetime.datetime.utcnow()
        # ==========
        # Initialize
        # ==========
        print(f"Phylogenomic Functional Components",
               "==================================",
               sep="\n",
               file=sys.stderr,
         )
        # Assign ORFs to PGFCs
        self.pgfc_collection = self._assign_orfs_to_pgfcs()
        # Compile metadata for each PGFC
        self.metadata = self._get_metadata()
        # Datasets
        self.datasets = OrderedDict()
    def __repr__(self):
        class_name = str(self.__class__).split(".")[-1][:-2]
        return f"{class_name}(name:{self.name}, num_datasets:{len(self.datasets)}, num_pgfcs:{self.metadata.shape[0]})"

    # Assign ORFS to each PGFC
    def _assign_orfs_to_pgfcs(self):
        pgfc_collection = defaultdict(list)
        for taxonomy_component, Se_data in tqdm(self.components_taxonomy.groupby(self.components_taxonomy), f"Assigning ORFs"):
            idx_orfs_for_taxonomycomponent = set(Se_data.index)
            # Process missing functions
            idx_orfs_with_function = set(self.components_functional.index)
            idx_orfs_no_function = idx_orfs_for_taxonomycomponent - idx_orfs_with_function
            if len(idx_orfs_no_function) > 0:
                id_pgfc = self.delimiter.join([taxonomy_component, self.label_unknown])
                pgfc_collection[id_pgfc] = list(idx_orfs_no_function)
                del id_pgfc # Reset the namespace
            # Process those with functional category
            for id_orf, functional_list in self.components_functional[idx_orfs_for_taxonomycomponent & idx_orfs_with_function].iteritems():
                for functional_component in functional_list:
                    id_pgfc = self.delimiter.join([taxonomy_component, functional_component])
                    pgfc_collection[id_pgfc].append(id_orf)
        # Convert list to non-redundant sets
        return {id_pgfc:set(orfs) for id_pgfc, orfs in pgfc_collection.items()}

    # Get metadata for PGFCs
    def _get_metadata(self):
        metadata = defaultdict(dict)
        for id_pgfc, orfs in tqdm(self.pgfc_collection.items(), "Compiling metadata"):
            # Length
            query_lengths = self.lengths[orfs].values
            try:
                k2, p = stats.normaltest(query_lengths)
            except ValueError:
                k2 = p = np.nan
            metadata[id_pgfc]["orfs"] = orfs
            metadata[id_pgfc]["number_of_orfs"] = len(query_lengths)
            metadata[id_pgfc]["lengths"] = list(query_lengths)
            metadata[id_pgfc]["µ(lengths)"] = query_lengths.mean()
            metadata[id_pgfc]["∑(lengths)"] = query_lengths.sum()
            metadata[id_pgfc]["sem(lengths)"] = stats.sem(query_lengths)
            metadata[id_pgfc]["var(lengths)"] = np.var(query_lengths)
            metadata[id_pgfc]["normaltest__statistic"] = k2
            metadata[id_pgfc]["normaltest__pvalue"] = p
            metadata[id_pgfc]["taxonomy_component"] = id_pgfc.split(self.delimiter)[0]
            if id_pgfc.endswith(self.label_unknown):
                metadata[id_pgfc]["functional_component"] = np.nan
            else:
                metadata[id_pgfc]["functional_component"] = id_pgfc.split(self.delimiter)[-1]
        # Metadata
        df_meta = pd.DataFrame(metadata).T.loc[:,["taxonomy_component", "functional_component","number_of_orfs", "∑(lengths)", "µ(lengths)","sem(lengths)","var(lengths)","normaltest__statistic", "normaltest__pvalue", "lengths", "orfs"]].sort_values(["taxonomy_component", "functional_component", "number_of_orfs"], ascending=[True, True, False])
        df_meta.index.name = "id_phylogenomic-functional-component"

        # Ints
        for id_field in ["number_of_orfs", "∑(lengths)"]:
            df_meta[id_field] = df_meta[id_field].astype(int)
        # Floats
        for id_field in ["µ(lengths)","sem(lengths)","var(lengths)","normaltest__statistic", "normaltest__pvalue"]:
            df_meta[id_field] = df_meta[id_field].astype(float)
        return df_meta

    # Compute the summed ORF counts for each PGFC
    def compute_dataset(self, df_orfs:pd.DataFrame, name=None, copy_data=True, return_dataset=True, **attr):
        assert all(set(self.components_taxonomy.index) <= query_gene_set for query_gene_set in [set(self.lengths.index), set(df_orfs.columns)]), f"All genes in `components_taxonomy` must be in `lengths` and `df_orfs`"
        assert set(df_orfs.columns) <= set(self.lengths.index), f"All genes in `df_orfs` must be in `lengths`"
        components_taxonomy = self.components_taxonomy.dropna()
        components_functional = self.components_functional.dropna()

        # Name
        if name is None:
            name = len(self.datasets)

        # Sum ORFs for each PGFC
        d_pgfc_counts = dict()
        for id_pgfc, orfs in tqdm(self.pgfc_collection.items(), f"Summing ORFs for dataset = `{name}`"):
            d_pgfc_counts[id_pgfc] = df_orfs.loc[:,orfs].sum(axis=1) # Removed set construction here because orfs are non-redundant from _assign_orfs_to_pgfcs
        df_pgfc = pd.DataFrame(d_pgfc_counts).sort_index(axis=1)

        # Store dataset
        if copy_data:

            self.datasets[name] = {
                "shape_input":df_orfs.shape,
                "counts":df_pgfc,
                "shape_output":df_pgfc.shape,
                **attr,
            }
        if return_dataset:
            return df_pgfc
        else:
            return self
    # Get PGFC counts
    def get_dataset(self, name):
        assert name in self.datasets, f"{name} not in `datasets`.  Try`compute_dataset` and repeat."
        return self.datasets[name]["counts"]
    # Write object to file
    def to_file(self, path:str, compression="infer"):
        write_object(self, path=path, compression=compression)
        return self

# Get taxonomy lineage
def get_taxonomy_lineage_from_identifier(identifiers, name=None, translate_ids=True, ncbi:ete3.NCBITaxa=None, verbose=0, mode="infer", include_taxid=True, include_levels=["id_taxon", "phylum", "class", "order", "family", "genus", "species"]):
    """
    Input: Single Taxonomy ID or a collection of identifiers w/ {id_orf:id_taxon}
    Output: pd.Series/pd.DataFrame of rank and taxonomy labels
    modes: {'infer', 'batch', 'singular'}
    """
    accepted_modes = {'batch', 'singular'}
    verbose = int(verbose)
    # Get database
    if ncbi is None:
        ncbi = ete3.NCBITaxa()

    # Infer mode
    if mode == "infer":
        if is_dict_like(identifiers):
            mode = "batch"
        else:
            mode = "singular"
        if verbose > 0:
            print(f"Inferred mode: {mode}", file=sys.stderr)
        assert mode != "infer", "Cannot infer `mode`.  Please explicitly provide mode."

    # Singular
    if mode == "singular":
        # Handle missing data
        if pd.isnull(identifiers):
            return pd.Series([], name=name)

        # Access the database
        try:
            id_taxon = int(identifiers)
            if name is None:
                name = id_taxon
            lineage = ncbi.get_lineage(id_taxon)
            ranks = dict(filter(lambda x: x[1] != "no rank", ncbi.get_rank(lineage).items()))
            Se_taxonomy = pd.Series(ncbi.get_taxid_translator(ranks.keys()), name=name)
            if translate_ids:
                Se_taxonomy.index = Se_taxonomy.index.map(lambda x:ranks[x])
            return Se_taxonomy

        # Handle taxonomy IDs that are not in database
        except ValueError:
            if verbose > 1:
                print(id_taxon, file=sys.stderr)
            return pd.Series([], name=name)
    # Batch
    if mode == "batch":
        if not is_query_class(identifiers, "Series"):
            identifiers = pd.Series(identifiers)

        # Group each taxonomy identifier
        dataframes = list()
        for id_taxon, group in tqdm(pd_series_collapse(identifiers).iteritems(), "Searching lineage from taxonomy identifier"):
            number_of_orfs_in_group = len(group)
            Se_taxonomy = get_taxonomy_lineage_from_identifier(identifiers=id_taxon, name=None, translate_ids=translate_ids, ncbi=ncbi, verbose=verbose, mode="singular")
            df_taxon = pd.DataFrame(number_of_orfs_in_group*[Se_taxonomy])
            df_taxon.index = group
            if include_taxid:
                df_taxon["id_taxon"] = id_taxon
            dataframes.append(df_taxon)
        df_collection = pd.concat(dataframes, axis=0)
        if include_levels is None:
            include_levels = df_collection.columns
        else:
            include_levels = [*filter(lambda level:level in df_collection.columns, include_levels)]

        idx_missing_orfs = set(identifiers.index) - set(df_collection.index)
        number_of_orfs_missing = len(idx_missing_orfs)
        if number_of_orfs_missing > 0:
            A = np.empty((number_of_orfs_missing, df_collection.shape[1]))
            A[:] = np.nan
            df_missing = pd.DataFrame(A, index=idx_missing_orfs, columns=df_collection.columns)
            return pd.concat([df_collection, df_missing], axis=0).loc[identifiers.index, include_levels]
        else:
            return df_collection.loc[identifiers.index, include_levels]

# Get taxonomy identifiers from string name_taxonomy
def get_taxonomy_identifier_from_name(name_taxonomy, mode="infer", ncbi=None, into=pd.Series):
    """
    name_taxonomy can be either a single string or iterable of strings
    """
    accepted_modes = {'batch', 'singular'}

    # Get database
    if ncbi is None:
        ncbi = ete3.NCBITaxa()

    if mode == "infer":
        if not is_nonstring_iterable(name_taxonomy):
            mode = "singular"
        else:
            mode = "batch"
        assert mode != "infer", "Cannot infer `mode`.  Please explicitly provide mode."

    if mode == "singular":
        name_taxonomy = [name_taxonomy]
    name_taxonomy = [*map(lambda x:x.strip(),name_taxonomy)]
    Se_taxonomy = pd.Series(ncbi.get_name_translator(name_taxonomy)).sort_index().map(lambda x:x[0])
    if mode == "singular":
        return Se_taxonomy.values[0]
    if mode == "batch":
        idx_missing = set(name_taxonomy) - set(Se_taxonomy.index)
        number_of_name_taxonomy_missing = len(idx_missing)
        if number_of_name_taxonomy_missing > 0:
            Se_missing = pd.Series([np.nan]*number_of_name_taxonomy_missing, index=idx_missing)
            Se_taxonomy = pd.concat([Se_taxonomy, Se_missing])
        return into(Se_taxonomy)

# Infer taxonomy for a sequence grouping
def infer_taxonomy(
        grouping_sequence:pd.Series,
        grouping_taxonomy:pd.Series,
        percent_identities:pd.Series,
        grouping_protein_domains:pd.Series=None,
        include_protein_domains=None
    ):
    """
    Format:
    group_sequence:pd.Series where {id_orf:contig-or-bin-identifier}
    grouping_taxonomy:pd.Series where {id_orf:taxonomy-label}
    percent_identities:pd.Series where {id_orf:percent-identity}
    grouping_protein_domains:pd.Series where {id_orf:[PFAM_1, TIGRFAM_2, etc.]} (Optional)
    include_protein_domains: iterable of target protein-domain names (Optional)
    """
    # Comparable set
    idx_comparable = grouping_sequence.index & grouping_taxonomy.index & percent_identities.index

    # Subset data based on HMMs
    if include_protein_domains is not None:
        assert grouping_protein_domains is not None, "If `include_protein_domains` is provided then you must also provide `grouping_protein_domains`"
        include_protein_domains = set(include_protein_domains)
        grouping_protein_domains = grouping_protein_domains[grouping_protein_domains.map(bool)]
        mask_include_protein_domains = grouping_protein_domains.map(lambda x: bool(set(x) & include_protein_domains))
        grouping_protein_domains = grouping_protein_domains[mask_include_protein_domains]
        idx_comparable = idx_comparable & grouping_protein_domains.index

    # Convert percent identities to ratios
    if percent_identities.max() > 1.0:
        percent_identities = percent_identities/100

    # Infer taxonomy
    d_seqgroup_hits = defaultdict(lambda: defaultdict(int))
    d_seqgroup_weights = defaultdict(lambda: defaultdict(float))

    data = pd.DataFrame([grouping_sequence, grouping_taxonomy, percent_identities]).T.loc[idx_comparable,:].values

    for i, (id_orf, row_data) in tqdm(enumerate(zip(idx_comparable, data)), "Summing weights from ORFs"):
        id_seqgroup, id_taxon, p_identity = row_data
        d_seqgroup_hits[id_seqgroup][id_taxon] += 1
        d_seqgroup_weights[id_seqgroup][id_taxon] += p_identity

    # Calculate scores
    d_seqgroup_inferred = defaultdict(dict)
    for id_seqgroup, d_taxa_weight in tqdm(d_seqgroup_weights.items(), "Scoring taxonomy for each sequence group"):
        w = pd.Series(d_taxa_weight, name=id_seqgroup)
        scores = w/w.sum()
        d_seqgroup_inferred[id_seqgroup]["weights"] = sorted(w.iteritems(), key=lambda x:x[-1], reverse=True)
        max_score = scores.max()
        d_seqgroup_inferred[id_seqgroup]["score"] = max_score
        if max_score < 1:
            max_scoring_taxa = scores[lambda x: x == max_score].index
            if max_scoring_taxa.size > 1:
                inferred_taxonomy = list(max_scoring_taxa) # Added 2018-11-30
            else:
                inferred_taxonomy = max_scoring_taxa[0]
        else:
            inferred_taxonomy = scores.index[0]
        d_seqgroup_inferred[id_seqgroup]["inferred_taxonomy"] = inferred_taxonomy
        d_seqgroup_inferred[id_seqgroup]["number_of_total_hits"] = sum(d_seqgroup_hits[id_seqgroup].values())

    df_inferred = pd.DataFrame(d_seqgroup_inferred).T.loc[:,[ "inferred_taxonomy", "score", "number_of_total_hits", "weights"]]
    df_inferred.index.name = "id_seqgroup"
    return df_inferred

# Group Otus by higher taxonomy level
def otu_to_level(X:pd.DataFrame, taxonomy:pd.DataFrame, level):
    """
    X => pd.DataFrame of Otu counts with columns as Otus and rows as samples
    taxonomy => pd.DataFrame of taxonomy assignments for Otus
    level => a taxonomy level in the `taxonomy` dataframe
    """
    return X.groupby(taxonomy[level], axis=1).sum()

# Alpha diversity calculations
def alpha_diversity(X:pd.DataFrame, metric="richness", taxonomy:pd.DataFrame=None, mode="infer", idx_taxonomy=['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'],  name=None, base=2, obsv_type=None, **alpha_kws):
    """
    X => pd.DataFrame of Otu counts with columns as Otus and rows as samples
    metric => a callable or a string {entropy, richness, gini, singletons}
    taxonomy => pd.DataFrame of taxonomy assignments for Otus
    """
    # Alpha measures
    def _entropy(x, **alpha_kws):
        return stats.entropy(x, base=base, **alpha_kws)
    def _richness(x, **alpha_kws):
        return (x > 0).sum().astype(int)
    def _gini(x, **alpha_kws):
        return skbio.diversity.alpha.gini_index(x, **alpha_kws)
    def _singletons(x, **alpha_kws):
        return (x == 1).sum().astype(int)
    d_metric_fn = {"entropy":_entropy, "richness":_richness, "gini":_gini, "singletons":_singletons}

    # Supported diversity measures
    if hasattr(metric, "__call__"):
        func = metric
    else:
        supported_metrics =  list(d_metric_fn.keys())
        assert metric in supported_metrics, f"`{metric}` is not compatible.  Only available alpha diversity measures are {supported_metrics}"
        func =  d_metric_fn[metric]
        name = metric

    # Compute diversity
    if mode == "infer":
        mode = "batch" if taxonomy is not None else "singular"

    assert mode in {"singular", "batch"}, "Please specify either 'singular', 'batch', or 'infer' for the mode"
    if mode == "singular":
        Se_alpha = X.apply(lambda x: func(x, **alpha_kws), axis=1)
        Se_alpha.index.name = name
        return Se_alpha

    if mode == "batch":
        assert taxonomy is not None, "`taxonomy` cannot be `None` when `mode='batch'`"
        d_level_metric = OrderedDict()
        for level in idx_taxonomy:
            if level in taxonomy.columns:
                df_level = otu_to_level(X, taxonomy, level=level)
                d_level_metric[level] = alpha_diversity(df_level, metric=metric, taxonomy=None, mode="singular", base=base, **alpha_kws)
            else:
                print(f"Skipping taxonomy level `{level}` because it is not in the taxonomy dataframe", file=sys.stderr)

        d_level_metric["Otu"] = alpha_diversity(X, metric=metric, taxonomy=None, mode="singular", base=base, **alpha_kws)
        df_level_metric = pd.DataFrame(d_level_metric)
        df_level_metric.index.name = f"id_{obsv_type}"
        return df_level_metric


# Compute the beta diversity
def beta_diversity(X:pd.DataFrame, y:pd.Series, label_intra="Intra", label_inter="Inter", label_metric="Distance", class_type="Class"):
    """
    X: a symmetric pd.DataFrame distance matrix with a diagonal of 0
    y: a pd.Series of class labels
    """
    if isinstance(X, Symmetric):
        X = X.to_dense()
    # Assertions
    assert is_symmetrical(X, tol=1e-10), "X must be symmetric"
    assert np.all(X.index == X.columns), "X.index != X.columns"
    assert set(X.index) <= set(y.index), "Not all elements in X.index are in y.index"
    assert np.all(X.notnull()), "X cannot contain any NaN.  Please preprocess the data accordingly"

    # Get indexable arrays
    idx_labels = X.index
    loc_labels = np.arange(X.shape[0])
    A = X.values

    # Calculate the diversity
    diversity_data = defaultdict(dict)
    for id_class, idx_class in pd_series_collapse(y[idx_labels], type_collection=pd.Index).items():
        loc_class = idx_class.map(lambda x:idx_labels.get_loc(x)).values
        loc_nonclass = np.delete(loc_labels, loc_class)
        # Intra
        intra_distances = squareform(A[loc_class,:][:,loc_class])
        diversity_data[(id_class, label_intra)] = pd.Series(intra_distances)
        # Inter
        inter_distances = A[loc_class,:][:,loc_nonclass].ravel()
        diversity_data[(id_class, label_inter)] = pd.Series(inter_distances)

    # Create output
    df_beta_diversity = pd.DataFrame(diversity_data)
    df_beta_diversity.columns.names = [class_type, "Diversity"]
    return df_beta_diversity





# Reverse Complement
def reverse_complement(seq:str):
    """
    Inputs a sequence of DNA and returns the reverse complement se$
    Outputs a reversed string of the input where A>T, T>A, G>C, an$
    """
    conversion = str.maketrans('ACGTacgt','TGCAtgca')
    comp = seq.translate(conversion)
    rev_comp = comp[::-1]
    return rev_comp

# # Murmurhash of K-mer
# def hash_kmer(kmer, random_state=0):
#     """
#     Adapted from the following source:
#     https://sourmash.readthedocs.io/en/latest/kmers-and-minhash.html
#     """
#     kmer = kmer.upper()

#     # Calculate the reverse complement
#     r_kmer = reverse_complement(kmer)

#     # Determine whether original k-mer or reverse complement is lesser
#     if kmer < r_kmer:
#         canonical_kmer = kmer
#     else:
#         canonical_kmer = r_kmer

#     # Calculate murmurhash using a hash seed
#     hash = mmh3.hash(canonical_kmer, seed=random_state, signed=False)
# #     if hash < 0:
# #         hash += 2**64

    return hash

def count_kmers(sequences, K=4, frequency=True, collapse_reverse_complement=False, pseudocount=0, fill_missing_kmers=False):
    """
    sequences: (1) a sequence string, (2) a dict_like of (id,seq) pairs, or (3) path to fasta
    collapse_reverse_complement reduces data dimensionality
    """
    def _infer_mode(sequences):
        mode = None
        if isinstance(sequences, str):
            if not os.path.exists(sequences):
                return "string_sequence"
            if os.path.exists(sequences):
                return "filepath"
        if is_dict_like(sequences):
            return "dict_like"
        assert mode is not None, "Unrecognized format:  Please provide: (1) a sequence string, (2) a dict_like of (id,seq) pairs, or (3) path to fasta"


    # Get mapping to collapse a strand
    def _collapse(kmers):
        kmer_collapse = dict()
        for kmer in kmers:
            r_kmer = reverse_complement(kmer)
            if r_kmer in kmer_collapse:
                kmer_collapse[kmer] = r_kmer
            else:
                kmer_collapse[kmer] = kmer
        return kmer_collapse

    def _count(seq:str, K, frequency, collapse_reverse_complement,  pseudocount, name=None):
        """
        Collapse only works for nondegenerate kmers
        """
        seq = seq.upper()
        alphabet = {"A", "C", "G", "T"}
        #query_alphabet = set(list(seq))
        # degenerate_kmers = {'B', 'D', 'H', 'K', 'M', 'N', 'R', 'S', 'V', 'W', 'Y'}
        #assert query_alphabet <= alphabet, "This cannot be used with degenerate characters"

        # Count kmers
        kmer_counts = defaultdict(int)
        for kmer in fragment(seq=seq, K=K, step=1, overlap=True):
            if set(kmer) <= alphabet:
                kmer_counts[kmer] += 1
        kmer_counts = pd.Series(kmer_counts, name=name) + pseudocount

        # Collapse double stranded
        if collapse_reverse_complement:
            kmer_collapse = _collapse(sorted(kmer_counts.index))
            kmer_counts = kmer_counts.groupby(kmer_collapse).sum()

        # Normalize
        kmer_counts = kmer_counts + pseudocount
        if frequency:
            return kmer_counts/kmer_counts.sum()
        if not frequency:
            return kmer_counts

    # Infer mode
    mode = _infer_mode(sequences)

    # Single sequence (string)
    if mode == "string_sequence":
        return _count(seq, K=K, frequency=frequency, collapse_reverse_complement=collapse_reverse_complement, pseudocount=pseudocount)
    else:
        # Multiple sequences
        params = dict(K=K, frequency=False, collapse_reverse_complement=False, pseudocount=0)
        f_sequences = None
        # Dict-like
        if mode == "dict_like":
            iterable = sequences.items()

        # Read in files
        if mode == "filepath":
            f_sequences = open(sequences, "r")
            iterable = SimpleFastaParser(f_sequences)

        # Count kmers
        data = list()
        for id, seq in tqdm(iterable, "Counting k-mers"):
            data.append(_count(name=id, seq=seq, **params))
        # Close file handle if necessary
        if f_sequences is not None:
            f_sequences.close()
        df_kmer_counts = pd.DataFrame(data)

        # Fill empty
        if fill_missing_kmers:
            missing_kmers = set(["".join(kmer) for kmer in itertools.product("ATCG",repeat=K)]) - set(df_kmer_counts.columns)
            A = np.zeros((df_kmer_counts.shape[0], len(missing_kmers)))
            df_missing = pd.DataFrame(A, index=df_kmer_counts.index, columns=list(missing_kmers))
            df_kmer_counts = pd.concat([df_kmer_counts, df_missing], axis=1)

        # Sort K-mers and fill null
        df_kmer_counts = df_kmer_counts.sort_index(axis=1, ascending=True).fillna(0)

        # Collapse Double-Stranded
        if collapse_reverse_complement:
            kmer_collapse = _collapse(df_kmer_counts.columns)
            df_kmer_counts = df_kmer_counts.groupby(kmer_collapse, axis=1).sum()

        # Pseudocount
        df_kmer_counts = df_kmer_counts + pseudocount

        # Normalize
        if frequency:
            return normalize(df_kmer_counts, method="tss", axis=1)
        else:
            return df_kmer_counts.astype(int)

# Prevalence
def prevalence(X:pd.DataFrame, interval_type="closed", name=None, scale=False) -> pd.Series:
    # Dimensions
    n,m = X.shape

    assert_acceptable_arguments(query=[interval_type],target=["closed", "open"], operation="le")

    # Attribtue depth
    data = (X > 0).sum(axis=0)

    # Comparison operation
    operation = {"closed":operator.ge, "open":operator.gt}[interval_type]

    # Number of attributes for tolerance
    d_tol_numattrs = OrderedDict()
    for tol in range(1,n+1):
        d_tol_numattrs[tol] = data.map(lambda x: operation(x,tol)).sum()

    # Prevalence
    x_prevalence = pd.Series(d_tol_numattrs, name=name)

    # Scale the prevalence
    if scale:
        x_prevalence.index = x_prevalence.index/n
    return x_prevalence
