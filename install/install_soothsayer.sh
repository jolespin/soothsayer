## Create Soothsayer Environment
# soothsayer_version=2019.6
#env_name=soothsayer_${soothsayer_version}_py${py_version}

now=$(date +"%T")
os_system=$(uname)
env_name=${1:-"soothsayer_env"} # Default: soothsayer_env
py_version=${2:-"3.6"} # Default: 3.6.6

echo "Start Time: $now"
echo "Creating a conda environment to run soothsayer and all dependencies: $env_name"
echo "Using Python version: $py_version"

conda create -n $env_name -y python=$py_version
source activate $env_name

conda config --add channels defaults
conda config --add channels conda-forge
conda config --add channels bioconda

conda install -y -c r rpy2 r-devtools
conda install -y -c bioconda r-dynamictreecut r-wgcna bioconductor-philr bioconductor-edger bioconductor-metagenomeseq bioconductor-phyloseq bioconductor-ggtree ete3 gneiss
conda install -y -c conda-forge r-ape biopython scikit-bio pandas scikit-learn xarray seaborn numpy networkx scipy matplotlib astropy pyhamcrest tqdm graphviz pydot pygraphviz fastcluster palettable matplotlib-venn python-ternary adjusttext tzlocal
conda install -y -c jolespin soothsayer

# End
now=$(date +"%T")
echo "End Time: $now"
