## Create Soothsayer Environment
now=$(date +"%T")
os_system=$(uname)
env_name=${1:-"soothsayer_py3.9_env"} # Default: soothsayer_env
py_version=${2:-"3.9"} # Default: 3.9.x

echo "Start Time: $now"
echo "Creating a conda environment to run soothsayer and all dependencies: $env_name"
echo "Using Python version: $py_version"

conda create -n $env_name -y python=$py_version
source activate $env_name

conda config --add channels defaults
conda config --add channels conda-forge
conda config --add channels bioconda
# conda config --add channels jolespin

conda install -y -c r r-devtools 
#'rpy2 >=2.9.4 < 3.0'

# pip install 'rpy2==2.9.6b'
pip install rpy2

# conda install -y -c conda-forge 'libopenblas==0.3.7' r-ape biopython scikit-bio  scikit-learn xarray seaborn numpy 'networkx >= 2' 'scipy >= 1' 'matplotlib >= 2'  tqdm graphviz  pygraphviz fastcluster palettable matplotlib-venn adjusttext tzlocal r-propr 
conda install -y -c conda-forge mmh3 biopython scikit-bio  statsmodels scikit-learn xarray seaborn numpy 'networkx >= 2' 'scipy >= 1' 'matplotlib >= 2'  tqdm graphviz  pygraphviz fastcluster palettable matplotlib-venn adjusttext tzlocal r-propr r-ape 'libopenblas==0.3.7'  #libopenblas needed for edgeR and metagenomeSeq

#pyhamcrest pydot

# conda install -y -c bioconda r-dynamictreecut r-wgcna bioconductor-philr bioconductor-edger bioconductor-metagenomeseq bioconductor-phyloseq bioconductor-ggtree ete3 gneiss 'bioconductor-preprocesscore==1.48.0' 
conda install -y -c bioconda r-dynamictreecut bioconductor-philr bioconductor-edger bioconductor-aldex2 bioconductor-metagenomeseq bioconductor-phyloseq ete3 gneiss bioconductor-lpsymphony # genomeinfodbdata is for ALDEx2

pip install teneto

## May need to run the following line if R packages fail from stringi.dylib
# conda install -c r r-stringi

pip install --no-deps soothsayer_utils
pip install --no-deps compositional
pip install --no-deps hive_networkx
pip install --no-deps ensemble_networkx
pip install soothsayer --no-deps

# End
now=$(date +"%T")
echo "End Time: $now"
