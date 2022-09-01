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

conda install -y -c conda-forge 'pandas>=1.2.4' mmh3 biopython scikit-bio  statsmodels scikit-learn xarray seaborn numpy 'networkx >= 2' 'scipy >= 1' 'matplotlib >= 3'  tqdm graphviz  pygraphviz fastcluster palettable matplotlib-venn adjusttext tzlocal  r-ape r-devtools rpy2 umap-learn leidenalg python-igraph ete3
#libopenblas needed for edgeR and metagenomeSeq

cconda install -y -c bioconda r-dynamictreecut bioconductor-philr bioconductor-edger bioconductor-aldex2 bioconductor-phyloseq gneiss bioconductor-lpsymphony r-fastcluster # genomeinfodbdata is for ALDEx2

## May need to run the following line if R packages fail from stringi.dylib
# conda install -c r r-stringi

pip install --no-deps git+https://github.com/jolespin/soothsayer_utils
pip install --no-deps git+https://github.com/jolespin/compositional
pip install --no-deps git+https://github.com/jolespin/hive_networkx
pip install --no-deps git+https://github.com/jolespin/ensemble_networkx
pip install --no-deps git+https://github.com/jolespin/soothsayer

echo "Checking soothsayer installation"
python -c "import soothsayer as sy"

# End
now=$(date +"%T")
echo "End Time: $now"
