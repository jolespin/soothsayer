## Create Soothsayer Environment
now=$(date +"%T")
os_system=$(uname)
env_name=${1:-"soothsayer_py3.9_env"} # Default: soothsayer_env
py_version=${2:-"3.9"} # Default: 3.9.x

echo "Start Time: $now"
echo "Creating a conda environment to run soothsayer and all dependencies: $env_name"
echo "Using Python version: $py_version"

# Package manager
PACKAGE_MANAGER=$(type -P mamba) || PACKAGE_MANAGER=$(type -P conda)
echo "Using the following package manager: ${PACKAGE_MANAGER}"

${PACKAGE_MANAGER} create -n $env_name -y python=$py_version
source activate $env_name

${PACKAGE_MANAGER} install -n $env_name -y -c conda-forge 'pandas>=1.2.4' biopython scikit-bio  statsmodels scikit-learn xarray seaborn numpy 'networkx >= 2' 'scipy >= 1' 'matplotlib >= 3'  tqdm graphviz  pygraphviz fastcluster palettable matplotlib-venn adjusttext tzlocal  r-ape r-devtools rpy2 leidenalg python-igraph ete3
#libopenblas needed for edgeR and metagenomeSeq

${PACKAGE_MANAGER} install -n $env_name -y -c bioconda r-dynamictreecut bioconductor-philr bioconductor-edger bioconductor-aldex2 bioconductor-phyloseq gneiss bioconductor-lpsymphony r-fastcluster # genomeinfodbdata is for ALDEx2

## May need to run the following line if R packages fail from stringi.dylib
# conda install -c r r-stringi

conda activate $env_name
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
