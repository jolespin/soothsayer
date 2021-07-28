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

conda install -y -c r r-devtools 

pip install rpy2

conda install -y -c conda-forge 'pandas>=1.2.4' mmh3 biopython scikit-bio  statsmodels scikit-learn xarray seaborn numpy 'networkx >= 2' 'scipy >= 1' 'matplotlib >= 2'  tqdm graphviz  pygraphviz fastcluster palettable matplotlib-venn adjusttext tzlocal  r-ape 'libopenblas==0.3.7'  
#libopenblas needed for edgeR and metagenomeSeq

conda install -y -c bioconda r-dynamictreecut bioconductor-philr bioconductor-edger bioconductor-aldex2 bioconductor-phyloseq ete3 gneiss bioconductor-lpsymphony # genomeinfodbdata is for ALDEx2

## May need to run the following line if R packages fail from stringi.dylib
# conda install -c r r-stringi

pip install --no-deps soothsayer_utils
pip install --no-deps compositional
pip install --no-deps hive_networkx
pip install --no-deps ensemble_networkx
pip install --no-deps  soothsayer

# End
now=$(date +"%T")
echo "End Time: $now"
