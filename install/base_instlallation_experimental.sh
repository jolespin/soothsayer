## Create Soothsayer Environment
now=$(date +"%T")
os_system=$(uname)
env_name=${1:-"soothsayer_py3.7_env"} # Default: soothsayer_env
py_version=${2:-"3.7"} # Default: 3.6.x

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

conda install -y -c conda-forge 'libopenblas==0.3.7' r-ape biopython scikit-bio  scikit-learn xarray seaborn numpy 'networkx >= 2' 'scipy >= 1' 'matplotlib >= 2'  tqdm graphviz  pygraphviz fastcluster palettable matplotlib-venn adjusttext tzlocal r-propr 
#pyhamcrest pydot

conda install -y -c bioconda r-dynamictreecut r-wgcna bioconductor-philr bioconductor-edger bioconductor-metagenomeseq bioconductor-phyloseq bioconductor-ggtree ete3 gneiss 'bioconductor-preprocesscore==1.48.0'
pip install teneto

## May need to run the following line if R packages fail from stringi.dylib
# conda install -c r r-stringi
# pip install soothsayer --no-deps

# End
now=$(date +"%T")
echo "End Time: $now"
