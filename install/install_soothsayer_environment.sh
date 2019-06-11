# install_main_environment.sh
now=$(date +"%T")
os_system=$(uname)
env_name=${1:-"soothsayer_env"} # Default: soothsayer_env
py_version=${2:-"3.6.6"} # Default: 3.6.6


echo "Start Time: $now"
echo "Creating a conda environment to run soothsayer and all dependencies: $env_name"
echo "Using Python version: $py_version"

## Create Main Environment
conda create -n $env_name python=$py_version --yes
source activate $env_name

conda install rpy2 --yes
conda install -c r r-devtools --yes

## Install Python Packages
source ./install_python_packages.sh

## Install R Packages
Rscript ./install_r_packages.r

## Install Soothsayer
pip install --upgrade --upgrade-strategy only-if-needed ..

# End
now=$(date +"%T")
echo "End Time: $now"
