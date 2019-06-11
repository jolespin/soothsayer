# install_python_packages.sh
## Python
#### PyData
# conda install pandas=0.23.4 --yes
# conda install scikit-learn --yes
pip install "pandas>=0.24.2"
pip install "scikit-learn>=0.20.2"

conda install xarray --yes
conda install seaborn --yes

conda install numpy --yes
conda install networkx --yes
conda install scipy --yes
conda install astropy --yes

#### Utilities
pip install PyHamcrest
pip install tqdm
conda install cython --yes
conda install joblib --yes
conda install tzlocal --yes
conda install xlrd --yes
pip install adjustText
pip install mmh3

#### Visualization
conda install matplotlib=3.0.1 --yes

#### PyGraphViz
conda install graphviz --yes
pip download "git+git://github.com/pygraphviz/pygraphviz.git#egg=pygraphviz"
unzip pygraphviz*.zip
cd pygraphviz
python setup.py install --include-path=$CONDA_PREFIX/include --library-path=$CONDA_PREFIX/lib
cd ..
rm -rf pygraph*
pip install pydot

#### Biology
pip install ete3
# conda install -c conda-forge scikit-bio==0.5.1 --yes
conda install -c conda-forge scikit-bio --yes

conda install BioPython --yes
pip install git+https://github.com/biocore/gneiss.git

#### Machine Learning
conda install -c conda-forge fastcluster --yes

# Ternray plots
pip install git+https://github.com/marcharper/python-ternary
