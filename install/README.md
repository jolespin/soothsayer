```

____ ____ ____ ___ _  _ ____ ____ _   _ ____ ____ 
[__  |  | |  |  |  |__| [__  |__|  \_/  |___ |__/ 
___] |__| |__|  |  |  | ___] |  |   |   |___ |  \ 
                                                  
```

## Installation instructions:
I've tried to make installation as seemless as possible.  There are several Python & R dependencies that are difficult to install but it should be working via `conda` with the preconfigured environment `.yml` files.  

Note, sometimes theres are issues with the conda installation from `-c jolespin soothsayer` but `pip install soothsayer` can be used instead.  

### Method 1 :
#### [OSX|Linux] Installation [Recommended]
Inspired by [qiime2](https://docs.qiime2.org/2019.4/install/native/) installation method.  Please use the most updated environment and then install the current developmental branch (described under `Update to the current release`).

```bash
# Download the conda environment instructions
wget https://raw.githubusercontent.com/jolespin/soothsayer/master/install/soothsayer_py36_v2019.06.osx.yml
# Create a new environment (you should probably do this from the base environment [conda activate base])
conda env create -y --name soothsayer_env --file soothsayer_py36_v2019.06.osx.yml
# [Optional] Remove the environment file
rm soothsayer_py36_v2019.06.osx.yml
# Activate environment
conda activate soothsayer_env
# Duration: This took < 5 minutes on `MacBook Pro v10.14.5` 

# For Linux, replace the `osx` with `linux`

```

### Method 2:
#### OSX | Linux | (Windows?) Installation
```bash
wget https://raw.githubusercontent.com/jolespin/soothsayer/master/install/install_soothsayer.sh
# bash install_soothsayer.sh  <env_name> <py_version>
bash install_soothsayer.sh 
# or
bash install_soothsayer.sh soothsayer_env
# or
bash install_soothsayer.sh soothsayer_env 3.6.7
# Duration: This will take a few hours and may require manually installing a few packages if certain ones fail.  
```

#### Update to the current release [Recommended]
Since `soothsayer` is still in a developmental stage, I'm constantly adding methods, fixing bugs, and moving code around.  You should run the following command to use the current version:

```bash
pip install git+https://github.com/jolespin/soothsayer

# pip install git+https://github.com/jolespin/soothsayer --force-reinstall --no-deps
```

#### Bug with `pip install` concerning `soothsayer`
There is a strange [issue](https://github.com/pypa/pip/issues/7170) that I'm working on with `PyPI`  where installing `soothsayer` via `pip` removes all of the packages in `site-directory`.  I've found a work around for the time being.  Please use the `remove_and_reinstall_soothsayer.sh` script to first remove an older instance of `soothsayer` and reinstall the newest version. 

Let me know if you have any issues before creating an issue on GitHub:
jespinoz[ a t ]jcvi[ d o t ]org


