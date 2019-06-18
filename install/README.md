```

____ ____ ____ ___ _  _ ____ ____ _   _ ____ ____ 
[__  |  | |  |  |  |__| [__  |__|  \_/  |___ |__/ 
___] |__| |__|  |  |  | ___] |  |   |   |___ |  \ 
                                                  
```

## Installation instructions:
<<<<<<< HEAD
I've tried to make installation as seemless as possible.  There are several Python & R dependencies that are difficult to install but it should be working through conda [OSX and Linux soon].

Current installation methods:

## Method 1 :
### OSX Installation
Inspired by [qiime2](https://docs.qiime2.org/2019.4/install/native/) installation method (kudos to them btw)
```bash
wget https://raw.githubusercontent.com/jolespin/soothsayer/master/install/soothsayer_py36_v2019.06.yml
conda env create -y --name soothsayer_env --file soothsayer_py36_v2019.06.yml
```

## Method 2:
### OSX|Linux|(Windows?) Installation
=======
I've tried to make installation as seemless as possible.  There are several Python & R dependencies that are difficult to install but it should be working through conda [OSX and Linux soon].  

Note, sometimes theres are issues with the conda installation from `-c jolespin soothsayer` but `pip install soothsayer` can be used instead.

##Current installation methods:

### Method 1 :
#### OSX Installation
Inspired by [qiime2](https://docs.qiime2.org/2019.4/install/native/) installation method

```bash
# Download the conda environment instructions
wget https://raw.githubusercontent.com/jolespin/soothsayer/master/install/soothsayer_py36_v2019.06.yml
# Create a new environment (you should probably do this from the base environment [conda activate base])
conda env create -y --name soothsayer_env --file soothsayer_py36_v2019.06.yml
# [Optional] Remove the environment file
rm soothsayer_py36_v2019.06.yml
# Activate environment
conda activate soothsayer_env
```

### Method 2:
#### OSX | Linux | (Windows?) Installation
>>>>>>> devel
```bash
wget https://raw.githubusercontent.com/jolespin/soothsayer/master/install/install_soothsayer.sh
# bash install_soothsayer.sh  <env_name> <py_version>
bash install_soothsayer.sh 
# or
bash install_soothsayer.sh soothsayer_env
# or
bash install_soothsayer.sh soothsayer_env 3.6.7
```

Let me know if you have any issues before creating an issue on GitHub:
jespinoz[ a t ]jcvi[ d o t ]org
