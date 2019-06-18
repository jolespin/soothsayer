# Installing Soothsayer
I've tried to make installation as seemless as possible.  There are several Python & R dependencies that are difficult to install but it should be working through conda [OSX and Linux soon].

Current installation methods:

## Method 1 (similar to [qiime2](https://docs.qiime2.org/2019.4/install/native/) installation):
### OSX Installation
```bash
wget https://raw.githubusercontent.com/jolespin/soothsayer/master/install/soothsayer_py36_v2019.06.yml
conda env create -y --name soothsayer_env --file soothsayer_py36_v2019.06.yml
```

## Method 2:
### OSX|Linux|(Windows?) Installation
```bash
wget https://raw.githubusercontent.com/jolespin/soothsayer/master/install/install_soothsayer.sh
# bash install_soothsayer.sh  <env_name> <py_version>
bash install_soothsayer.sh 
# or
bash install_soothsayer.sh soothsayer_env
# or
bash install_soothsayer.sh soothsayer_env 3.6.8
```

Let me know if you have any issues before creating an issue on GitHub:
jespinoz[ a t ]jcvi[ d o t ]org
