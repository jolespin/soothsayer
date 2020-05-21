# Get date YYYY.MM
DATE=`date +%Y.%m`
# Get version (also takes user input)
VERSION=${1:-$DATE}
# Get cwd
CWD=$PWD
# Get the directory of the releases
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR
# Create a release
mkdir soothsayer_v${VERSION}
cp -r ../setup.py ../soothsayer ../standalone ../README.md ../meta.yaml ../MANIFEST.in ../logo.png ../license.txt ../install ../bin soothsayer_v${VERSION}
tar -zcf soothsayer_v${VERSION}.tar.gz soothsayer_v${VERSION}
# Make open permissions
chmod 777 soothsayer_v${VERSION}.tar.gz
# Remove the uncompressed release
rm -r soothsayer_v${VERSION}
# Go back to the og directory
cd $CWD
