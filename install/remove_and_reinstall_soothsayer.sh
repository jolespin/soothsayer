SITEPACKAGES=$(python -c "from site import getsitepackages; print(getsitepackages()[0])")

rm -r ${SITEPACKAGES}/soothsayer
rm -r ${SITEPACKAGES}/soothsayer-*


SOURCE=${1:-"git+https://github.com/jolespin/soothsayer"}

pip install --no-deps ${SOURCE}
