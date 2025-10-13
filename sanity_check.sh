set -e
pushd alpha-beta-CROWN-vnncomp2025/examples/simple
export PYTHONPATH=$(realpath ../../):$PYTHONPATH
python toy.py
popd 