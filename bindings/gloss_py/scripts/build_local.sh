#!/bin/bash

#this build and installs the package for local development. 
#it allows you to import gloss_py and try the examples
#more info https://github.com/PyO3/maturin/issues/284#issuecomment-1365638907

#we only create the virtual env if we are not in conda
if [ -z "${CONDA_PREFIX}" ]; then
    export VIRTUAL_ENV=$(python3 -c 'import sys; print(sys.base_prefix)')
else
    echo "We are in Conda so we already have an virtual environment for python"
fi
maturin develop 

# Generate stubs for the python bindings
python3 generate_stubs.py