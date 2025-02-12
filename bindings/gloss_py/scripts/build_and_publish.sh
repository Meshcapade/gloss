#!/bin/bash

#this build a wheel and publishes it to pypi
#TODO check maturin publich --help options and how to change the pypi repo


#basically this mounts the volume of gloss into /gloss inside the container and then runs  "maturin pubish <options>" command
# extension module is needed in order for manylinux to work https://pyo3.rs/v0.19.2/building_and_distribution.html#linking

# docker run  -v $(pwd)/../..:/gloss   $(docker build -q ./docker)    publish --release --features "extension-module"  
docker run  -v $(pwd)/../..:/gloss   $(docker build -q ./docker)   publish --features "pyo3/extension-module" --username meshy_laptop --password ${GITLAB_TOKEN}  --repository-url https://gitlab.com/api/v4/projects/53021447/packages/pypi --no-sdist

# maturin publish --username meshy_laptop --password glpat-TxrrpCz5Z9FP53R1-5QN  --repository-url https://gitlab.com/api/v4/projects/53021447/packages/pypi