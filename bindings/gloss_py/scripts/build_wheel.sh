#!/bin/bash

#this build a wheel that can be used to share with others or possibly publish to pypi 
#however for building and publishing in one step, you should use build_and_publish.sh

#basically this mounts the volume of gloss into /gloss inside the container and then runs  "maturin build --release --features "extension-module" command
# extension module is needed in order for manylinux to work https://pyo3.rs/v0.19.2/building_and_distribution.html#linking

docker run  -v $(pwd)/../..:/gloss   $(docker build -q ./docker)    build --release --features "pyo3/extension-module"  