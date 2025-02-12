# Gloss_py 

These are the python bindings for Gloss

## Python build local 
This build and installs the package for local development. 
It allows you to import gloss_py and try the examples.
Build local also generates stubs for the python bindings. This is optional but highly recommended.
This would let us see all methods under classes and function signatures in IDEs, making gloss easier to use.

`./scripts/build_local.sh`

## Python build wheel for sharing
This build a wheel that can be used to share with others or possibly publish to pypi.
However for building and publishing in one step, you should use `build_and_publish.sh`

`./scripts/build_wheel.sh`

## Python build and publish on pypi
`./scripts/build_and_publish.sh`

