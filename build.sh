#!/bin/sh

source ./config.sh
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" ..
make
cd ..
