#!/bin/sh

source ./config.sh
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="${TORCH_ROOT}" -DTORCH_ROOT="${TORCH_ROOT}" -DQNETS_ROOT="${QNETS_ROOT}" ..
make
cd ..
