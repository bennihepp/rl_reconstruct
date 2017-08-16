#!/bin/bash

#BUILD_TYPE=Debug
BUILD_TYPE=Release

pushd .
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE
popd

