#!/bin/bash

BUILD_TYPE=Debug
#BUILD_TYPE=Release

pushd .
rm -rf build_debug
mkdir build_debug
cd build_debug
cmake \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DBOOST_ROOT=$HOME/Projects/Libraries/boost_1_61_0/ \
  -DMLIB_ROOT=$HOME/Projects/mLib \
  .. 
popd

