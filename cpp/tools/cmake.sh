#!/bin/bash

#BUILD_TYPE=Debug
BUILD_TYPE=Release

pushd .
rm -rf build
mkdir build
cd build
cmake \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DMLIB_ROOT=$HOME/Projects/mLib \
  -DBOOST_ROOT=$HOME/Projects/Libraries/boost_1_61_0/ \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH;" \
  .. 
popd

