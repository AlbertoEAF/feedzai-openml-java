#!/usr/bin/env bash

# Copyright (c) 2020 Feedzai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# @author Sheng Wang (sheng.wang@feedzai.com)

VERSION=v2.3.1 # choose your LightGBM build version (git tag/commit/etc.)




echo "Checking built version..."
OUTPUT_FOLDER=lightgbmlib_build
BUILT_VERSION="$(cat $OUTPUT_FOLDER/__version__)"

pwd

set -e
SUCCESS=0
function at_exit {
    if [[ "$SUCCESS" == "0" ]]; then
        echo "LightGBM build failed!"
    else
        echo "LightGBM build finished!"
    fi
}
trap at_exit EXIT

echo "Updating make-lightgbm submodule..."
git submodule update --init --recursive

if [[ "$BUILT_VERSION" != "$VERSION" ]]; then
  rm -rf $OUTPUT_FOLDER
  echo "entering the folder."
  cd make-lightgbm
  echo "starting run the script"
  bash make.sh "$VERSION"
  echo "exiting the folder."
  cd ..
  echo "move the folder."
  mv make-lightgbm/build $OUTPUT_FOLDER
fi

SUCCESS=1
