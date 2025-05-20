#!/usr/bin/env bash

 # Copyright (C) 2018-present, Facebook, Inc.
 #
 # This program is free software; you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation; version 2 of the License.
 #
 # This program is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License along
 # with this program; if not, write to the Free Software Foundation, Inc.,
 # 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

set -xeo pipefail

ROOT_DIR=$(pwd)

# Useful constants
COLOR_RED="\033[0;31m"
COLOR_GREEN="\033[0;32m"
COLOR_OFF="\033[0m"
UPDATE_SUBMODULES=0
VERBOSE=0

usage() {
cat 1>&2 <<EOF

Usage ${0##*/} [-h|?][-u][-p PATH]
  -p BUILD_DIR                       (optional): Path to the base dir for strobelight
  -v                                 (optional): make it verbose (even more)
  -u                                 (optional): Update Submodules
  -h|?                                           Show this help message
EOF
}

while getopts ":hp:v:u" arg; do
  case $arg in
    p)
      BUILD_DIR="${OPTARG}"
      ;;
    v)
      VERBOSE=1
      ;;
    u)
      UPDATE_SUBMODULES=1
      ;;
    h | *) # Display help.
      usage
      exit 0
      ;;
  esac
done

# Validate required parameters
if [ -z "${BUILD_DIR-}" ] ; then
  echo -e "${COLOR_RED}[ INFO ] Build dir is not set. So going to build into _build ${COLOR_OFF}"
  BUILD_DIR=${ROOT_DIR}/strobelight/src/_build
  mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR" || exit

get_cargo_blazesym() {
    if command -v cargo > /dev/null 2>&1; then
        echo -e "${COLOR_GREEN}Cargo already installed ${COLOR_OFF}"
        return
    fi

    echo -e "${COLOR_GREEN}Cargo not found, intaslling rust / cargo ${COLOR_OFF}"
    curl https://sh.rustup.rs -sSf | sh
    export PATH="$HOME/.cargo/bin/:$PATH"
}

update_submodules() {
    if [ "$UPDATE_SUBMODULES" -eq 1 ]; then
        git submodule update --init --recursive
    fi
}

build_strobelight() {
    pushd .
    STROBELIGHT_BUILD_DIR=$BUILD_DIR
    rm -rf "$STROBELIGHT_BUILD_DIR"
    mkdir -p "$STROBELIGHT_BUILD_DIR"

    cd "$STROBELIGHT_BUILD_DIR" || exit

    # Append verbose flag if VERBOSE is set to 1
    if [ "$VERBOSE" -eq 1 ]; then
        CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON"
    fi

    # Run CMake with the constructed flags
    cmake $CMAKE_FLAGS ..

    make
    popd
}

get_required_libs() {
    if [ -x "$(command -v dnf)" ]; then
       sudo dnf install \
            clang \
            elfutils-libelf \
            elfutils-libelf-devel\
            fmt-devel
    elif [ -x "$(command -v yum)" ]; then
        sudo yum install -y \
            git \
            elfutils-libelf-devel \
            fmt-devel
    elif [ -x "$(command -v apt)" ]; then
sudo apt install -y \
   git \
   cmake \
   clang \
   libfmt-dev
   else
echo "Package manager not found or not recognized."
    fi
}
update_submodules
get_required_libs
get_cargo_blazesym
build_strobelight
