#!/usr/bin/env bash
set -e

if [ -z "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not set; please set it to the Android NDK directory"
  exit 1
fi

if [ ! -d "$ANDROID_NDK" ]; then
  echo "ANDROID_NDK not a directory; did you install it under $ANDROID_NDK?"
  exit 1
fi
if [ -z "$ANDROID_ABI" ]; then
  ANDROID_ABI="arm64-v8a"
fi

ANDROID_NDK_PROPERTIES="$ANDROID_NDK/source.properties"
[ -f "$ANDROID_NDK_PROPERTIES" ] && ANDROID_NDK_VERSION=$(sed -n 's/^Pkg.Revision[^=]*= *\([0-9]*\)\..*$/\1/p' "$ANDROID_NDK_PROPERTIES")
echo "Using Android NDK at $ANDROID_NDK"
echo "Android NDK version: $ANDROID_NDK_VERSION"

ROOT="$( cd "$(dirname "$0")"/. ; pwd -P)"

BUILD_ROOT="$ROOT/build_android"
INSTALL_PREFIX=${BUILD_ROOT}/install
mkdir -p $BUILD_ROOT
cd $BUILD_ROOT

CMAKE_ARGS=()
CMAKE_ARGS+=("-DCMAKE_VERBOSE_MAKEFILE=1")
CMAKE_ARGS+=("-DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')")
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake")
CMAKE_ARGS+=("-DANDROID_NDK=$ANDROID_NDK")
CMAKE_ARGS+=("-DANDROID_ABI=$ANDROID_ABI")
CMAKE_ARGS+=("-DANDROID_STL=c++_static")
CMAKE_ARGS+=("-DANDROID_PLATFORM=android-21")

if (( "${ANDROID_NDK_VERSION:-0}" < 18 )); then
  CMAKE_ARGS+=("-DANDROID_TOOLCHAIN=gcc")
else
  CMAKE_ARGS+=("-DANDROID_TOOLCHAIN=clang")
fi

cmake "$ROOT" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
    -DCMAKE_BUILD_TYPE=Debug \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ -z "$MAX_JOBS" ]; then
  if [ "$(uname)" == 'Darwin' ]; then
    MAX_JOBS=$(sysctl -n hw.ncpu)
  else
    MAX_JOBS=$(nproc)
  fi
fi

echo "Building... $INSTALL_PREFIX"
cmake --build . --target all -- "-j${MAX_JOBS}"
echo "Build complete $INSTALL_PREFIX"
