install-build-deps () {
   apt-get update -y
   apt-get install clang vim tree lld ccache -y
   pip install pybind11
}


triton-pip-install () {
  REPO_BASE_DIR=$(git rev-parse --show-toplevel)
  #TRITON_BUILD_WITH_CCACHE=true TRITON_BUILD_WITH_CLANG_LLD=true \
  pip install --no-build-isolation ${REPO_BASE_DIR}
}


# <source-dir> should be the local checkout directory for
#   https://github.com/llvm/llvm-project/tree/main/llvm
# <target-dir> is where to put the compiled LLVM/MLIR artifacts
triton-configure-mlir() {
  if (( $# < 3 ))
  then
    echo "usage: $0 <source-dir> <target-dir> <build-type>"
    return 1
  fi

  SOURCE_DIR=$1; shift
  TARGET_DIR=$1; shift
  BUILD_TYPE=$1; shift

  cmake -G Ninja \
    -S ${SOURCE_DIR} -B ${TARGET_DIR} \
    -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/install \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;lld" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=$(which clang) -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86" "$@"
}


# <source-dir> should be the local checkout directory for
#   https://github.com/triton-lang/triton
# <target-dir> is where to put the compiled Triton artifacts
# <mlir-dir> should be the LLVM/MLIR artifacts directory
triton-cmake() {
  if (( $# < 4 ))
  then
    echo "usage: $0 <source-dir> <target-dir> <build-type> <mlir-dir>"
    return 1
  fi

  SOURCE_DIR=$(realpath $1); shift
  TARGET_DIR=$(realpath $1); shift
  BUILD_TYPE=$1; shift
  MLIR_DIR=$(realpath $1); shift

  if [[ "$(uname)" == "Darwin" ]]; then
    LINKER_FLAGS=()
  else
    LINKER_FLAGS=(
      "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld"
      "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld"
      "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld"
    )
  fi

  REPO_BASE_DIR=$(git rev-parse --show-toplevel)

  cmake -GNinja \
    -S ${SOURCE_DIR} -B ${TARGET_DIR} \
    -DTRITON_WHEEL_DIR=${TARGET_DIR}/wheel \
    -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/install \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DTRITON_CODEGEN_BACKENDS="amd;nvidia" \
    -DLLVM_INCLUDE_DIRS=${MLIR_DIR}/include \
    -DLLVM_LIBRARY_DIR=${MLIR_DIR}/lib \
    -DLLVM_SYSPATH=${MLIR_DIR} \
    -DCMAKE_C_COMPILER=$(which clang) -DCMAKE_CXX_COMPILER=$(which clang++) \
    -DCMAKE_LINKER=lld ${LINKER_FLAGS[@]} \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DTRITON_BUILD_PYTHON_MODULE=ON \
    -DTRITON_BUILD_PROTON=ON \
    -DCUPTI_INCLUDE_DIR=${REPO_BASE_DIR}/third_party/nvidia/backend/include \
    -DROCTRACER_INCLUDE_DIR=${REPO_BASE_DIR}/third_party/amd/backend/include \
    -DJSON_INCLUDE_DIR=$HOME/.triton/json/include "$@"
}

# Install build dependency
pip uninstall triton -y
rm -rf ~/.triton

install-build-deps

# Export environmetn variable to use clang provided by rocm
# export PATH=/opt/rocm/llvm/bin:$PATH

# Build LLVM/MLIR at the specific commit needed by Triton
cd llvm-project && git checkout $(cat ../triton/cmake/llvm-hash.txt) && cd ..
triton-configure-mlir llvm-project/llvm build/mlir-debug Debug
cmake --build build/mlir-debug

# Use triton-pip-install to download dependencies like NIVIDA toolchain
cd triton && triton-pip-install

# Build Triton itself
triton-cmake . ../build/triton-debug Debug ../build/mlir-debug
cmake --build ../build/triton-debug
