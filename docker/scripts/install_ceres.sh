#!/usr/bin/env bash
set -euo pipefail

# Commit/branch/tag to build; defaults to your commit.
CERES_REF="2.2.0"

echo "Installing Ceres @ ${CERES_REF}"

# Fetch the exact ref robustly (works for commits, tags, or branches)
install_root=/opt/ceres
rm -rf "${install_root}"
git init "${install_root}"
git -C "${install_root}" remote add origin https://github.com/ceres-solver/ceres-solver.git
# Try shallow first; fall back to full fetch if needed
if ! git -C "${install_root}" fetch --no-tags --depth 1 origin "${CERES_REF}" 2>/dev/null; then
  git -C "${install_root}" fetch --no-tags origin "${CERES_REF}"
fi
git -C "${install_root}" checkout -q FETCH_HEAD
# Pull submodules (Abseil, GTest, etc.)
git -C "${install_root}" submodule update --init --recursive

# Configure & build
mkdir -p "${install_root}/build" "${install_root}/install"
gen=""
command -v ninja >/dev/null 2>&1 && gen="-G Ninja"

cmake -S "${install_root}" -B "${install_root}/build" ${gen} \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="${install_root}/install" \
  \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_BENCHMARKS=OFF \
  -DBUILD_DOCUMENTATION=OFF \
  -DSUITESPARSE=ON \
  -DEIGENSPARSE=ON \
  -DEIGENMETIS=ON \
  -DUSE_CUDA=OFF \
  -DLAPACK=ON \
  -DSCHUR_SPECIALIZATIONS=ON \
  -DCUSTOM_BLAS=ON

cmake --build "${install_root}/build" --parallel "$(nproc)"
cmake --install "${install_root}/build"

# Make discoverable: prefer Ceres_DIR or CMAKE_PREFIX_PATH
echo 'export Ceres_DIR=/opt/ceres/install/lib/cmake/Ceres' >> /etc/bash.bashrc
echo 'export CMAKE_PREFIX_PATH=/opt/ceres/install:${CMAKE_PREFIX_PATH}' >> /etc/bash.bashrc
echo "/opt/ceres/install/lib" > /etc/ld.so.conf.d/ceres.conf && ldconfig

echo "Ceres installed to ${install_root}/install @ $(git -C "${install_root}" rev-parse --short HEAD)"

# Clean up build tree
rm -rf "${install_root}/build"
