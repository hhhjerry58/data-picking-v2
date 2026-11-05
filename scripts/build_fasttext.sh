#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="$repo_dir/.venv/bin/python"
extension_suffix="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
read -r -a python_includes <<< "$("$python_bin" -m pybind11 --includes)"
fasttext_sources=(args dictionary matrix densematrix quantmatrix vector model utils loss productquantizer meter fasttext)
source_files=()
for source_name in "${fasttext_sources[@]}"; do source_files+=("vendor/fasttext/src/$source_name.cc"); done
c++ -O3 -funroll-loops -DNDEBUG -march=native -std=c++17 -shared -fPIC -fvisibility=hidden -fvisibility-inlines-hidden -pthread "${python_includes[@]}" -Ivendor/fasttext/src src/picking/fasttext_native.cpp "${source_files[@]}" -o "src/picking/_fasttext_native${extension_suffix}"
make -C vendor/fasttext -j8
