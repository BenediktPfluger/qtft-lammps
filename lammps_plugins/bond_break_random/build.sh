#!/usr/bin/env bash
# Build bondbreakrandom.so for the lmp of the active conda env (lammps).
# Usage: ./build.sh [LAMMPS source folder]   (default ~/src/lammps; must be the same version as lmp)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="${1:-$HOME/src/lammps}"

want="$(lmp -h | sed -n 's/.*Simulator - //p' | head -1)"
have="$(sed -n 's/#define LAMMPS_VERSION "\(.*\)"/\1/p' "$src/src/version.h")"
if [ "$want" != "$have" ]; then
  echo "LAMMPS source $src is version '$have', lmp is '$want': check out the matching version" >&2
  exit 1
fi

cmake -S "$here" -B "$here/build" -D LAMMPS_SOURCE_DIR="$src/src" -D CMAKE_BUILD_TYPE=Release
cmake --build "$here/build" -j 4
echo "built $here/bondbreakrandom.so for LAMMPS $want"
