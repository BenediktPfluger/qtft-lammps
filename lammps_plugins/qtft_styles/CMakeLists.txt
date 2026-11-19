# Plugin qtftstyles: the LAMMPS styles of qtft-lammps (fix bond/break/random, bond_style harmonic/anchor).
# Needs the LAMMPS source of exactly the version of the lmp that loads it (headers only):
# cmake -S . -B build -D LAMMPS_SOURCE_DIR=<lammps>/src; see build.sh.
cmake_minimum_required(VERSION 3.20)
project(qtftstyles VERSION 1.1 LANGUAGES CXX)

if(NOT LAMMPS_SOURCE_DIR)
  message(FATAL_ERROR "Set LAMMPS_SOURCE_DIR to the src/ folder of the LAMMPS source")
endif()
set(LAMMPS_HEADER_DIR ${LAMMPS_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI, integer sizes (smallbig) and OpenMP settings as in a LAMMPS build, from the LAMMPS plugin example
list(APPEND CMAKE_MODULE_PATH ${LAMMPS_SOURCE_DIR}/../examples/plugins)
include(CheckIncludeFileCXX)
include(LAMMPSInterfaceCXX)

add_library(qtftstyles MODULE plugin.cpp fix_bond_break_random.cpp bond_harmonic_anchor.cpp)
target_link_libraries(qtftstyles PRIVATE lammps)
set_target_properties(qtftstyles PROPERTIES PREFIX "" SUFFIX ".so" LINK_FLAGS "-rdynamic"
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
