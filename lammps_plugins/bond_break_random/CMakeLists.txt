# Plugin fix bond/break/random. Needs the LAMMPS source of exactly the version of the lmp that loads it
# (headers only): cmake -S . -B build -D LAMMPS_SOURCE_DIR=<lammps>/src; see build.sh.
cmake_minimum_required(VERSION 3.20)
project(bondbreakrandom VERSION 1.0 LANGUAGES CXX)

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

add_library(bondbreakrandom MODULE plugin.cpp fix_bond_break_random.cpp)
target_link_libraries(bondbreakrandom PRIVATE lammps)
set_target_properties(bondbreakrandom PROPERTIES PREFIX "" SUFFIX ".so" LINK_FLAGS "-rdynamic"
                      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
