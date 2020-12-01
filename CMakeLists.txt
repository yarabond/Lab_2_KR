cmake_minimum_required(VERSION 3.17)
project(mmm)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_COMPILER /usr/bin/mpicc)
set(CMAKE_CXX_COMPILER /usr/bin/mpic++)

include_directories(/usr/include/mpi/)

add_executable(mmm cannon.cpp)