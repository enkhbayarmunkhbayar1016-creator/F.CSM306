cmake_minimum_required(VERSION 3.15)
project(ParallelSum CXX)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenMP REQUIRED)

add_executable(parallel_sum main.cpp)
target_link_libraries(parallel_sum OpenMP::OpenMP_CXX)
