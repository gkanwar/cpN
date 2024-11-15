cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cpn_flow)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(NC 2)

find_package(Torch REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)
add_subdirectory(example_simple)
add_subdirectory(src)

