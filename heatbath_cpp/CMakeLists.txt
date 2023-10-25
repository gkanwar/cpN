cmake_minimum_required(VERSION 3.10)
project("CPN Heatbath")

if(NOT DEFINED NC)
  message(FATAL_ERROR "You must set NC")
endif()
if(NOT DEFINED ND)
  message(FATAL_ERROR "You must set ND")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(lib)
add_subdirectory(src)
