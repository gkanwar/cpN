project(example-simple)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-simple example-simple.cpp)
target_link_libraries(example-simple "${TORCH_LIBRARIES}")
