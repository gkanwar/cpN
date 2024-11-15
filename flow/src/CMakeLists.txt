project("main")

if(NOT DEFINED NC)
  message(FATAL_ERROR "You must set NC")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main)
target_sources(main PRIVATE main.cpp args.hxx)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
