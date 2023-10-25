add_executable(run_heatbath run_heatbath.cpp)
target_include_directories(run_heatbath PRIVATE "${PROJECT_BINARY_DIR}/lib" "${PROJECT_SOURCE_DIR}/lib")
target_link_libraries(run_heatbath cpn)

install(TARGETS run_heatbath DESTINATION bin)
