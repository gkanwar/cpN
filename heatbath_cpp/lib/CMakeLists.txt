set(SOURCES cpn.cpp)
set(HEADERS cpn.h lattice.h metropolis.h util.h)
add_library(cpn ${SOURCES})
target_include_directories(cpn PRIVATE .)

configure_file(config.h.in config.h)
target_include_directories(cpn PRIVATE "${PROJECT_BINARY_DIR}/lib")

install(TARGETS cpn DESTINATION lib)
install(FILES config.h DESTINATION include)
install(FILES ${HEADERS} DESTINATION include)
