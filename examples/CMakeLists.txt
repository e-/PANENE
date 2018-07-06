add_executable(simple_kdtree simple_kdtree.cpp)
target_link_libraries(simple_kdtree ${PANENE_LIB_NAME})

add_subdirectory(tsne)