add_executable(density density.cpp)
target_link_libraries(density ${PANENE_LIB_NAME})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data.txt
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data.sorted.txt
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

