set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories ("${CMAKE_SOURCE_DIR}")
set(benchmarks answer kd_tree_benchmark knn_table_benchmark node_split_n_test)

foreach (benchmark ${benchmarks})
  add_executable (${benchmark} ${benchmark}.cpp)
  target_link_libraries (${benchmark} LINK_PUBLIC panene)
endforeach (benchmark ${benchmarks})

add_custom_command(TARGET answer POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:answer> ../../data)
