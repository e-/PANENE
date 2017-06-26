set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories ("${CMAKE_SOURCE_DIR}")
include_directories ("include")
set(tests progressive_kd_tree_index_benchmark progressive_knn_table_benchmark)

foreach (test ${tests})
  add_executable (${test} ${test}.cpp)
  target_link_libraries (${test} LINK_PUBLIC panene)
endforeach (test ${tests})
