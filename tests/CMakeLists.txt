set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories ("${CMAKE_SOURCE_DIR}")
include_directories ("include")
file(GLOB SourceFiles *.cpp)
add_executable (tests ${SourceFiles} )
target_link_libraries (tests LINK_PUBLIC panene)

add_test (NAME MyTest COMMAND tests)
