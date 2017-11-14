CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
project(PANENE)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type selected, default to Release")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

set(PANENE_LIB_NAME panene)
set(PANENE_VERSION 0.0.1)

option(BUILD_STATIC "Build a static library" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(COMPILE_FOR_BENCHMARK "Compile for benchmark (e.g., profiling)" ON)

find_package ( OpenMP )
if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(include)

# Require C++11 support, prefer ISO C++ over GNU variants,
# as relying solely on ISO C++ is more portable.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (COMPILE_FOR_BENCHMARK)
  add_definitions(-DBENCHMARK)
endif()

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(benchmark)
enable_testing ()
add_subdirectory(tests)

# find python
find_package(PythonInterp)
if (NOT PYTHON_EXECUTABLE)
    set(BUILD_PYTHON_BINDINGS OFF)
endif()

if (BUILD_PYTHON_BINDINGS)
  add_subdirectory(python)
endif()

message( STATUS "CMAKE_SYSTEM_PROCESSOR: " ${CMAKE_SYSTEM_PROCESSOR})
message( STATUS "CMAKE_BUILD_TYPE: " ${CMAKE_BUILD_TYPE} )
message( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} ) # important to know which compiler is used
message( STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS} ) # important to know the flags
message( STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG} )
message( STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE} )
message( STATUS "PYTHON_EXECUTABLE: " ${PYTHON_EXECUTABLE} )
message( STATUS "Building python bindings:" ${BUILD_PYTHON_BINDINGS})
message( STATUS "Compile for benchmark: " ${COMPILE_FOR_BENCHMARK})
