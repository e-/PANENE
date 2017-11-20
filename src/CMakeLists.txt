# we default on a shared library.
if(BUILD_STATIC)
  set(PANENE_LIB_TYPE STATIC)
  MESSAGE( STATUS "Building a static library." )
else()
  MESSAGE( STATUS "Building a dynamic library (default)." )
  set(PANENE_LIB_TYPE SHARED)
endif()

MESSAGE( STATUS "PANENE_LIB_TYPE: " ${PANENE_LIB_TYPE})

set(PANENE_SRC
    panene.cpp roaring.c)

add_library(${PANENE_LIB_NAME} ${PANENE_LIB_TYPE} ${PANENE_SRC})
install(TARGETS ${PANENE_LIB_NAME} DESTINATION lib)

if(NOT MSVC)
set_target_properties(${PANENE_LIB_NAME} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
MESSAGE( STATUS "Library output directory (does not apply to Visual Studio): " ${CMAKE_BINARY_DIR})
endif()

if(MSVC AND (PANENE_LIB_TYPE STREQUAL "SHARED"))
  if (CMAKE_VERSION VERSION_LESS 3.4)
    MESSAGE( STATUS "To build  a Windows DLL using Visual Studio, you may need cmake 3.4 or better." )
  endif()
  MESSAGE( STATUS "Building a Windows DLL using Visual Studio, exporting all symbols automatically." )
 set_target_properties(${PANENE_LIB_NAME}
    PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1)
endif()

