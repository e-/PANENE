add_executable(tsne tsne.cpp sptree.cpp tsne_main.cpp)
target_link_libraries(tsne ${PANENE_LIB_NAME})

add_executable(progressive_tsne progressive_tsne.cpp sptree.cpp progressive_tsne_main.cpp)
target_link_libraries(progressive_tsne LINK_PUBLIC ${PANENE_LIB_NAME})

add_custom_command(
  TARGET progressive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/data.py
  ${CMAKE_CURRENT_BINARY_DIR}/data.py
)

add_custom_command(
  TARGET progressive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/mnist.sh
  ${CMAKE_CURRENT_BINARY_DIR}/mnist.sh
)

add_custom_command(
  TARGET progressive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/result
  ${CMAKE_CURRENT_BINARY_DIR}/result
  )

add_custom_command(
  TARGET progressive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/mnist
  ${CMAKE_CURRENT_BINARY_DIR}/mnist
  )
