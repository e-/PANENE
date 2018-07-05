add_executable(responsive_tsne responsive_tsne.cpp responsive_tsne_main.cpp ../lib/sptree.cpp ../lib/config.cpp)
target_link_libraries(responsive_tsne LINK_PUBLIC ${PANENE_LIB_NAME})

add_custom_command(
  TARGET responsive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/../data.py
  ${CMAKE_CURRENT_BINARY_DIR}/../data.py
)

add_custom_command(
  TARGET responsive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/../mnist.sh
  ${CMAKE_CURRENT_BINARY_DIR}/../mnist.sh
)

add_custom_command(
  TARGET responsive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/../visualizer.html
  ${CMAKE_CURRENT_BINARY_DIR}/../visualizer.html
)

add_custom_command(
  TARGET responsive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/../compare.html
  ${CMAKE_CURRENT_BINARY_DIR}/../compare.html
)

add_custom_command(
  TARGET responsive_tsne POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../mnist
  ${CMAKE_CURRENT_BINARY_DIR}/../mnist
  )
