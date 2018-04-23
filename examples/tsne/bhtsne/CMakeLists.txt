add_executable(bhtsne tsne.cpp sptree.cpp tsne_main.cpp ../config/config.cpp)
target_link_libraries(bhtsne LINK_PUBLIC ${PANENE_LIB_NAME})