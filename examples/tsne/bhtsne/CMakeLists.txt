add_executable(bhtsne tsne.cpp tsne_main.cpp ../lib/sptree.cpp ../lib/config.cpp)
target_link_libraries(bhtsne LINK_PUBLIC ${PANENE_LIB_NAME})