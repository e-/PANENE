set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")

install( DIRECTORY pyflann DESTINATION share/panene/python )
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/setup.py DESTINATION share/panene/python )


# python instalation
if (BUILD_PYTHON_BINDINGS)
    set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pynene/__init__.py")
    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

    configure_file(${SETUP_PY_IN} ${SETUP_PY})

    add_custom_command(OUTPUT ${OUTPUT}
                       COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
                       DEPENDS ${DEPS})

    add_custom_target(target ALL DEPENDS ${OUTPUT})

    install(CODE "execute_process(
        COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py install
        WORKING_DIRECTORY \"${CMAKE_CURRENT_SOURCE_DIR}\")")
endif()
