add_executable(run test.cpp)

target_link_libraries(
    run
    PRIVATE llm_system
)

set_target_properties(run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_executable(analytical_pim analytical_pim.cpp)

target_link_libraries(
    analytical_pim
    PRIVATE analytical_pim_model
)

set_target_properties(analytical_pim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
