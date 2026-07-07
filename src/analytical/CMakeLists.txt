add_library(analytical_pim_model STATIC)

target_sources(
    analytical_pim_model PRIVATE
    analytical_pim.cpp
)

target_link_libraries(
    analytical_pim_model
    PUBLIC yaml-cpp
)
