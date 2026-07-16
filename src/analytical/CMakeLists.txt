add_library(analytical_pim_model STATIC)

target_sources(
    analytical_pim_model PRIVATE
    analytical_pim.cpp
    gofa_trace.cpp
)

target_link_libraries(
    analytical_pim_model
    PUBLIC yaml-cpp
)

target_compile_features(
    analytical_pim_model
    PUBLIC cxx_std_17
)
