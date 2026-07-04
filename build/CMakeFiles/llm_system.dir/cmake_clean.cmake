file(REMOVE_RECURSE
  "libllm_system.pdb"
  "libllm_system.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/llm_system.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
