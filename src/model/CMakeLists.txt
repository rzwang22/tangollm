add_library(model OBJECT)

target_sources(
    model PRIVATE
    
    llm.cpp   
    TF_GNN.cpp
)
