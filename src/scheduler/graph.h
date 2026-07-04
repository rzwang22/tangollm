#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

struct Graph {

    int num_nodes;
    int num_edges;

    std::vector<int> token_count;

    std::vector<std::vector<int>> adj;
};

inline std::vector<Graph> loadGraphList(
        const std::string& filename)
{
    std::ifstream fin(filename);

    if (!fin.is_open()) {
        throw std::runtime_error(
            "cannot open file: " + filename);
    }

    int graph_num;

    fin >> graph_num;

    std::vector<Graph> graphs;

    for (int g = 0; g < graph_num; g++) {

        std::string tag;
        int gid;

        fin >> tag >> gid;

        Graph graph;

        fin >> graph.num_nodes
            >> graph.num_edges;

        graph.token_count.resize(
            graph.num_nodes);

        graph.adj.resize(
            graph.num_nodes);

        for (int i = 0; i < graph.num_nodes; i++) {
            fin >> graph.token_count[i];
        }

        for (int e = 0; e < graph.num_edges; e++) {

            int u, v;

            fin >> u >> v;

            graph.adj[u].push_back(v);
        }

        graphs.push_back(std::move(graph));
    }

    return graphs;
}