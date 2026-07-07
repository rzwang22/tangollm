#include "analytical/analytical_pim.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace llm_system {
namespace analytical {

namespace {

constexpr const char* kH100SelectiveKV = "h100_selective_kv";
constexpr const char* kPIMNoLocalCombine = "pim_selective_kv_no_local_combine";
constexpr const char* kPIMLocalCombine = "pim_selective_kv_local_combine";

struct TopologyConfig {
  int num_stacks = 1;
  int channels_per_stack = 8;
  int pseudo_channels_per_channel = 2;
  int banks_per_pseudo_channel = 16;
  int pe_per_bank = 1;

  int total_pseudo_channels() const {
    return num_stacks * channels_per_stack * pseudo_channels_per_channel;
  }

  int total_banks() const {
    return total_pseudo_channels() * banks_per_pseudo_channel;
  }

  int bank_to_pseudo_channel(int bank) const {
    return bank / banks_per_pseudo_channel;
  }
};

struct PEConfig {
  double q8k8_group_cycles = 1.0;
  double p8v8_group_cycles = 1.0;
  double q8k2_lut_group_cycles = 1.0;
  double p8v2_lut_group_cycles = 1.0;
  double vadd_group_cycles = 1.0;
  double scale_group_cycles = 1.0;
  double reducer_group_cycles = 1.0;
  double softmax_group_cycles = 1.0;
  double scheduling_overhead_per_tile_cycles = 16.0;
  double h100_group_cycles = 1.0;
  double h100_bytes_per_cycle = 4096.0;
  double clock_ns = 1.0;
};

struct PrecisionConfig {
  double q8_bytes = 1.0;
  double k8_bytes = 1.0;
  double p8_bytes = 1.0;
  double v8_bytes = 1.0;
  double fp16_bytes = 2.0;
  double score_bytes = 2.0;
  double partial_msg_bytes = 2.0;
};

struct TileConfig {
  int channel_group = 16;
  int head_tile = 1;
  std::vector<int> memory_token_tiles{1, 4, 8};
};

struct ModelConfig {
  int memory_tokens = 128;
  int hidden_dim = 4096;
  int gnn_heads = 32;
  int head_dim = 128;
  int suffix_layers = 6;
  int text_len = 256;
  int kv_heads = 8;
  int kv_bits = 2;
  bool include_scale_metadata = false;
  int scale_group_channels = 16;
  int scale_bytes = 2;
};

struct WorkloadConfig {
  std::string mode = "query_sampled_subgraph";
  int num_queries = 16;
  int hop = 1;
  int fanout = 16;
  int seed = 777;
  double high_degree_top_percent = 0.05;
};

struct OutputConfig {
  std::string per_query_csv = "../data/analytical_pim_per_query.csv";
  std::string aggregate_csv = "../data/analytical_pim_aggregate.csv";
};

struct Edge {
  int src = 0;
  int dst = 0;
  int id = 0;
  int bank = -1;
  int pseudo_channel = 0;
};

struct FullGraph {
  int num_nodes = 0;
  std::vector<Edge> edges;
  std::vector<std::vector<int>> out_edges;
  std::vector<std::vector<int>> in_edges;
  std::vector<int> node_degree;
  std::vector<int> node_to_bank;
  std::vector<bool> high_degree_mask;
};

struct QuerySample {
  int query_id = 0;
  int target_node = 0;
  std::vector<int> sampled_nodes;
  std::vector<int> sampled_edges;
  std::vector<bool> selected_kv_mask;
};

struct SimulationConfig {
  TopologyConfig topology;
  PEConfig pe;
  PrecisionConfig precision;
  TileConfig tile;
  ModelConfig model;
  WorkloadConfig workload;
  OutputConfig output;
  FullGraph graph;
  std::vector<std::string> baselines{kH100SelectiveKV, kPIMNoLocalCombine,
                                     kPIMLocalCombine};
};

struct QueryResult {
  int query_id = 0;
  int target_node = 0;
  std::string baseline;
  int memory_token_tile = 1;
  int num_token_tiles = 0;
  int sampled_node_count = 0;
  int sampled_edge_count = 0;
  int selected_kv_count = 0;
  double selected_kv_ratio_vs_sampled_nodes = 0.0;
  double selected_kv_ratio_vs_full_graph = 0.0;
  double selected_kv_bytes = 0.0;
  double q_broadcast_bytes = 0.0;
  double score_traffic_bytes = 0.0;
  double p_return_traffic_bytes = 0.0;
  double message_reduce_traffic_bytes = 0.0;
  double local_combine_buffer_max_bytes = 0.0;
  double pc_reducer_buffer_max_bytes = 0.0;
  double gnn_score_cycles = 0.0;
  double gnn_message_cycles = 0.0;
  double cached_kv_cycles = 0.0;
  double reducer_cycles = 0.0;
  double scheduling_cycles = 0.0;
  double total_cycles = 0.0;
  double latency_ns = 0.0;
  double near_bank_pe_utilization = 0.0;
  double reducer_utilization = 0.0;
  int active_banks = 0;
  int active_pseudo_channels = 0;
};

template <typename T>
T ReadScalar(const YAML::Node& node, const std::string& key, T default_value) {
  if (!node || !node[key]) {
    return default_value;
  }
  return node[key].as<T>();
}

std::vector<int> ReadIntVector(const YAML::Node& node) {
  std::vector<int> values;
  if (!node || !node.IsSequence()) {
    return values;
  }
  for (const auto& item : node) {
    values.push_back(item.as<int>());
  }
  return values;
}

std::vector<std::string> ReadStringVector(const YAML::Node& node) {
  std::vector<std::string> values;
  if (!node || !node.IsSequence()) {
    return values;
  }
  for (const auto& item : node) {
    values.push_back(item.as<std::string>());
  }
  return values;
}

std::vector<bool> ReadBoolVector(const YAML::Node& node) {
  std::vector<bool> values;
  if (!node || !node.IsSequence()) {
    return values;
  }
  for (const auto& item : node) {
    values.push_back(item.as<bool>());
  }
  return values;
}

int PositiveModulo(int value, int modulo) {
  int result = value % modulo;
  return result < 0 ? result + modulo : result;
}

int CeilDiv(int lhs, int rhs) {
  return (lhs + rhs - 1) / rhs;
}

int DeterministicHash(int value, int seed, int modulo) {
  const int64_t mixed = static_cast<int64_t>(value) * 1103515245LL + seed;
  return static_cast<int>(PositiveModulo(static_cast<int>(mixed % modulo), modulo));
}

void ValidateTopology(const TopologyConfig& topology) {
  if (topology.num_stacks <= 0 || topology.channels_per_stack <= 0 ||
      topology.pseudo_channels_per_channel <= 0 ||
      topology.banks_per_pseudo_channel <= 0 || topology.pe_per_bank <= 0) {
    throw std::runtime_error("Invalid HBM3 logical topology");
  }
}

int GroupsPerHead(const SimulationConfig& config) {
  return CeilDiv(config.model.head_dim, config.tile.channel_group);
}

int KVGroupsPerHead(const ModelConfig& model) {
  return CeilDiv(model.head_dim, model.scale_group_channels);
}

double TextKVPayloadBytesPerItem(const ModelConfig& model) {
  return 1.0 * model.suffix_layers * 2 * model.text_len * model.kv_heads *
         model.head_dim * model.kv_bits / 8.0;
}

double ScaleMetadataBytesPerItem(const ModelConfig& model) {
  if (!model.include_scale_metadata) {
    return 0.0;
  }
  return 1.0 * model.suffix_layers * 2 * model.text_len * model.kv_heads *
         KVGroupsPerHead(model) * model.scale_bytes;
}

double TextKVBytesPerItem(const ModelConfig& model) {
  return TextKVPayloadBytesPerItem(model) + ScaleMetadataBytesPerItem(model);
}

void PrintSanityCheck(const SimulationConfig& config) {
  const int groups_per_head = GroupsPerHead(config);
  std::cout << "Analytical model sanity check\n";
  std::cout << "memory_tokens=" << config.model.memory_tokens
            << ", hidden_dim=" << config.model.hidden_dim
            << ", gnn_heads=" << config.model.gnn_heads
            << ", head_dim=" << config.model.head_dim
            << ", channel_group=" << config.tile.channel_group
            << ", groups_per_head=" << groups_per_head << "\n";
  std::cout << "suffix_layers=" << config.model.suffix_layers
            << ", text_len=" << config.model.text_len
            << ", kv_heads=" << config.model.kv_heads
            << ", kv_bits=" << config.model.kv_bits << "\n";
  std::cout << "text_kv_payload_bytes_per_item="
            << std::fixed << std::setprecision(0)
            << TextKVPayloadBytesPerItem(config.model)
            << ", include_scale_metadata="
            << (config.model.include_scale_metadata ? "true" : "false")
            << ", scale_metadata_bytes_per_item="
            << ScaleMetadataBytesPerItem(config.model)
            << ", total_kv_bytes_per_item="
            << TextKVBytesPerItem(config.model) << "\n";
}

FullGraph BuildSyntheticGraph(const YAML::Node& graph_node,
                              const TopologyConfig& topology,
                              const WorkloadConfig& workload) {
  FullGraph graph;
  graph.num_nodes = ReadScalar<int>(graph_node, "num_nodes", 2708);
  const int num_edges = ReadScalar<int>(graph_node, "num_edges", 10556);
  const int seed = ReadScalar<int>(graph_node, "seed", workload.seed);
  if (graph.num_nodes <= 0 || num_edges < 0) {
    throw std::runtime_error("query_graph synthetic size is invalid");
  }

  graph.node_to_bank = ReadIntVector(graph_node["node_to_bank"]);
  if (graph.node_to_bank.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.node_to_bank.resize(graph.num_nodes);
    for (int node = 0; node < graph.num_nodes; ++node) {
      graph.node_to_bank[node] =
          PositiveModulo(node * 131 + seed, topology.total_banks());
    }
  }

  graph.edges.reserve(num_edges);
  for (int edge_id = 0; edge_id < num_edges; ++edge_id) {
    int src = DeterministicHash(edge_id, seed, graph.num_nodes);
    int dst = DeterministicHash(edge_id * 17 + 3, seed + 97, graph.num_nodes);
    if (dst == src) {
      dst = (dst + 1) % graph.num_nodes;
    }
    const int bank = graph.node_to_bank[src];
    graph.edges.push_back(
        Edge{src, dst, edge_id, bank, topology.bank_to_pseudo_channel(bank)});
  }

  return graph;
}

FullGraph ReadGraphFromSchema(const YAML::Node& graph_node,
                              const TopologyConfig& topology) {
  FullGraph graph;
  graph.num_nodes = ReadScalar<int>(graph_node, "num_nodes", 0);
  if (graph.num_nodes <= 0) {
    throw std::runtime_error("query_graph.num_nodes must be positive");
  }

  graph.node_to_bank = ReadIntVector(graph_node["node_to_bank"]);
  if (graph.node_to_bank.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.node_to_bank.resize(graph.num_nodes);
    for (int node = 0; node < graph.num_nodes; ++node) {
      graph.node_to_bank[node] = node % topology.total_banks();
    }
  }

  const auto edge_to_bank = ReadIntVector(graph_node["edge_to_bank"]);
  const auto edge_to_pc = ReadIntVector(graph_node["edge_to_pseudo_channel"]);
  const YAML::Node edge_list = graph_node["edge_list"];
  if (!edge_list || !edge_list.IsSequence()) {
    throw std::runtime_error("query_graph.edge_list must be a sequence");
  }

  int edge_idx = 0;
  for (const auto& item : edge_list) {
    if (!item.IsSequence() || item.size() < 2) {
      throw std::runtime_error("Each edge must be [src, dst] or [src, dst, id]");
    }
    Edge edge;
    edge.src = item[0].as<int>();
    edge.dst = item[1].as<int>();
    edge.id = item.size() >= 3 ? item[2].as<int>() : edge_idx;
    if (edge.src < 0 || edge.src >= graph.num_nodes || edge.dst < 0 ||
        edge.dst >= graph.num_nodes) {
      throw std::runtime_error("edge_list contains node id outside num_nodes");
    }
    if (edge_idx < static_cast<int>(edge_to_bank.size())) {
      edge.bank = edge_to_bank[edge_idx];
    } else if (edge_idx < static_cast<int>(edge_to_pc.size())) {
      edge.bank = edge_to_pc[edge_idx] * topology.banks_per_pseudo_channel;
    } else {
      edge.bank = graph.node_to_bank[edge.src];
    }
    edge.bank = PositiveModulo(edge.bank, topology.total_banks());
    edge.pseudo_channel = topology.bank_to_pseudo_channel(edge.bank);
    graph.edges.push_back(edge);
    edge_idx++;
  }

  graph.node_degree = ReadIntVector(graph_node["node_degree"]);
  graph.high_degree_mask = ReadBoolVector(graph_node["high_degree_mask"]);
  if (graph.high_degree_mask.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.high_degree_mask.assign(graph.num_nodes, false);
    for (int node : ReadIntVector(graph_node["high_degree_nodes"])) {
      if (node >= 0 && node < graph.num_nodes) {
        graph.high_degree_mask[node] = true;
      }
    }
  }
  return graph;
}

void FinalizeGraph(FullGraph& graph, const TopologyConfig& topology,
                   const WorkloadConfig& workload) {
  graph.out_edges.assign(graph.num_nodes, {});
  graph.in_edges.assign(graph.num_nodes, {});
  graph.node_degree.assign(graph.num_nodes, 0);

  for (int edge_idx = 0; edge_idx < static_cast<int>(graph.edges.size());
       ++edge_idx) {
    Edge& edge = graph.edges[edge_idx];
    edge.bank = PositiveModulo(edge.bank, topology.total_banks());
    edge.pseudo_channel = topology.bank_to_pseudo_channel(edge.bank);
    graph.out_edges[edge.src].push_back(edge_idx);
    graph.in_edges[edge.dst].push_back(edge_idx);
    graph.node_degree[edge.src]++;
    graph.node_degree[edge.dst]++;
  }

  if (graph.high_degree_mask.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.high_degree_mask.assign(graph.num_nodes, false);
    int top_k = static_cast<int>(
        std::ceil(workload.high_degree_top_percent * graph.num_nodes));
    top_k = std::max(0, std::min(top_k, graph.num_nodes));
    std::vector<int> nodes(graph.num_nodes);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::sort(nodes.begin(), nodes.end(), [&](int lhs, int rhs) {
      return graph.node_degree[lhs] > graph.node_degree[rhs];
    });
    for (int idx = 0; idx < top_k; ++idx) {
      graph.high_degree_mask[nodes[idx]] = true;
    }
  }
}

FullGraph LoadGraph(const YAML::Node& root, const TopologyConfig& topology,
                    const WorkloadConfig& workload) {
  const YAML::Node graph_node = root["query_graph"];
  const std::string source =
      ReadScalar<std::string>(graph_node, "source", "synthetic");

  FullGraph graph;
  if (source == "schema") {
    graph = ReadGraphFromSchema(graph_node, topology);
  } else if (source == "file") {
    const std::string path = ReadScalar<std::string>(graph_node, "file", "");
    if (path.empty()) {
      throw std::runtime_error("query_graph.file is required for source=file");
    }
    graph = ReadGraphFromSchema(YAML::LoadFile(path), topology);
  } else {
    graph = BuildSyntheticGraph(graph_node, topology, workload);
  }
  FinalizeGraph(graph, topology, workload);
  return graph;
}

SimulationConfig LoadConfig(const std::string& config_path) {
  YAML::Node root = YAML::LoadFile(config_path);
  SimulationConfig config;

  const YAML::Node hbm = root["hbm3"];
  config.topology.num_stacks = ReadScalar<int>(hbm, "num_stacks", 1);
  config.topology.channels_per_stack =
      ReadScalar<int>(hbm, "channels_per_stack", 8);
  config.topology.pseudo_channels_per_channel =
      ReadScalar<int>(hbm, "pseudo_channels_per_channel", 2);
  config.topology.banks_per_pseudo_channel =
      ReadScalar<int>(hbm, "banks_per_pseudo_channel", 16);
  config.topology.pe_per_bank = ReadScalar<int>(hbm, "pe_per_bank", 1);
  ValidateTopology(config.topology);

  const YAML::Node pe = root["near_bank_pe"];
  config.pe.q8k8_group_cycles =
      ReadScalar<double>(pe, "q8k8_group_cycles", 1.0);
  config.pe.p8v8_group_cycles =
      ReadScalar<double>(pe, "p8v8_group_cycles", 1.0);
  config.pe.q8k2_lut_group_cycles =
      ReadScalar<double>(pe, "q8k2_lut_group_cycles", 1.0);
  config.pe.p8v2_lut_group_cycles =
      ReadScalar<double>(pe, "p8v2_lut_group_cycles", 1.0);
  config.pe.vadd_group_cycles =
      ReadScalar<double>(pe, "vadd_group_cycles", 1.0);
  config.pe.scale_group_cycles =
      ReadScalar<double>(pe, "scale_group_cycles", 1.0);
  config.pe.reducer_group_cycles =
      ReadScalar<double>(pe, "reducer_group_cycles", 1.0);
  config.pe.softmax_group_cycles =
      ReadScalar<double>(pe, "softmax_group_cycles", 1.0);
  config.pe.scheduling_overhead_per_tile_cycles =
      ReadScalar<double>(pe, "scheduling_overhead_per_tile_cycles", 16.0);
  config.pe.h100_group_cycles =
      ReadScalar<double>(pe, "h100_group_cycles", 1.0);
  config.pe.h100_bytes_per_cycle =
      ReadScalar<double>(pe, "h100_bytes_per_cycle", 4096.0);
  config.pe.clock_ns = ReadScalar<double>(pe, "clock_ns", 1.0);

  const YAML::Node tile = root["tile"];
  config.tile.channel_group = ReadScalar<int>(tile, "channel_group", 16);
  config.tile.head_tile = ReadScalar<int>(tile, "head_tile", 1);
  config.tile.memory_token_tiles = ReadIntVector(tile["memory_token_tiles"]);
  if (config.tile.memory_token_tiles.empty()) {
    config.tile.memory_token_tiles = {1, 4, 8};
  }

  const YAML::Node model = root["model"];
  config.model.memory_tokens = ReadScalar<int>(model, "memory_tokens", 128);
  config.model.hidden_dim = ReadScalar<int>(model, "hidden_dim", 4096);
  config.model.gnn_heads = ReadScalar<int>(model, "gnn_heads", 32);
  config.model.head_dim = ReadScalar<int>(model, "head_dim", 128);
  config.model.suffix_layers = ReadScalar<int>(model, "suffix_layers", 6);
  config.model.text_len = ReadScalar<int>(model, "text_len", 256);
  config.model.kv_heads = ReadScalar<int>(model, "kv_heads", 8);
  config.model.kv_bits = ReadScalar<int>(model, "kv_bits", 2);
  config.model.include_scale_metadata =
      ReadScalar<bool>(model, "include_scale_metadata", false);
  config.model.scale_group_channels =
      ReadScalar<int>(model, "scale_group_channels", 16);
  config.model.scale_bytes = ReadScalar<int>(model, "scale_bytes", 2);

  const YAML::Node workload = root["workload"];
  config.workload.mode =
      ReadScalar<std::string>(workload, "mode", "query_sampled_subgraph");
  config.workload.num_queries = ReadScalar<int>(workload, "num_queries", 16);
  config.workload.hop = ReadScalar<int>(workload, "hop", 1);
  config.workload.fanout = ReadScalar<int>(workload, "fanout", 16);
  config.workload.seed = ReadScalar<int>(workload, "seed", 777);
  config.workload.high_degree_top_percent =
      ReadScalar<double>(workload, "high_degree_top_percent", 0.05);

  const auto baselines = ReadStringVector(root["baselines"]);
  if (!baselines.empty()) {
    config.baselines = baselines;
  }

  const YAML::Node output = root["output"];
  config.output.per_query_csv = ReadScalar<std::string>(
      output, "per_query_csv", "../data/analytical_pim_per_query.csv");
  config.output.aggregate_csv = ReadScalar<std::string>(
      output, "aggregate_csv", "../data/analytical_pim_aggregate.csv");

  if (config.model.memory_tokens <= 0 || config.model.gnn_heads <= 0 ||
      config.model.head_dim <= 0 || config.tile.channel_group <= 0 ||
      config.workload.num_queries <= 0 || config.workload.fanout <= 0) {
    throw std::runtime_error("Invalid analytical PIM model/workload config");
  }
  config.graph = LoadGraph(root, config.topology, config.workload);
  return config;
}

void AddNode(std::vector<bool>& mask, std::vector<int>& nodes, int node) {
  if (!mask[node]) {
    mask[node] = true;
    nodes.push_back(node);
  }
}

QuerySample SampleQuery(const FullGraph& graph, const WorkloadConfig& workload,
                        int query_id) {
  QuerySample query;
  query.query_id = query_id;
  query.target_node =
      DeterministicHash(query_id * 97 + 13, workload.seed, graph.num_nodes);

  std::vector<bool> sampled_node_mask(graph.num_nodes, false);
  std::vector<bool> sampled_edge_mask(graph.edges.size(), false);
  std::vector<int> frontier;
  AddNode(sampled_node_mask, query.sampled_nodes, query.target_node);
  frontier.push_back(query.target_node);

  for (int depth = 0; depth < workload.hop; ++depth) {
    std::vector<int> next_frontier;
    for (int node : frontier) {
      std::vector<int> incident_edges = graph.out_edges[node];
      incident_edges.insert(incident_edges.end(), graph.in_edges[node].begin(),
                            graph.in_edges[node].end());
      if (incident_edges.empty()) {
        continue;
      }
      const int offset =
          DeterministicHash(node + query_id * 31 + depth * 17, workload.seed,
                            static_cast<int>(incident_edges.size()));
      const int take = std::min<int>(workload.fanout, incident_edges.size());
      for (int item = 0; item < take; ++item) {
        const int edge_idx =
            incident_edges[(offset + item) % incident_edges.size()];
        if (!sampled_edge_mask[edge_idx]) {
          sampled_edge_mask[edge_idx] = true;
          query.sampled_edges.push_back(edge_idx);
        }
        const Edge& edge = graph.edges[edge_idx];
        const int neighbor = edge.src == node ? edge.dst : edge.src;
        if (!sampled_node_mask[neighbor]) {
          AddNode(sampled_node_mask, query.sampled_nodes, neighbor);
          next_frontier.push_back(neighbor);
        }
      }
    }
    frontier.swap(next_frontier);
    if (frontier.empty()) {
      break;
    }
  }

  query.selected_kv_mask.assign(graph.num_nodes, false);
  query.selected_kv_mask[query.target_node] = true;
  for (int edge_idx : query.sampled_edges) {
    const Edge& edge = graph.edges[edge_idx];
    if (edge.src == query.target_node) {
      query.selected_kv_mask[edge.dst] = true;
    }
    if (edge.dst == query.target_node) {
      query.selected_kv_mask[edge.src] = true;
    }
  }
  for (int node : query.sampled_nodes) {
    if (graph.high_degree_mask[node]) {
      query.selected_kv_mask[node] = true;
    }
  }
  return query;
}

std::vector<QuerySample> BuildQueries(const FullGraph& graph,
                                      const WorkloadConfig& workload) {
  if (workload.mode != "query_sampled_subgraph") {
    throw std::runtime_error("Only workload.mode=query_sampled_subgraph is supported");
  }
  std::vector<QuerySample> queries;
  queries.reserve(workload.num_queries);
  for (int query_id = 0; query_id < workload.num_queries; ++query_id) {
    queries.push_back(SampleQuery(graph, workload, query_id));
  }
  return queries;
}

int CountSelectedKV(const QuerySample& query) {
  return std::count(query.selected_kv_mask.begin(), query.selected_kv_mask.end(),
                    true);
}

std::set<int> SelectedBanks(const FullGraph& graph, const QuerySample& query) {
  std::set<int> banks;
  for (int node : query.sampled_nodes) {
    if (query.selected_kv_mask[node]) {
      banks.insert(graph.node_to_bank[node]);
    }
  }
  return banks;
}

QueryResult SimulateQuery(const SimulationConfig& config,
                          const QuerySample& query,
                          const std::string& baseline,
                          int memory_token_tile) {
  const auto& graph = config.graph;
  const auto& topology = config.topology;
  const auto& model = config.model;
  const auto& precision = config.precision;
  const auto& pe = config.pe;
  const int total_banks = topology.total_banks();
  const int total_pcs = topology.total_pseudo_channels();
  const int groups_per_head = GroupsPerHead(config);
  const int num_token_tiles = CeilDiv(model.memory_tokens, memory_token_tile);
  const int edge_count = static_cast<int>(query.sampled_edges.size());
  const int selected_kv_count = CountSelectedKV(query);

  std::vector<int> edge_count_by_bank(total_banks, 0);
  std::vector<std::set<int>> dst_by_bank(total_banks);
  std::vector<std::set<int>> dst_by_pc(total_pcs);
  std::set<int> active_banks;
  std::set<int> active_pcs;
  std::set<int> active_dst;

  for (int edge_idx : query.sampled_edges) {
    const Edge& edge = graph.edges[edge_idx];
    edge_count_by_bank[edge.bank]++;
    dst_by_bank[edge.bank].insert(edge.dst);
    dst_by_pc[edge.pseudo_channel].insert(edge.dst);
    active_banks.insert(edge.bank);
    active_pcs.insert(edge.pseudo_channel);
    active_dst.insert(edge.dst);
  }
  for (int bank : SelectedBanks(graph, query)) {
    active_banks.insert(bank);
    active_pcs.insert(topology.bank_to_pseudo_channel(bank));
  }

  std::vector<int> selected_by_bank(total_banks, 0);
  for (int node : query.sampled_nodes) {
    if (query.selected_kv_mask[node]) {
      selected_by_bank[graph.node_to_bank[node]]++;
    }
  }

  const double score_groups =
      1.0 * edge_count * model.memory_tokens * model.gnn_heads *
      groups_per_head;
  const double message_groups = score_groups;
  const double cached_kv_groups =
      1.0 * selected_kv_count * model.suffix_layers * model.text_len *
      model.gnn_heads * groups_per_head;

  QueryResult result;
  result.query_id = query.query_id;
  result.target_node = query.target_node;
  result.baseline = baseline;
  result.memory_token_tile = memory_token_tile;
  result.num_token_tiles = num_token_tiles;
  result.sampled_node_count = static_cast<int>(query.sampled_nodes.size());
  result.sampled_edge_count = edge_count;
  result.selected_kv_count = selected_kv_count;
  result.selected_kv_ratio_vs_sampled_nodes =
      query.sampled_nodes.empty()
          ? 0.0
          : 1.0 * selected_kv_count / query.sampled_nodes.size();
  result.selected_kv_ratio_vs_full_graph =
      graph.num_nodes == 0 ? 0.0 : 1.0 * selected_kv_count / graph.num_nodes;
  result.selected_kv_bytes = selected_kv_count * TextKVBytesPerItem(model);
  result.active_banks = static_cast<int>(active_banks.size());
  result.active_pseudo_channels = static_cast<int>(active_pcs.size());

  if (baseline == kH100SelectiveKV) {
    const double compute_cycles =
        (score_groups + message_groups + cached_kv_groups) *
        pe.h100_group_cycles;
    const double memory_cycles =
        result.selected_kv_bytes / std::max(1.0, pe.h100_bytes_per_cycle);
    result.gnn_score_cycles = score_groups * pe.h100_group_cycles;
    result.gnn_message_cycles = message_groups * pe.h100_group_cycles;
    result.cached_kv_cycles = cached_kv_groups * pe.h100_group_cycles;
    result.total_cycles = std::max(compute_cycles, memory_cycles);
    result.latency_ns = result.total_cycles * pe.clock_ns;
    return result;
  }

  double score_cycles = 0.0;
  double message_cycles = 0.0;
  double kv_cycles = 0.0;
  double pe_work_cycles = 0.0;
  for (int bank = 0; bank < total_banks; ++bank) {
    const double bank_score_groups =
        1.0 * edge_count_by_bank[bank] * model.memory_tokens *
        model.gnn_heads * groups_per_head;
    const double bank_message_groups = bank_score_groups;
    const double bank_kv_groups =
        1.0 * selected_by_bank[bank] * model.suffix_layers * model.text_len *
        model.gnn_heads * groups_per_head;
    const double bank_score_cycles =
        bank_score_groups * (pe.q8k8_group_cycles + pe.scale_group_cycles) /
        topology.pe_per_bank;
    const double bank_message_cycles =
        bank_message_groups *
        (pe.p8v8_group_cycles + pe.scale_group_cycles +
         (baseline == kPIMLocalCombine ? pe.vadd_group_cycles : 0.0)) /
        topology.pe_per_bank;
    const double bank_kv_cycles =
        bank_kv_groups *
        (pe.q8k2_lut_group_cycles + pe.p8v2_lut_group_cycles +
         2.0 * pe.scale_group_cycles) /
        topology.pe_per_bank;
    score_cycles = std::max(score_cycles, bank_score_cycles);
    message_cycles = std::max(message_cycles, bank_message_cycles);
    kv_cycles = std::max(kv_cycles, bank_kv_cycles);
    pe_work_cycles += bank_score_cycles + bank_message_cycles + bank_kv_cycles;
  }

  result.q_broadcast_bytes =
      1.0 * num_token_tiles * active_banks.size() * model.gnn_heads *
      model.head_dim * precision.q8_bytes;
  result.score_traffic_bytes =
      1.0 * edge_count * model.memory_tokens * model.gnn_heads *
      precision.score_bytes;
  result.p_return_traffic_bytes =
      1.0 * edge_count * model.memory_tokens * model.gnn_heads *
      precision.p8_bytes;

  const double buffer_per_active_group =
      1.0 * memory_token_tile * config.tile.head_tile *
      config.tile.channel_group * precision.partial_msg_bytes;
  double reducer_groups = 0.0;
  double reducer_work = 0.0;

  if (baseline == kPIMNoLocalCombine) {
    const double message_values =
        1.0 * edge_count * model.memory_tokens * model.gnn_heads *
        model.head_dim;
    result.message_reduce_traffic_bytes =
        message_values * precision.partial_msg_bytes;
    reducer_groups = message_values / config.tile.channel_group;
    result.reducer_cycles = reducer_groups * pe.reducer_group_cycles /
                            std::max<int>(1, active_pcs.size());
    reducer_work = reducer_groups * pe.reducer_group_cycles;
  } else if (baseline == kPIMLocalCombine) {
    double bank_dst_pairs = 0.0;
    double pc_dst_pairs = 0.0;
    for (int bank = 0; bank < total_banks; ++bank) {
      const double dst_count = static_cast<double>(dst_by_bank[bank].size());
      bank_dst_pairs += dst_count;
      result.local_combine_buffer_max_bytes =
          std::max(result.local_combine_buffer_max_bytes,
                   dst_count * model.gnn_heads * groups_per_head *
                       buffer_per_active_group);
    }
    for (int pc = 0; pc < total_pcs; ++pc) {
      const double dst_count = static_cast<double>(dst_by_pc[pc].size());
      pc_dst_pairs += dst_count;
      result.pc_reducer_buffer_max_bytes =
          std::max(result.pc_reducer_buffer_max_bytes,
                   dst_count * model.gnn_heads * groups_per_head *
                       buffer_per_active_group);
    }
    const double bank_to_pc_groups =
        bank_dst_pairs * model.memory_tokens * model.gnn_heads *
        groups_per_head;
    const double pc_to_global_groups =
        pc_dst_pairs * model.memory_tokens * model.gnn_heads *
        groups_per_head;
    result.message_reduce_traffic_bytes =
        (bank_to_pc_groups + pc_to_global_groups) *
        config.tile.channel_group * precision.partial_msg_bytes;
    reducer_groups = bank_to_pc_groups + pc_to_global_groups;
    result.reducer_cycles =
        bank_to_pc_groups * pe.reducer_group_cycles /
            std::max<int>(1, active_pcs.size()) +
        pc_to_global_groups * pe.reducer_group_cycles;
    reducer_work = reducer_groups * pe.reducer_group_cycles;
  } else {
    throw std::runtime_error("Unknown analytical baseline: " + baseline);
  }

  result.gnn_score_cycles = score_cycles;
  result.gnn_message_cycles = message_cycles;
  result.cached_kv_cycles = kv_cycles;
  result.scheduling_cycles =
      num_token_tiles * pe.scheduling_overhead_per_tile_cycles;
  const double near_bank_cycles = score_cycles + message_cycles + kv_cycles;
  result.total_cycles =
      near_bank_cycles + result.reducer_cycles + result.scheduling_cycles;
  result.latency_ns = result.total_cycles * pe.clock_ns;
  result.near_bank_pe_utilization =
      near_bank_cycles == 0.0
          ? 0.0
          : pe_work_cycles /
                (near_bank_cycles * total_banks * topology.pe_per_bank);
  result.reducer_utilization =
      result.reducer_cycles == 0.0
          ? 0.0
          : reducer_work /
                (result.reducer_cycles * std::max<int>(1, active_pcs.size() + 1));
  return result;
}

std::filesystem::path PrepareCSV(const std::string& path) {
  std::filesystem::path csv_path(path);
  if (csv_path.has_parent_path()) {
    std::filesystem::create_directories(csv_path.parent_path());
  }
  return csv_path;
}

void WritePerQueryCSV(const std::string& path,
                      const std::vector<QueryResult>& results) {
  PrepareCSV(path);
  std::ofstream csv(path);
  if (!csv.is_open()) {
    throw std::runtime_error("Cannot open per-query CSV: " + path);
  }
  csv << "query_id,target_node,baseline,memory_token_tile,num_token_tiles,"
         "sampled_node_count,sampled_edge_count,selected_kv_count,"
         "selected_kv_ratio_vs_sampled_nodes,selected_kv_ratio_vs_full_graph,"
         "selected_kv_bytes,q_broadcast_bytes,score_traffic_bytes,"
         "p_return_traffic_bytes,message_reduce_traffic_bytes,"
         "local_combine_buffer_max_bytes,pc_reducer_buffer_max_bytes,"
         "gnn_score_cycles,gnn_message_cycles,cached_kv_cycles,"
         "reducer_cycles,scheduling_cycles,total_cycles,latency_ns,"
         "near_bank_pe_utilization,reducer_utilization,active_banks,"
         "active_pseudo_channels\n";
  csv << std::fixed << std::setprecision(6);
  for (const auto& r : results) {
    csv << r.query_id << "," << r.target_node << "," << r.baseline << ","
        << r.memory_token_tile << "," << r.num_token_tiles << ","
        << r.sampled_node_count << "," << r.sampled_edge_count << ","
        << r.selected_kv_count << ","
        << r.selected_kv_ratio_vs_sampled_nodes << ","
        << r.selected_kv_ratio_vs_full_graph << "," << r.selected_kv_bytes
        << "," << r.q_broadcast_bytes << "," << r.score_traffic_bytes << ","
        << r.p_return_traffic_bytes << "," << r.message_reduce_traffic_bytes
        << "," << r.local_combine_buffer_max_bytes << ","
        << r.pc_reducer_buffer_max_bytes << "," << r.gnn_score_cycles << ","
        << r.gnn_message_cycles << "," << r.cached_kv_cycles << ","
        << r.reducer_cycles << "," << r.scheduling_cycles << ","
        << r.total_cycles << "," << r.latency_ns << ","
        << r.near_bank_pe_utilization << "," << r.reducer_utilization << ","
        << r.active_banks << "," << r.active_pseudo_channels << "\n";
  }
}

double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double Percentile(std::vector<double> values, double percentile) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = percentile * (values.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(rank));
  const size_t hi = static_cast<size_t>(std::ceil(rank));
  if (lo == hi) {
    return values[lo];
  }
  const double weight = rank - lo;
  return values[lo] * (1.0 - weight) + values[hi] * weight;
}

void WriteAggregateCSV(const std::string& path,
                       const std::vector<QueryResult>& results,
                       const std::vector<std::string>& baselines,
                       const std::vector<int>& tiles) {
  PrepareCSV(path);
  std::ofstream csv(path);
  if (!csv.is_open()) {
    throw std::runtime_error("Cannot open aggregate CSV: " + path);
  }
  csv << "baseline,memory_token_tile,num_queries,mean_latency_ns,p50_latency_ns,"
         "p95_latency_ns,selected_kv_count,q_broadcast_bytes,"
         "score_traffic_bytes,p_return_traffic_bytes,"
         "message_reduce_traffic_bytes,local_combine_buffer_max_bytes,"
         "pc_reducer_buffer_max_bytes,active_banks,active_pseudo_channels,"
         "near_bank_pe_utilization,reducer_utilization\n";
  csv << std::fixed << std::setprecision(6);
  for (const std::string& baseline : baselines) {
    for (int tile : tiles) {
      std::vector<const QueryResult*> group;
      for (const auto& r : results) {
        if (r.baseline == baseline && r.memory_token_tile == tile) {
          group.push_back(&r);
        }
      }
      if (group.empty()) {
        continue;
      }
      std::vector<double> latencies;
      std::vector<double> selected;
      std::vector<double> q_broadcast;
      std::vector<double> score_traffic;
      std::vector<double> p_return;
      std::vector<double> message_reduce;
      std::vector<double> local_buffer;
      std::vector<double> pc_buffer;
      std::vector<double> active_banks;
      std::vector<double> active_pcs;
      std::vector<double> pe_util;
      std::vector<double> reducer_util;
      for (const auto* r : group) {
        latencies.push_back(r->latency_ns);
        selected.push_back(r->selected_kv_count);
        q_broadcast.push_back(r->q_broadcast_bytes);
        score_traffic.push_back(r->score_traffic_bytes);
        p_return.push_back(r->p_return_traffic_bytes);
        message_reduce.push_back(r->message_reduce_traffic_bytes);
        local_buffer.push_back(r->local_combine_buffer_max_bytes);
        pc_buffer.push_back(r->pc_reducer_buffer_max_bytes);
        active_banks.push_back(r->active_banks);
        active_pcs.push_back(r->active_pseudo_channels);
        pe_util.push_back(r->near_bank_pe_utilization);
        reducer_util.push_back(r->reducer_utilization);
      }
      csv << baseline << "," << tile << "," << group.size() << ","
          << Mean(latencies) << "," << Percentile(latencies, 0.50) << ","
          << Percentile(latencies, 0.95) << "," << Mean(selected) << ","
          << Mean(q_broadcast) << "," << Mean(score_traffic) << ","
          << Mean(p_return) << "," << Mean(message_reduce) << ","
          << Mean(local_buffer) << "," << Mean(pc_buffer) << ","
          << Mean(active_banks) << "," << Mean(active_pcs) << ","
          << Mean(pe_util) << "," << Mean(reducer_util) << "\n";
    }
  }
}

void PrintSummary(const SimulationConfig& config,
                  const std::vector<QueryResult>& results) {
  std::cout << "Analytical PIM Patch 1 calibration run\n";
  std::cout << "full_graph_nodes=" << config.graph.num_nodes
            << ", full_graph_edges=" << config.graph.edges.size()
            << ", num_queries=" << config.workload.num_queries
            << ", hop=" << config.workload.hop
            << ", fanout=" << config.workload.fanout << "\n";
  for (const auto& baseline : config.baselines) {
    for (int tile : config.tile.memory_token_tiles) {
      std::vector<double> latencies;
      std::vector<double> selected_ratios;
      for (const auto& r : results) {
        if (r.baseline == baseline && r.memory_token_tile == tile) {
          latencies.push_back(r.latency_ns);
          selected_ratios.push_back(r.selected_kv_ratio_vs_sampled_nodes);
        }
      }
      if (latencies.empty()) {
        continue;
      }
      std::cout << baseline << " T=" << tile
                << " mean_latency_ns=" << std::fixed << std::setprecision(2)
                << Mean(latencies)
                << " p95_latency_ns=" << Percentile(latencies, 0.95)
                << " selected_ratio_vs_sampled="
                << std::setprecision(4) << Mean(selected_ratios) << "\n";
    }
  }
  std::cout << "per_query_csv=" << config.output.per_query_csv << "\n";
  std::cout << "aggregate_csv=" << config.output.aggregate_csv << "\n";
}

}  // namespace

int RunAnalyticalPIM(const std::string& config_path) {
  try {
    SimulationConfig config = LoadConfig(config_path);
    PrintSanityCheck(config);

    const std::vector<QuerySample> queries =
        BuildQueries(config.graph, config.workload);
    std::vector<QueryResult> results;
    for (const auto& query : queries) {
      for (const auto& baseline : config.baselines) {
        for (int tile : config.tile.memory_token_tiles) {
          if (tile <= 0) {
            throw std::runtime_error("memory_token_tiles must be positive");
          }
          results.push_back(SimulateQuery(config, query, baseline, tile));
        }
      }
    }

    WritePerQueryCSV(config.output.per_query_csv, results);
    WriteAggregateCSV(config.output.aggregate_csv, results, config.baselines,
                      config.tile.memory_token_tiles);
    PrintSummary(config, results);
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}

}  // namespace analytical
}  // namespace llm_system
