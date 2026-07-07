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
  double clock_ns = 1.0;
};

struct PrecisionConfig {
  double q8_bytes = 1.0;
  double k8_bytes = 1.0;
  double p8_bytes = 1.0;
  double v8_bytes = 1.0;
  double k2_bytes = 0.25;
  double v2_bytes = 0.25;
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
  int num_heads = 1;
  int head_dim = 128;
};

struct Edge {
  int src = 0;
  int dst = 0;
  int id = 0;
  int bank = -1;
  int pseudo_channel = 0;
};

struct QueryGraph {
  int num_nodes = 0;
  std::vector<Edge> edges;
  std::vector<int> target_nodes;
  std::vector<int> node_degree;
  std::vector<int> node_to_bank;
  std::vector<bool> high_degree_mask;
  std::vector<bool> selected_kv_mask;
};

struct SimulationConfig {
  TopologyConfig topology;
  PEConfig pe;
  PrecisionConfig precision;
  TileConfig tile;
  ModelConfig model;
  QueryGraph graph;
  std::string output_csv = "../data/analytical_pim_tiles.csv";
};

struct TileResult {
  int memory_token_tile = 1;
  int selected_kv_count = 0;
  double selected_kv_ratio = 0.0;
  double selected_kv_bytes = 0.0;
  double kv_reduction_vs_all_nodes = 0.0;
  double kv_reduction_vs_all_edges = 0.0;
  double q_broadcast_bytes = 0.0;
  double score_traffic_bytes = 0.0;
  double p_return_traffic_bytes = 0.0;
  double message_reduce_traffic_bytes = 0.0;
  double local_combine_buffer_max_bytes = 0.0;
  double local_combine_buffer_total_bytes = 0.0;
  double pc_reducer_buffer_max_bytes = 0.0;
  double pc_reducer_buffer_total_bytes = 0.0;
  double gnn_score_cycles = 0.0;
  double gnn_message_cycles = 0.0;
  double cached_kv_cycles = 0.0;
  double reducer_cycles = 0.0;
  double total_cycles = 0.0;
  double total_latency_ns = 0.0;
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

void ValidateTopology(const TopologyConfig& topology) {
  if (topology.num_stacks <= 0 || topology.channels_per_stack <= 0 ||
      topology.pseudo_channels_per_channel <= 0 ||
      topology.banks_per_pseudo_channel <= 0 || topology.pe_per_bank <= 0) {
    throw std::runtime_error("Invalid HBM3 logical topology");
  }
}

void NormalizeGraph(QueryGraph& graph, const TopologyConfig& topology) {
  const int total_banks = topology.total_banks();
  if (graph.num_nodes <= 0) {
    throw std::runtime_error("query_graph.num_nodes must be positive");
  }

  if (graph.node_to_bank.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.node_to_bank.resize(graph.num_nodes);
    for (int node = 0; node < graph.num_nodes; ++node) {
      graph.node_to_bank[node] = node % total_banks;
    }
  }

  if (graph.node_degree.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.node_degree.assign(graph.num_nodes, 0);
    for (const auto& edge : graph.edges) {
      if (edge.src >= 0 && edge.src < graph.num_nodes) {
        graph.node_degree[edge.src]++;
      }
      if (edge.dst >= 0 && edge.dst < graph.num_nodes) {
        graph.node_degree[edge.dst]++;
      }
    }
  }

  if (graph.high_degree_mask.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.high_degree_mask.assign(graph.num_nodes, false);
  }

  if (graph.selected_kv_mask.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.selected_kv_mask.assign(graph.num_nodes, false);
    std::unordered_set<int> target_set;
    for (int node : graph.target_nodes) {
      if (node >= 0 && node < graph.num_nodes) {
        graph.selected_kv_mask[node] = true;
        target_set.insert(node);
      }
    }
    for (const auto& edge : graph.edges) {
      if (target_set.count(edge.dst) && edge.src >= 0 &&
          edge.src < graph.num_nodes) {
        graph.selected_kv_mask[edge.src] = true;
      }
      if (target_set.count(edge.src) && edge.dst >= 0 &&
          edge.dst < graph.num_nodes) {
        graph.selected_kv_mask[edge.dst] = true;
      }
    }
    for (int node = 0; node < graph.num_nodes; ++node) {
      if (graph.high_degree_mask[node]) {
        graph.selected_kv_mask[node] = true;
      }
    }
  }

  for (auto& edge : graph.edges) {
    if (edge.src < 0 || edge.src >= graph.num_nodes || edge.dst < 0 ||
        edge.dst >= graph.num_nodes) {
      throw std::runtime_error("edge_list contains node id outside num_nodes");
    }
    if (edge.bank < 0) {
      edge.bank = graph.node_to_bank[edge.src];
    }
    edge.bank = PositiveModulo(edge.bank, total_banks);
    edge.pseudo_channel = topology.bank_to_pseudo_channel(edge.bank);
  }
}

QueryGraph BuildSyntheticGraph(const YAML::Node& graph_node,
                               const TopologyConfig& topology) {
  QueryGraph graph;
  graph.num_nodes = ReadScalar<int>(graph_node, "num_nodes", 2708);
  const int num_edges = ReadScalar<int>(graph_node, "num_edges", 10556);
  const int seed = ReadScalar<int>(graph_node, "seed", 777);
  const int num_target_nodes =
      ReadScalar<int>(graph_node, "num_target_nodes", 32);
  const int num_high_degree_nodes =
      ReadScalar<int>(graph_node, "num_high_degree_nodes", 0);

  graph.target_nodes = ReadIntVector(graph_node["target_nodes"]);
  if (graph.target_nodes.empty()) {
    for (int node = 0; node < std::min(num_target_nodes, graph.num_nodes);
         ++node) {
      graph.target_nodes.push_back(node);
    }
  }

  graph.node_to_bank = ReadIntVector(graph_node["node_to_bank"]);
  if (graph.node_to_bank.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.node_to_bank.resize(graph.num_nodes);
    for (int node = 0; node < graph.num_nodes; ++node) {
      graph.node_to_bank[node] = (node * 131 + seed) % topology.total_banks();
    }
  }

  graph.edges.reserve(num_edges);
  const int target_count = std::max<int>(1, graph.target_nodes.size());
  for (int edge_id = 0; edge_id < num_edges; ++edge_id) {
    const int src = static_cast<int>(
        (static_cast<int64_t>(edge_id) * 1103515245LL + seed) %
        graph.num_nodes);
    const int dst = graph.target_nodes[edge_id % target_count];
    const int bank = graph.node_to_bank[src];
    graph.edges.push_back(
        Edge{src, dst, edge_id, bank, topology.bank_to_pseudo_channel(bank)});
  }

  graph.node_degree.assign(graph.num_nodes, 0);
  for (const auto& edge : graph.edges) {
    graph.node_degree[edge.src]++;
    graph.node_degree[edge.dst]++;
  }

  graph.high_degree_mask = ReadBoolVector(graph_node["high_degree_mask"]);
  if (graph.high_degree_mask.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.high_degree_mask.assign(graph.num_nodes, false);
    const auto high_degree_nodes = ReadIntVector(graph_node["high_degree_nodes"]);
    if (!high_degree_nodes.empty()) {
      for (int node : high_degree_nodes) {
        if (node >= 0 && node < graph.num_nodes) {
          graph.high_degree_mask[node] = true;
        }
      }
    } else {
      std::vector<int> nodes(graph.num_nodes);
      std::iota(nodes.begin(), nodes.end(), 0);
      std::sort(nodes.begin(), nodes.end(), [&](int lhs, int rhs) {
        return graph.node_degree[lhs] > graph.node_degree[rhs];
      });
      for (int idx = 0; idx < std::min(num_high_degree_nodes, graph.num_nodes);
           ++idx) {
        graph.high_degree_mask[nodes[idx]] = true;
      }
    }
  }

  graph.selected_kv_mask = ReadBoolVector(graph_node["selected_kv_mask"]);
  NormalizeGraph(graph, topology);
  return graph;
}

QueryGraph ReadGraphFromSchema(const YAML::Node& graph_node,
                               const TopologyConfig& topology) {
  QueryGraph graph;
  graph.num_nodes = ReadScalar<int>(graph_node, "num_nodes", 0);
  graph.target_nodes = ReadIntVector(graph_node["target_nodes"]);
  graph.node_degree = ReadIntVector(graph_node["node_degree"]);
  graph.node_to_bank = ReadIntVector(graph_node["node_to_bank"]);
  graph.high_degree_mask = ReadBoolVector(graph_node["high_degree_mask"]);
  if (graph.high_degree_mask.size() != static_cast<size_t>(graph.num_nodes)) {
    graph.high_degree_mask.assign(graph.num_nodes, false);
    for (int node : ReadIntVector(graph_node["high_degree_nodes"])) {
      if (node >= 0 && node < graph.num_nodes) {
        graph.high_degree_mask[node] = true;
      }
    }
  }
  graph.selected_kv_mask = ReadBoolVector(graph_node["selected_kv_mask"]);

  const auto edge_to_bank = ReadIntVector(graph_node["edge_to_bank"]);
  const auto edge_to_pc = ReadIntVector(graph_node["edge_to_pseudo_channel"]);
  const YAML::Node edge_list = graph_node["edge_list"];
  if (!edge_list || !edge_list.IsSequence()) {
    throw std::runtime_error("query_graph.edge_list must be a sequence");
  }

  int edge_id = 0;
  for (const auto& item : edge_list) {
    if (!item.IsSequence() || item.size() < 2) {
      throw std::runtime_error("Each edge must be [src, dst] or [src, dst, id]");
    }
    Edge edge;
    edge.src = item[0].as<int>();
    edge.dst = item[1].as<int>();
    edge.id = item.size() >= 3 ? item[2].as<int>() : edge_id;
    if (edge_id < static_cast<int>(edge_to_bank.size())) {
      edge.bank = edge_to_bank[edge_id];
    } else if (edge_id < static_cast<int>(edge_to_pc.size())) {
      edge.bank = edge_to_pc[edge_id] * topology.banks_per_pseudo_channel;
    }
    graph.edges.push_back(edge);
    edge_id++;
  }

  NormalizeGraph(graph, topology);
  return graph;
}

QueryGraph LoadQueryGraph(const YAML::Node& root, const TopologyConfig& topology) {
  const YAML::Node graph_node = root["query_graph"];
  if (!graph_node) {
    return BuildSyntheticGraph(YAML::Node(), topology);
  }

  const std::string source =
      ReadScalar<std::string>(graph_node, "source", "synthetic");
  if (source == "schema") {
    return ReadGraphFromSchema(graph_node, topology);
  }
  if (source == "file") {
    const std::string path = ReadScalar<std::string>(graph_node, "file", "");
    if (path.empty()) {
      throw std::runtime_error("query_graph.file is required for source=file");
    }
    YAML::Node file_graph = YAML::LoadFile(path);
    return ReadGraphFromSchema(file_graph, topology);
  }
  return BuildSyntheticGraph(graph_node, topology);
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
  config.pe.clock_ns = ReadScalar<double>(pe, "clock_ns", 1.0);

  const YAML::Node tile = root["tile"];
  config.tile.channel_group = ReadScalar<int>(tile, "channel_group", 16);
  config.tile.head_tile = ReadScalar<int>(tile, "head_tile", 1);
  config.tile.memory_token_tiles =
      ReadIntVector(tile["memory_token_tiles"]);
  if (config.tile.memory_token_tiles.empty()) {
    config.tile.memory_token_tiles = {1, 4, 8};
  }

  const YAML::Node model = root["model"];
  config.model.num_heads = ReadScalar<int>(model, "num_heads", 1);
  config.model.head_dim = ReadScalar<int>(model, "head_dim", 128);

  const YAML::Node output = root["output"];
  config.output_csv =
      ReadScalar<std::string>(output, "csv", "../data/analytical_pim_tiles.csv");

  config.graph = LoadQueryGraph(root, config.topology);
  return config;
}

int CountSelectedKV(const QueryGraph& graph) {
  return std::count(graph.selected_kv_mask.begin(), graph.selected_kv_mask.end(),
                    true);
}

std::vector<int> CountSelectedByBank(const QueryGraph& graph,
                                     const TopologyConfig& topology) {
  std::vector<int> counts(topology.total_banks(), 0);
  for (int node = 0; node < graph.num_nodes; ++node) {
    if (!graph.selected_kv_mask[node]) {
      continue;
    }
    const int bank = PositiveModulo(graph.node_to_bank[node], topology.total_banks());
    counts[bank]++;
  }
  return counts;
}

TileResult SimulateTile(const SimulationConfig& config, int memory_token_tile) {
  const auto& graph = config.graph;
  const auto& topology = config.topology;
  const auto& precision = config.precision;
  const auto& pe = config.pe;

  const int total_banks = topology.total_banks();
  const int total_pcs = topology.total_pseudo_channels();
  const int channel_groups =
      CeilDiv(config.model.head_dim, config.tile.channel_group);
  const int heads = config.model.num_heads;
  const int edge_count = static_cast<int>(graph.edges.size());
  const int selected_kv_count = CountSelectedKV(graph);

  std::vector<int> edge_count_by_bank(total_banks, 0);
  std::vector<std::set<int>> dst_by_bank(total_banks);
  std::vector<std::set<int>> dst_by_pc(total_pcs);
  std::set<int> active_banks_set;
  std::set<int> active_pcs_set;
  std::set<int> active_dst_set;

  for (const auto& edge : graph.edges) {
    edge_count_by_bank[edge.bank]++;
    dst_by_bank[edge.bank].insert(edge.dst);
    dst_by_pc[edge.pseudo_channel].insert(edge.dst);
    active_banks_set.insert(edge.bank);
    active_pcs_set.insert(edge.pseudo_channel);
    active_dst_set.insert(edge.dst);
  }

  const auto selected_by_bank = CountSelectedByBank(graph, topology);

  double score_cycles = 0.0;
  double message_cycles = 0.0;
  double kv_cycles = 0.0;
  double pe_work_cycles = 0.0;

  for (int bank = 0; bank < total_banks; ++bank) {
    const double score_groups =
        1.0 * edge_count_by_bank[bank] * memory_token_tile * heads *
        channel_groups;
    const double message_groups = score_groups;
    const double kv_groups =
        1.0 * selected_by_bank[bank] * memory_token_tile * heads *
        channel_groups;

    const double bank_score_cycles =
        score_groups * (pe.q8k8_group_cycles + pe.scale_group_cycles) /
        topology.pe_per_bank;
    const double bank_message_cycles =
        message_groups *
        (pe.p8v8_group_cycles + pe.vadd_group_cycles + pe.scale_group_cycles) /
        topology.pe_per_bank;
    const double bank_kv_cycles =
        kv_groups *
        (pe.q8k2_lut_group_cycles + pe.p8v2_lut_group_cycles +
         2.0 * pe.scale_group_cycles) /
        topology.pe_per_bank;

    score_cycles = std::max(score_cycles, bank_score_cycles);
    message_cycles = std::max(message_cycles, bank_message_cycles);
    kv_cycles = std::max(kv_cycles, bank_kv_cycles);
    pe_work_cycles += bank_score_cycles + bank_message_cycles + bank_kv_cycles;
  }

  double pc_softmax_groups = 0.0;
  double pc_message_groups = 0.0;
  double pc_buffer_total = 0.0;
  double pc_buffer_max = 0.0;
  const double buffer_per_active_group =
      1.0 * memory_token_tile * config.tile.head_tile *
      config.tile.channel_group * precision.partial_msg_bytes;

  for (int pc = 0; pc < total_pcs; ++pc) {
    const double dst_count = static_cast<double>(dst_by_pc[pc].size());
    const double softmax_groups = dst_count * memory_token_tile * heads;
    const double message_groups =
        dst_count * memory_token_tile * heads * channel_groups;
    pc_softmax_groups += softmax_groups;
    pc_message_groups += message_groups;

    const double pc_buffer =
        dst_count * heads * channel_groups * buffer_per_active_group;
    pc_buffer_total += pc_buffer;
    pc_buffer_max = std::max(pc_buffer_max, pc_buffer);
  }

  double local_buffer_total = 0.0;
  double local_buffer_max = 0.0;
  for (int bank = 0; bank < total_banks; ++bank) {
    const double bank_buffer =
        static_cast<double>(dst_by_bank[bank].size()) * heads * channel_groups *
        buffer_per_active_group;
    local_buffer_total += bank_buffer;
    local_buffer_max = std::max(local_buffer_max, bank_buffer);
  }

  const double global_softmax_groups =
      static_cast<double>(active_dst_set.size()) * memory_token_tile * heads;
  const double global_message_groups =
      static_cast<double>(active_dst_set.size()) * memory_token_tile * heads *
      channel_groups;

  const double l1_softmax_cycles =
      pc_softmax_groups * pe.softmax_group_cycles / std::max(1, total_pcs);
  const double l2_softmax_cycles =
      global_softmax_groups * pe.softmax_group_cycles;
  const double l1_message_cycles =
      pc_message_groups * pe.reducer_group_cycles / std::max(1, total_pcs);
  const double l2_message_cycles =
      global_message_groups * pe.reducer_group_cycles;
  const double reducer_cycles =
      l1_softmax_cycles + l2_softmax_cycles + l1_message_cycles +
      l2_message_cycles;

  const double near_bank_cycles = score_cycles + message_cycles + kv_cycles;
  const double total_cycles = near_bank_cycles + reducer_cycles;

  const double kv_bytes_per_item =
      heads * (config.model.head_dim * (precision.k2_bytes + precision.v2_bytes) +
               channel_groups * 2.0 * precision.fp16_bytes);
  const double selected_kv_bytes = selected_kv_count * kv_bytes_per_item;
  const double all_node_kv_bytes = graph.num_nodes * kv_bytes_per_item;
  const double all_edge_kv_bytes = edge_count * kv_bytes_per_item;

  TileResult result;
  result.memory_token_tile = memory_token_tile;
  result.selected_kv_count = selected_kv_count;
  result.selected_kv_ratio =
      graph.num_nodes == 0 ? 0.0 : 1.0 * selected_kv_count / graph.num_nodes;
  result.selected_kv_bytes = selected_kv_bytes;
  result.kv_reduction_vs_all_nodes =
      all_node_kv_bytes == 0.0 ? 0.0 : 1.0 - selected_kv_bytes / all_node_kv_bytes;
  result.kv_reduction_vs_all_edges =
      all_edge_kv_bytes == 0.0 ? 0.0 : 1.0 - selected_kv_bytes / all_edge_kv_bytes;
  result.q_broadcast_bytes =
      1.0 * active_banks_set.size() * memory_token_tile * heads *
      config.model.head_dim * precision.q8_bytes;
  result.score_traffic_bytes =
      1.0 * edge_count * memory_token_tile * heads * precision.score_bytes +
      (pc_softmax_groups + global_softmax_groups) * 2.0 * precision.score_bytes;
  result.p_return_traffic_bytes =
      1.0 * edge_count * memory_token_tile * heads * precision.p8_bytes;
  result.message_reduce_traffic_bytes =
      (pc_message_groups + global_message_groups) * config.tile.channel_group *
      precision.partial_msg_bytes;
  result.local_combine_buffer_max_bytes = local_buffer_max;
  result.local_combine_buffer_total_bytes = local_buffer_total;
  result.pc_reducer_buffer_max_bytes = pc_buffer_max;
  result.pc_reducer_buffer_total_bytes = pc_buffer_total;
  result.gnn_score_cycles = score_cycles;
  result.gnn_message_cycles = message_cycles;
  result.cached_kv_cycles = kv_cycles;
  result.reducer_cycles = reducer_cycles;
  result.total_cycles = total_cycles;
  result.total_latency_ns = total_cycles * pe.clock_ns;
  result.near_bank_pe_utilization =
      near_bank_cycles == 0.0
          ? 0.0
          : pe_work_cycles /
                (near_bank_cycles * total_banks * topology.pe_per_bank);
  const double reducer_work =
      pc_softmax_groups * pe.softmax_group_cycles +
      global_softmax_groups * pe.softmax_group_cycles +
      pc_message_groups * pe.reducer_group_cycles +
      global_message_groups * pe.reducer_group_cycles;
  result.reducer_utilization =
      reducer_cycles == 0.0
          ? 0.0
          : reducer_work / (reducer_cycles * (total_pcs + 1));
  result.active_banks = static_cast<int>(active_banks_set.size());
  result.active_pseudo_channels = static_cast<int>(active_pcs_set.size());
  return result;
}

void WriteResultsCSV(const std::string& path,
                     const std::vector<TileResult>& results) {
  std::filesystem::path csv_path(path);
  if (csv_path.has_parent_path()) {
    std::filesystem::create_directories(csv_path.parent_path());
  }

  std::ofstream csv(path);
  if (!csv.is_open()) {
    throw std::runtime_error("Cannot open analytical PIM output CSV: " + path);
  }

  csv << "memory_token_tile,selected_kv_count,selected_kv_ratio,"
         "selected_kv_bytes,kv_reduction_vs_all_nodes,"
         "kv_reduction_vs_all_edges,q_broadcast_bytes,score_traffic_bytes,"
         "p_return_traffic_bytes,message_reduce_traffic_bytes,"
         "local_combine_buffer_max_bytes,local_combine_buffer_total_bytes,"
         "pc_reducer_buffer_max_bytes,pc_reducer_buffer_total_bytes,"
         "gnn_score_cycles,gnn_message_cycles,cached_kv_cycles,"
         "reducer_cycles,total_cycles,total_latency_ns,"
         "near_bank_pe_utilization,reducer_utilization,active_banks,"
         "active_pseudo_channels\n";

  csv << std::fixed << std::setprecision(6);
  for (const auto& result : results) {
    csv << result.memory_token_tile << "," << result.selected_kv_count << ","
        << result.selected_kv_ratio << "," << result.selected_kv_bytes << ","
        << result.kv_reduction_vs_all_nodes << ","
        << result.kv_reduction_vs_all_edges << ","
        << result.q_broadcast_bytes << "," << result.score_traffic_bytes << ","
        << result.p_return_traffic_bytes << ","
        << result.message_reduce_traffic_bytes << ","
        << result.local_combine_buffer_max_bytes << ","
        << result.local_combine_buffer_total_bytes << ","
        << result.pc_reducer_buffer_max_bytes << ","
        << result.pc_reducer_buffer_total_bytes << ","
        << result.gnn_score_cycles << "," << result.gnn_message_cycles << ","
        << result.cached_kv_cycles << "," << result.reducer_cycles << ","
        << result.total_cycles << "," << result.total_latency_ns << ","
        << result.near_bank_pe_utilization << ","
        << result.reducer_utilization << "," << result.active_banks << ","
        << result.active_pseudo_channels << "\n";
  }
}

void PrintSummary(const SimulationConfig& config,
                  const std::vector<TileResult>& results) {
  std::cout << "Analytical PIM architecture-level simulation\n";
  std::cout << "nodes=" << config.graph.num_nodes
            << ", edges=" << config.graph.edges.size()
            << ", selected_kv=" << CountSelectedKV(config.graph) << "\n";
  std::cout << "logical_hbm3: stacks=" << config.topology.num_stacks
            << ", channels_per_stack=" << config.topology.channels_per_stack
            << ", pseudo_channels="
            << config.topology.total_pseudo_channels()
            << ", banks=" << config.topology.total_banks() << "\n";
  for (const auto& result : results) {
    std::cout << "T=" << result.memory_token_tile
              << " latency_ns=" << std::fixed << std::setprecision(2)
              << result.total_latency_ns
              << " pe_util=" << std::setprecision(4)
              << result.near_bank_pe_utilization
              << " reducer_util=" << result.reducer_utilization
              << " selected_kv_ratio=" << result.selected_kv_ratio << "\n";
  }
  std::cout << "csv=" << config.output_csv << "\n";
}

}  // namespace

int RunAnalyticalPIM(const std::string& config_path) {
  try {
    SimulationConfig config = LoadConfig(config_path);
    std::vector<TileResult> results;
    for (int tile : config.tile.memory_token_tiles) {
      if (tile <= 0) {
        throw std::runtime_error("memory_token_tiles must be positive");
      }
      results.push_back(SimulateTile(config, tile));
    }
    WriteResultsCSV(config.output_csv, results);
    PrintSummary(config, results);
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}

}  // namespace analytical
}  // namespace llm_system
