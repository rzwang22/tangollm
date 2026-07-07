#include "analytical/analytical_pim.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm_system {
namespace analytical {

namespace {

constexpr const char* kH100Ideal = "h100_ideal_compute_only";
constexpr const char* kH100Realistic = "h100_realistic_cache_path";
constexpr const char* kPIMNoLocalCombine = "pim_selective_kv_no_local_combine";
constexpr const char* kPIMLocalCombine = "pim_selective_kv_local_combine";

constexpr const char* kPlacementHash = "hash";
constexpr const char* kPlacementDegreeBalanced = "degree_balanced";
constexpr const char* kPlacementSourceDstLocality = "source_dst_locality";

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
  double h100_cache_bytes_per_cycle = 4096.0;
  double h100_int2_unpack_group_cycles = 1.0;
  double h100_scale_dequant_group_cycles = 1.0;
  double h100_layout_conversion_group_cycles = 1.0;
  double h100_irregular_gather_penalty = 1.25;
  double h100_small_batch_efficiency = 0.50;

  double clock_ns = 1.0;
};

struct PrecisionConfig {
  double q8_bytes = 1.0;
  double p8_bytes = 1.0;
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

struct WorkloadCase {
  std::string name;
  int sampled_nodes = 5;
  int sampled_edges = 15;
  bool dst_skewed = false;
};

struct WorkloadSuiteConfig {
  int full_graph_nodes = 2708;
  int full_graph_edges = 10556;
  int num_queries_per_workload = 16;
  double high_degree_top_percent = 0.05;
  int seed = 777;
  std::vector<WorkloadCase> workloads;
};

struct OutputConfig {
  std::string per_query_csv = "../data/analytical_pim_per_query.csv";
  std::string aggregate_csv = "../data/analytical_pim_aggregate.csv";
};

struct Edge {
  int src = 0;
  int dst = 0;
  int id = 0;
  int bank = 0;
  int pseudo_channel = 0;
};

struct QuerySample {
  int query_id = 0;
  std::string workload;
  int target_node = 0;
  int full_graph_nodes = 0;
  std::vector<int> sampled_nodes;
  std::vector<Edge> edges;
  std::vector<bool> selected_kv_mask;
  std::vector<int> node_degree;
  std::vector<bool> high_degree_mask;
};

struct SimulationConfig {
  TopologyConfig topology;
  PEConfig pe;
  PrecisionConfig precision;
  TileConfig tile;
  ModelConfig model;
  WorkloadSuiteConfig suite;
  OutputConfig output;
  std::vector<std::string> placements{kPlacementHash, kPlacementDegreeBalanced,
                                      kPlacementSourceDstLocality};
  std::vector<std::string> baselines{kH100Ideal, kH100Realistic,
                                     kPIMNoLocalCombine, kPIMLocalCombine};
};

struct Diagnosis {
  double edge_message_count_before_local_combine = 0.0;
  double bank_local_group_count_after_combine = 0.0;
  double pc_group_count_after_pc_reduce = 0.0;
  double local_combine_reduction_ratio = 0.0;
  double pc_reduction_ratio = 0.0;
  double avg_edges_per_bank_dst = 0.0;
  double p95_edges_per_bank_dst = 0.0;
  double max_edges_per_bank_dst = 0.0;
  double message_traffic_before_local_combine = 0.0;
  double message_traffic_after_local_combine = 0.0;
  double bank_imbalance = 0.0;
  double pseudo_channel_imbalance = 0.0;
};

struct QueryResult {
  std::string workload;
  std::string placement;
  std::string baseline;
  int query_id = 0;
  int target_node = 0;
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
  Diagnosis diagnosis;
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

int PositiveModulo(int value, int modulo) {
  int result = value % modulo;
  return result < 0 ? result + modulo : result;
}

int DeterministicHash(int value, int seed, int modulo) {
  const int64_t mixed = static_cast<int64_t>(value) * 1103515245LL + seed;
  return static_cast<int>(PositiveModulo(static_cast<int>(mixed % modulo), modulo));
}

int CeilDiv(int lhs, int rhs) {
  return (lhs + rhs - 1) / rhs;
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

std::vector<WorkloadCase> DefaultWorkloads() {
  return {{"smoke", 5, 15, false},
          {"small", 16, 64, false},
          {"medium", 64, 512, false},
          {"large", 128, 2048, false},
          {"skewed_high_degree", 128, 2048, true}};
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
  config.pe.h100_cache_bytes_per_cycle =
      ReadScalar<double>(pe, "h100_cache_bytes_per_cycle", 4096.0);
  config.pe.h100_int2_unpack_group_cycles =
      ReadScalar<double>(pe, "h100_int2_unpack_group_cycles", 1.0);
  config.pe.h100_scale_dequant_group_cycles =
      ReadScalar<double>(pe, "h100_scale_dequant_group_cycles", 1.0);
  config.pe.h100_layout_conversion_group_cycles =
      ReadScalar<double>(pe, "h100_layout_conversion_group_cycles", 1.0);
  config.pe.h100_irregular_gather_penalty =
      ReadScalar<double>(pe, "h100_irregular_gather_penalty", 1.25);
  config.pe.h100_small_batch_efficiency =
      ReadScalar<double>(pe, "h100_small_batch_efficiency", 0.50);
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

  const YAML::Node suite = root["workload_suite"];
  config.suite.full_graph_nodes =
      ReadScalar<int>(suite, "full_graph_nodes", 2708);
  config.suite.full_graph_edges =
      ReadScalar<int>(suite, "full_graph_edges", 10556);
  config.suite.num_queries_per_workload =
      ReadScalar<int>(suite, "num_queries_per_workload", 16);
  config.suite.high_degree_top_percent =
      ReadScalar<double>(suite, "high_degree_top_percent", 0.05);
  config.suite.seed = ReadScalar<int>(suite, "seed", 777);
  const YAML::Node workloads = suite["workloads"];
  if (workloads && workloads.IsSequence()) {
    for (const auto& item : workloads) {
      WorkloadCase workload;
      workload.name = ReadScalar<std::string>(item, "name", "workload");
      workload.sampled_nodes = ReadScalar<int>(item, "sampled_nodes", 16);
      workload.sampled_edges = ReadScalar<int>(item, "sampled_edges", 64);
      workload.dst_skewed = ReadScalar<bool>(item, "dst_skewed", false);
      config.suite.workloads.push_back(workload);
    }
  }
  if (config.suite.workloads.empty()) {
    config.suite.workloads = DefaultWorkloads();
  }

  const auto placements = ReadStringVector(root["placement_sweep"]);
  if (!placements.empty()) {
    config.placements = placements;
  }
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
      config.suite.full_graph_nodes <= 0 ||
      config.suite.num_queries_per_workload <= 0) {
    throw std::runtime_error("Invalid analytical PIM config");
  }
  return config;
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
  std::cout << "text_kv_payload_bytes_per_item=" << std::fixed
            << std::setprecision(0) << TextKVPayloadBytesPerItem(config.model)
            << ", include_scale_metadata="
            << (config.model.include_scale_metadata ? "true" : "false")
            << ", scale_metadata_bytes_per_item="
            << ScaleMetadataBytesPerItem(config.model)
            << ", total_kv_bytes_per_item="
            << TextKVBytesPerItem(config.model) << "\n";
}

void AddUniqueNode(std::vector<int>& nodes, std::vector<bool>& mask, int node) {
  if (!mask[node]) {
    mask[node] = true;
    nodes.push_back(node);
  }
}

QuerySample GenerateQuery(const SimulationConfig& config,
                          const WorkloadCase& workload, int query_id) {
  QuerySample query;
  query.query_id = query_id;
  query.workload = workload.name;
  query.full_graph_nodes = config.suite.full_graph_nodes;
  query.target_node = DeterministicHash(query_id * 97 + 13, config.suite.seed,
                                        config.suite.full_graph_nodes);

  const int sampled_nodes =
      std::max(2, std::min(workload.sampled_nodes, config.suite.full_graph_nodes));
  const int sampled_edges = std::max(0, workload.sampled_edges);

  std::vector<bool> node_mask(config.suite.full_graph_nodes, false);
  AddUniqueNode(query.sampled_nodes, node_mask, query.target_node);
  for (int idx = 1; idx < sampled_nodes; ++idx) {
    int node = DeterministicHash(query_id * 1009 + idx * 9176,
                                 config.suite.seed + 31,
                                 config.suite.full_graph_nodes);
    while (node_mask[node]) {
      node = (node + 1) % config.suite.full_graph_nodes;
    }
    AddUniqueNode(query.sampled_nodes, node_mask, node);
  }

  const int local_nodes = static_cast<int>(query.sampled_nodes.size());
  query.node_degree.assign(config.suite.full_graph_nodes, 0);
  query.edges.reserve(sampled_edges);
  for (int edge_id = 0; edge_id < sampled_edges; ++edge_id) {
    const int src_idx =
        DeterministicHash(edge_id * 19 + query_id, config.suite.seed, local_nodes);
    int dst_idx = 0;
    if (workload.dst_skewed) {
      const int hot_set = std::max(1, local_nodes / 16);
      dst_idx = DeterministicHash(edge_id * 7 + query_id,
                                  config.suite.seed + 17, hot_set);
    } else {
      dst_idx = DeterministicHash(edge_id * 29 + query_id,
                                  config.suite.seed + 71, local_nodes);
    }
    if (dst_idx == src_idx) {
      dst_idx = (dst_idx + 1) % local_nodes;
    }
    Edge edge;
    edge.src = query.sampled_nodes[src_idx];
    edge.dst = query.sampled_nodes[dst_idx];
    edge.id = edge_id;
    query.edges.push_back(edge);
    query.node_degree[edge.src]++;
    query.node_degree[edge.dst]++;
  }

  query.high_degree_mask.assign(config.suite.full_graph_nodes, false);
  int high_degree_count = static_cast<int>(
      std::ceil(config.suite.high_degree_top_percent * query.sampled_nodes.size()));
  high_degree_count = std::max(0, std::min(high_degree_count, local_nodes));
  std::vector<int> ranked_nodes = query.sampled_nodes;
  std::sort(ranked_nodes.begin(), ranked_nodes.end(), [&](int lhs, int rhs) {
    return query.node_degree[lhs] > query.node_degree[rhs];
  });
  for (int idx = 0; idx < high_degree_count; ++idx) {
    query.high_degree_mask[ranked_nodes[idx]] = true;
  }

  query.selected_kv_mask.assign(config.suite.full_graph_nodes, false);
  query.selected_kv_mask[query.target_node] = true;
  for (const auto& edge : query.edges) {
    if (edge.src == query.target_node) {
      query.selected_kv_mask[edge.dst] = true;
    }
    if (edge.dst == query.target_node) {
      query.selected_kv_mask[edge.src] = true;
    }
  }
  for (int node : query.sampled_nodes) {
    if (query.high_degree_mask[node]) {
      query.selected_kv_mask[node] = true;
    }
  }
  return query;
}

std::vector<QuerySample> GenerateQueries(const SimulationConfig& config) {
  std::vector<QuerySample> queries;
  for (const auto& workload : config.suite.workloads) {
    for (int query_id = 0; query_id < config.suite.num_queries_per_workload;
         ++query_id) {
      queries.push_back(GenerateQuery(config, workload, query_id));
    }
  }
  return queries;
}

int NodeBankHash(int node, const SimulationConfig& config) {
  return DeterministicHash(node, config.suite.seed, config.topology.total_banks());
}

std::map<int, int> BuildDegreeBalancedNodePlacement(
    const QuerySample& query, const SimulationConfig& config) {
  std::vector<int> nodes = query.sampled_nodes;
  std::sort(nodes.begin(), nodes.end(), [&](int lhs, int rhs) {
    return query.node_degree[lhs] > query.node_degree[rhs];
  });
  std::vector<int> bank_load(config.topology.total_banks(), 0);
  std::map<int, int> node_to_bank;
  for (int node : nodes) {
    int best_bank = 0;
    for (int bank = 1; bank < config.topology.total_banks(); ++bank) {
      if (bank_load[bank] < bank_load[best_bank]) {
        best_bank = bank;
      }
    }
    node_to_bank[node] = best_bank;
    bank_load[best_bank] += std::max(1, query.node_degree[node]);
  }
  return node_to_bank;
}

QuerySample ApplyPlacement(const QuerySample& input,
                           const SimulationConfig& config,
                           const std::string& placement) {
  QuerySample query = input;
  const auto degree_balanced = BuildDegreeBalancedNodePlacement(query, config);
  for (auto& edge : query.edges) {
    if (placement == kPlacementHash) {
      edge.bank = NodeBankHash(edge.src, config);
    } else if (placement == kPlacementDegreeBalanced) {
      edge.bank = degree_balanced.at(edge.src);
    } else if (placement == kPlacementSourceDstLocality) {
      edge.bank = NodeBankHash(edge.dst, config);
    } else {
      throw std::runtime_error("Unknown placement policy: " + placement);
    }
    edge.bank = PositiveModulo(edge.bank, config.topology.total_banks());
    edge.pseudo_channel = config.topology.bank_to_pseudo_channel(edge.bank);
  }
  return query;
}

int CountSelectedKV(const QuerySample& query) {
  int count = 0;
  for (int node : query.sampled_nodes) {
    if (query.selected_kv_mask[node]) {
      count++;
    }
  }
  return count;
}

std::vector<int> SelectedKVCountByBank(const QuerySample& query,
                                       const SimulationConfig& config,
                                       const std::string& placement) {
  std::vector<int> counts(config.topology.total_banks(), 0);
  const auto degree_balanced = BuildDegreeBalancedNodePlacement(query, config);
  for (int node : query.sampled_nodes) {
    if (!query.selected_kv_mask[node]) {
      continue;
    }
    int bank = 0;
    if (placement == kPlacementDegreeBalanced) {
      bank = degree_balanced.at(node);
    } else {
      bank = NodeBankHash(node, config);
    }
    counts[PositiveModulo(bank, config.topology.total_banks())]++;
  }
  return counts;
}

Diagnosis DiagnoseLocalCombine(const QuerySample& query,
                               const SimulationConfig& config) {
  const int groups_per_head = GroupsPerHead(config);
  const double group_multiplier =
      1.0 * config.model.memory_tokens * config.model.gnn_heads *
      groups_per_head;
  std::map<std::pair<int, int>, int> bank_dst_edges;
  std::map<std::pair<int, int>, int> pc_dst_edges;
  std::vector<int> bank_counts(config.topology.total_banks(), 0);
  std::vector<int> pc_counts(config.topology.total_pseudo_channels(), 0);

  for (const auto& edge : query.edges) {
    bank_dst_edges[{edge.bank, edge.dst}]++;
    pc_dst_edges[{edge.pseudo_channel, edge.dst}]++;
    bank_counts[edge.bank]++;
    pc_counts[edge.pseudo_channel]++;
  }

  std::vector<double> edges_per_bank_dst;
  for (const auto& item : bank_dst_edges) {
    edges_per_bank_dst.push_back(item.second);
  }

  const double before = query.edges.size() * group_multiplier;
  const double after_bank = bank_dst_edges.size() * group_multiplier;
  const double after_pc = pc_dst_edges.size() * group_multiplier;

  Diagnosis d;
  d.edge_message_count_before_local_combine = before;
  d.bank_local_group_count_after_combine = after_bank;
  d.pc_group_count_after_pc_reduce = after_pc;
  d.local_combine_reduction_ratio =
      before == 0.0 ? 0.0 : 1.0 - after_bank / before;
  d.pc_reduction_ratio = after_bank == 0.0 ? 0.0 : 1.0 - after_pc / after_bank;
  d.avg_edges_per_bank_dst = Mean(edges_per_bank_dst);
  d.p95_edges_per_bank_dst = Percentile(edges_per_bank_dst, 0.95);
  d.max_edges_per_bank_dst =
      edges_per_bank_dst.empty()
          ? 0.0
          : *std::max_element(edges_per_bank_dst.begin(),
                              edges_per_bank_dst.end());
  d.message_traffic_before_local_combine =
      before * config.tile.channel_group * config.precision.partial_msg_bytes;
  d.message_traffic_after_local_combine =
      after_bank * config.tile.channel_group * config.precision.partial_msg_bytes;

  auto imbalance = [](const std::vector<int>& counts) {
    int active = 0;
    int max_count = 0;
    int total = 0;
    for (int count : counts) {
      if (count > 0) {
        active++;
        max_count = std::max(max_count, count);
        total += count;
      }
    }
    if (active == 0 || total == 0) {
      return 0.0;
    }
    return max_count / (1.0 * total / active);
  };
  d.bank_imbalance = imbalance(bank_counts);
  d.pseudo_channel_imbalance = imbalance(pc_counts);
  return d;
}

QueryResult SimulateQuery(const SimulationConfig& config,
                          const QuerySample& query,
                          const std::string& placement,
                          const std::string& baseline,
                          int memory_token_tile) {
  const int groups_per_head = GroupsPerHead(config);
  const int num_token_tiles =
      CeilDiv(config.model.memory_tokens, memory_token_tile);
  const int selected_kv_count = CountSelectedKV(query);
  const int edge_count = static_cast<int>(query.edges.size());
  const double score_groups =
      1.0 * edge_count * config.model.memory_tokens * config.model.gnn_heads *
      groups_per_head;
  const double message_groups = score_groups;
  const double cached_kv_groups =
      1.0 * selected_kv_count * config.model.suffix_layers *
      config.model.text_len * config.model.gnn_heads * groups_per_head;

  std::set<int> active_banks;
  std::set<int> active_pcs;
  std::vector<int> edge_count_by_bank(config.topology.total_banks(), 0);
  std::vector<int> selected_count_by_bank =
      SelectedKVCountByBank(query, config, placement);
  for (const auto& edge : query.edges) {
    active_banks.insert(edge.bank);
    active_pcs.insert(edge.pseudo_channel);
    edge_count_by_bank[edge.bank]++;
  }
  for (int bank = 0; bank < config.topology.total_banks(); ++bank) {
    if (selected_count_by_bank[bank] > 0) {
      active_banks.insert(bank);
      active_pcs.insert(config.topology.bank_to_pseudo_channel(bank));
    }
  }

  QueryResult result;
  result.workload = query.workload;
  result.placement = placement;
  result.baseline = baseline;
  result.query_id = query.query_id;
  result.target_node = query.target_node;
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
      query.full_graph_nodes == 0
          ? 0.0
          : 1.0 * selected_kv_count / query.full_graph_nodes;
  result.selected_kv_bytes =
      selected_kv_count * TextKVBytesPerItem(config.model);
  result.active_banks = static_cast<int>(active_banks.size());
  result.active_pseudo_channels = static_cast<int>(active_pcs.size());
  result.diagnosis = DiagnoseLocalCombine(query, config);

  if (baseline == kH100Ideal || baseline == kH100Realistic) {
    result.gnn_score_cycles = score_groups * config.pe.h100_group_cycles;
    result.gnn_message_cycles = message_groups * config.pe.h100_group_cycles;
    result.cached_kv_cycles = cached_kv_groups * config.pe.h100_group_cycles;
    double total = result.gnn_score_cycles + result.gnn_message_cycles +
                   result.cached_kv_cycles;
    if (baseline == kH100Realistic) {
      const double efficiency =
          std::max(0.01, config.pe.h100_small_batch_efficiency);
      const double cache_read_cycles =
          result.selected_kv_bytes /
          std::max(1.0, config.pe.h100_cache_bytes_per_cycle) / efficiency *
          config.pe.h100_irregular_gather_penalty;
      const double unpack_cycles =
          cached_kv_groups * config.pe.h100_int2_unpack_group_cycles;
      const double scale_cycles =
          cached_kv_groups * config.pe.h100_scale_dequant_group_cycles;
      const double layout_cycles =
          cached_kv_groups * config.pe.h100_layout_conversion_group_cycles;
      total += cache_read_cycles + unpack_cycles + scale_cycles + layout_cycles;
    }
    result.total_cycles = total;
    result.latency_ns = total * config.pe.clock_ns;
    return result;
  }

  result.q_broadcast_bytes =
      1.0 * num_token_tiles * active_banks.size() * config.model.gnn_heads *
      config.model.head_dim * config.precision.q8_bytes;
  result.score_traffic_bytes =
      score_groups / groups_per_head * config.precision.score_bytes;
  result.p_return_traffic_bytes =
      score_groups / groups_per_head * config.precision.p8_bytes;

  double score_cycles = 0.0;
  double message_cycles = 0.0;
  double kv_cycles = 0.0;
  double pe_work_cycles = 0.0;
  for (int bank = 0; bank < config.topology.total_banks(); ++bank) {
    const double bank_score_groups =
        1.0 * edge_count_by_bank[bank] * config.model.memory_tokens *
        config.model.gnn_heads * groups_per_head;
    const double bank_message_groups = bank_score_groups;
    const double bank_kv_groups =
        1.0 * selected_count_by_bank[bank] * config.model.suffix_layers *
        config.model.text_len * config.model.gnn_heads * groups_per_head;
    const double bank_score_cycles =
        bank_score_groups *
        (config.pe.q8k8_group_cycles + config.pe.scale_group_cycles) /
        config.topology.pe_per_bank;
    const double bank_message_cycles =
        bank_message_groups *
        (config.pe.p8v8_group_cycles + config.pe.scale_group_cycles +
         (baseline == kPIMLocalCombine ? config.pe.vadd_group_cycles : 0.0)) /
        config.topology.pe_per_bank;
    const double bank_kv_cycles =
        bank_kv_groups *
        (config.pe.q8k2_lut_group_cycles + config.pe.p8v2_lut_group_cycles +
         2.0 * config.pe.scale_group_cycles) /
        config.topology.pe_per_bank;
    score_cycles = std::max(score_cycles, bank_score_cycles);
    message_cycles = std::max(message_cycles, bank_message_cycles);
    kv_cycles = std::max(kv_cycles, bank_kv_cycles);
    pe_work_cycles += bank_score_cycles + bank_message_cycles + bank_kv_cycles;
  }

  const double buffer_per_active_group =
      1.0 * memory_token_tile * config.tile.head_tile *
      config.tile.channel_group * config.precision.partial_msg_bytes;
  result.local_combine_buffer_max_bytes =
      result.diagnosis.max_edges_per_bank_dst * config.model.gnn_heads *
      groups_per_head * buffer_per_active_group;
  result.pc_reducer_buffer_max_bytes =
      result.diagnosis.pc_group_count_after_pc_reduce /
      std::max(1, config.model.memory_tokens) * buffer_per_active_group;

  if (baseline == kPIMNoLocalCombine) {
    result.message_reduce_traffic_bytes =
        result.diagnosis.message_traffic_before_local_combine;
    result.reducer_cycles =
        result.diagnosis.edge_message_count_before_local_combine *
        config.pe.reducer_group_cycles / std::max<int>(1, active_pcs.size());
  } else if (baseline == kPIMLocalCombine) {
    result.message_reduce_traffic_bytes =
        (result.diagnosis.bank_local_group_count_after_combine +
         result.diagnosis.pc_group_count_after_pc_reduce) *
        config.tile.channel_group * config.precision.partial_msg_bytes;
    result.reducer_cycles =
        result.diagnosis.bank_local_group_count_after_combine *
            config.pe.reducer_group_cycles / std::max<int>(1, active_pcs.size()) +
        result.diagnosis.pc_group_count_after_pc_reduce *
            config.pe.reducer_group_cycles;
  } else {
    throw std::runtime_error("Unknown baseline: " + baseline);
  }

  result.gnn_score_cycles = score_cycles;
  result.gnn_message_cycles = message_cycles;
  result.cached_kv_cycles = kv_cycles;
  result.scheduling_cycles =
      num_token_tiles * config.pe.scheduling_overhead_per_tile_cycles;
  const double near_bank_cycles = score_cycles + message_cycles + kv_cycles;
  result.total_cycles =
      near_bank_cycles + result.reducer_cycles + result.scheduling_cycles;
  result.latency_ns = result.total_cycles * config.pe.clock_ns;
  result.near_bank_pe_utilization =
      near_bank_cycles == 0.0
          ? 0.0
          : pe_work_cycles /
                (near_bank_cycles * config.topology.total_banks() *
                 config.topology.pe_per_bank);
  result.reducer_utilization =
      result.reducer_cycles == 0.0
          ? 0.0
          : (baseline == kPIMLocalCombine
                 ? result.diagnosis.bank_local_group_count_after_combine +
                       result.diagnosis.pc_group_count_after_pc_reduce
                 : result.diagnosis.edge_message_count_before_local_combine) /
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
  csv << "workload,placement,baseline,query_id,target_node,memory_token_tile,"
         "num_token_tiles,sampled_node_count,sampled_edge_count,"
         "selected_kv_count,selected_kv_ratio_vs_sampled_nodes,"
         "selected_kv_ratio_vs_full_graph,selected_kv_bytes,"
         "q_broadcast_bytes,score_traffic_bytes,p_return_traffic_bytes,"
         "message_reduce_traffic_bytes,local_combine_buffer_max_bytes,"
         "pc_reducer_buffer_max_bytes,gnn_score_cycles,gnn_message_cycles,"
         "cached_kv_cycles,reducer_cycles,scheduling_cycles,total_cycles,"
         "latency_ns,near_bank_pe_utilization,reducer_utilization,"
         "active_banks,active_pseudo_channels,"
         "edge_message_count_before_local_combine,"
         "bank_local_group_count_after_combine,pc_group_count_after_pc_reduce,"
         "local_combine_reduction_ratio,pc_reduction_ratio,"
         "avg_edges_per_bank_dst,p95_edges_per_bank_dst,max_edges_per_bank_dst,"
         "message_traffic_before_local_combine,"
         "message_traffic_after_local_combine,bank_imbalance,"
         "pseudo_channel_imbalance\n";
  csv << std::fixed << std::setprecision(6);
  for (const auto& r : results) {
    const Diagnosis& d = r.diagnosis;
    csv << r.workload << "," << r.placement << "," << r.baseline << ","
        << r.query_id << "," << r.target_node << "," << r.memory_token_tile
        << "," << r.num_token_tiles << "," << r.sampled_node_count << ","
        << r.sampled_edge_count << "," << r.selected_kv_count << ","
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
        << r.active_banks << "," << r.active_pseudo_channels << ","
        << d.edge_message_count_before_local_combine << ","
        << d.bank_local_group_count_after_combine << ","
        << d.pc_group_count_after_pc_reduce << ","
        << d.local_combine_reduction_ratio << "," << d.pc_reduction_ratio
        << "," << d.avg_edges_per_bank_dst << ","
        << d.p95_edges_per_bank_dst << "," << d.max_edges_per_bank_dst << ","
        << d.message_traffic_before_local_combine << ","
        << d.message_traffic_after_local_combine << "," << d.bank_imbalance
        << "," << d.pseudo_channel_imbalance << "\n";
  }
}

void WriteAggregateCSV(const std::string& path,
                       const std::vector<QueryResult>& results,
                       const SimulationConfig& config) {
  PrepareCSV(path);
  std::ofstream csv(path);
  if (!csv.is_open()) {
    throw std::runtime_error("Cannot open aggregate CSV: " + path);
  }
  csv << "workload,placement,baseline,memory_token_tile,num_queries,"
         "mean_latency_ns,p50_latency_ns,p95_latency_ns,selected_kv_count,"
         "q_broadcast_bytes,score_traffic_bytes,p_return_traffic_bytes,"
         "message_reduce_traffic_bytes,local_combine_buffer_max_bytes,"
         "pc_reducer_buffer_max_bytes,active_banks,active_pseudo_channels,"
         "near_bank_pe_utilization,reducer_utilization,bank_imbalance,"
         "pseudo_channel_imbalance,local_combine_reduction_ratio,"
         "pc_reduction_ratio,avg_edges_per_bank_dst,p95_edges_per_bank_dst,"
         "max_edges_per_bank_dst,message_traffic_before_local_combine,"
         "message_traffic_after_local_combine\n";
  csv << std::fixed << std::setprecision(6);
  for (const auto& workload : config.suite.workloads) {
    for (const auto& placement : config.placements) {
      for (const auto& baseline : config.baselines) {
        for (int tile : config.tile.memory_token_tiles) {
          std::vector<const QueryResult*> group;
          for (const auto& r : results) {
            if (r.workload == workload.name && r.placement == placement &&
                r.baseline == baseline && r.memory_token_tile == tile) {
              group.push_back(&r);
            }
          }
          if (group.empty()) {
            continue;
          }
          std::vector<double> latency;
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
          std::vector<double> bank_imbalance;
          std::vector<double> pc_imbalance;
          std::vector<double> local_reduction;
          std::vector<double> pc_reduction;
          std::vector<double> avg_edges;
          std::vector<double> p95_edges;
          std::vector<double> max_edges;
          std::vector<double> traffic_before;
          std::vector<double> traffic_after;
          for (const auto* r : group) {
            latency.push_back(r->latency_ns);
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
            bank_imbalance.push_back(r->diagnosis.bank_imbalance);
            pc_imbalance.push_back(r->diagnosis.pseudo_channel_imbalance);
            local_reduction.push_back(
                r->diagnosis.local_combine_reduction_ratio);
            pc_reduction.push_back(r->diagnosis.pc_reduction_ratio);
            avg_edges.push_back(r->diagnosis.avg_edges_per_bank_dst);
            p95_edges.push_back(r->diagnosis.p95_edges_per_bank_dst);
            max_edges.push_back(r->diagnosis.max_edges_per_bank_dst);
            traffic_before.push_back(
                r->diagnosis.message_traffic_before_local_combine);
            traffic_after.push_back(
                r->diagnosis.message_traffic_after_local_combine);
          }
          csv << workload.name << "," << placement << "," << baseline << ","
              << tile << "," << group.size() << "," << Mean(latency) << ","
              << Percentile(latency, 0.50) << ","
              << Percentile(latency, 0.95) << "," << Mean(selected) << ","
              << Mean(q_broadcast) << "," << Mean(score_traffic) << ","
              << Mean(p_return) << "," << Mean(message_reduce) << ","
              << Mean(local_buffer) << "," << Mean(pc_buffer) << ","
              << Mean(active_banks) << "," << Mean(active_pcs) << ","
              << Mean(pe_util) << "," << Mean(reducer_util) << ","
              << Mean(bank_imbalance) << "," << Mean(pc_imbalance) << ","
              << Mean(local_reduction) << "," << Mean(pc_reduction) << ","
              << Mean(avg_edges) << "," << Mean(p95_edges) << ","
              << Mean(max_edges) << "," << Mean(traffic_before) << ","
              << Mean(traffic_after) << "\n";
        }
      }
    }
  }
}

void PrintSummary(const SimulationConfig& config,
                  const std::vector<QueryResult>& results) {
  std::cout << "Analytical PIM Patch 2 workload-suite run\n";
  std::cout << "workloads=" << config.suite.workloads.size()
            << ", placements=" << config.placements.size()
            << ", baselines=" << config.baselines.size()
            << ", queries_per_workload="
            << config.suite.num_queries_per_workload << "\n";
  for (const auto& workload : config.suite.workloads) {
    for (const auto& placement : config.placements) {
      std::vector<double> local_reduction;
      std::vector<double> latency;
      for (const auto& r : results) {
        if (r.workload == workload.name && r.placement == placement &&
            r.baseline == kPIMLocalCombine && r.memory_token_tile == 4) {
          local_reduction.push_back(
              r.diagnosis.local_combine_reduction_ratio);
          latency.push_back(r.latency_ns);
        }
      }
      if (!latency.empty()) {
        std::cout << workload.name << "/" << placement
                  << " local-combine T=4 mean_latency_ns=" << std::fixed
                  << std::setprecision(2) << Mean(latency)
                  << " mean_local_reduction=" << std::setprecision(4)
                  << Mean(local_reduction) << "\n";
      }
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
    std::vector<QuerySample> base_queries = GenerateQueries(config);
    std::vector<QueryResult> results;
    for (const auto& base_query : base_queries) {
      for (const auto& placement : config.placements) {
        QuerySample placed_query = ApplyPlacement(base_query, config, placement);
        for (const auto& baseline : config.baselines) {
          for (int tile : config.tile.memory_token_tiles) {
            if (tile <= 0) {
              throw std::runtime_error("memory_token_tiles must be positive");
            }
            results.push_back(
                SimulateQuery(config, placed_query, placement, baseline, tile));
          }
        }
      }
    }
    WritePerQueryCSV(config.output.per_query_csv, results);
    WriteAggregateCSV(config.output.aggregate_csv, results, config);
    PrintSummary(config, results);
  } catch (const std::exception& ex) {
    std::cerr << "ERROR: " << ex.what() << std::endl;
    return 1;
  }
  return 0;
}

}  // namespace analytical
}  // namespace llm_system
