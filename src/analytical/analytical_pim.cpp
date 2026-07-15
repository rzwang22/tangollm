#include "analytical/analytical_pim.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
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
constexpr const char* kPlacementHybrid = "hybrid_locality_balanced";

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
  std::string degree_distribution = "uniform";
  double power_law_exponent = 1.2;
  double destination_skew = 0.0;
};

struct WorkloadSuiteConfig {
  int full_graph_nodes = 2708;
  int full_graph_edges = 10556;
  int num_queries_per_workload = 16;
  double high_degree_top_percent = 0.05;
  int seed = 777;
  bool allow_duplicate_edges = false;
  std::vector<WorkloadCase> workloads;
};

struct HybridPlacementConfig {
  int hot_dst_degree_threshold = 64;
  int target_edges_per_bank = 32;
  int max_banks_per_destination = 16;
  double locality_weight = 0.25;
  double bank_balance_weight = 1.0;
  double pseudo_channel_balance_weight = 1.0;
};

struct CommunicationConfig {
  double q_broadcast_bandwidth_bytes_per_cycle_per_bank = 64.0;
  double q_broadcast_startup_cycles = 16.0;
  double bank_to_pc_bandwidth_bytes_per_cycle_per_bank = 32.0;
  double bank_to_pc_startup_cycles = 16.0;
  double pc_to_global_bandwidth_bytes_per_cycle_per_pc = 64.0;
  double pc_to_global_startup_cycles = 32.0;
  double global_to_npu_bandwidth_bytes_per_cycle = 256.0;
  double global_to_npu_startup_cycles = 32.0;
};

struct ReducerConfig {
  int pc_lanes_per_pseudo_channel = 16;
  double pc_throughput_groups_per_cycle_per_lane = 1.0;
  double pc_input_bandwidth_bytes_per_cycle = 512.0;
  int pc_concurrent_destination_groups = 64;
  int global_units = 1;
  int global_lanes_per_unit = 64;
  double global_throughput_groups_per_cycle_per_lane = 1.0;
  double global_input_bandwidth_bytes_per_cycle = 1024.0;
  int global_concurrent_destination_groups = 256;
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

struct GraphSanity {
  double duplicate_edge_count = 0.0;
  double duplicate_edge_ratio = 0.0;
  double unique_src_dst_count = 0.0;
  double in_degree_mean = 0.0;
  double in_degree_p50 = 0.0;
  double in_degree_p95 = 0.0;
  double in_degree_max = 0.0;
  double out_degree_mean = 0.0;
  double out_degree_p50 = 0.0;
  double out_degree_p95 = 0.0;
  double out_degree_max = 0.0;
  double dst_degree_hist_0 = 0.0;
  double dst_degree_hist_1 = 0.0;
  double dst_degree_hist_2_3 = 0.0;
  double dst_degree_hist_4_7 = 0.0;
  double dst_degree_hist_8_15 = 0.0;
  double dst_degree_hist_16_31 = 0.0;
  double dst_degree_hist_32_63 = 0.0;
  double dst_degree_hist_64_plus = 0.0;
  double unique_source_banks = 0.0;
};

struct PlacementValidation {
  double mapping_difference_ratio_vs_hash = 0.0;
  double edge_active_banks = 0.0;
  double edge_active_pseudo_channels = 0.0;
  double bank_edge_count_mean = 0.0;
  double bank_edge_count_p50 = 0.0;
  double bank_edge_count_p95 = 0.0;
  double bank_edge_count_max = 0.0;
  double pc_edge_count_mean = 0.0;
  double pc_edge_count_p50 = 0.0;
  double pc_edge_count_p95 = 0.0;
  double pc_edge_count_max = 0.0;
  double sharded_destination_count = 0.0;
  double total_destination_shards = 0.0;
  double average_shards_per_destination = 0.0;
  double max_shards_per_destination = 0.0;
  std::string active_bank_histogram;
  std::string active_pc_histogram;
};

struct SimulationConfig {
  TopologyConfig topology;
  PEConfig pe;
  PrecisionConfig precision;
  TileConfig tile;
  ModelConfig model;
  WorkloadSuiteConfig suite;
  HybridPlacementConfig hybrid;
  CommunicationConfig communication;
  ReducerConfig reducer;
  OutputConfig output;
  std::vector<std::string> placements{kPlacementHash, kPlacementDegreeBalanced,
                                      kPlacementSourceDstLocality,
                                      kPlacementHybrid};
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
  double q8k8_vdot_cycles = 0.0;
  double gnn_score_scale_cycles = 0.0;
  double p8v8_vmul_cycles = 0.0;
  double gnn_value_scale_cycles = 0.0;
  double local_vadd_cycles = 0.0;
  double q8k2_lut_cycles = 0.0;
  double p8v2_lut_cycles = 0.0;
  double cached_kv_scale_cycles = 0.0;
  double pc_reduce_cycles = 0.0;
  double global_reduce_cycles = 0.0;
  double h100_cache_read_cycles = 0.0;
  double h100_int2_unpack_cycles = 0.0;
  double h100_scale_dequant_cycles = 0.0;
  double h100_layout_conversion_cycles = 0.0;
  double h100_irregular_gather_penalty_cycles = 0.0;
  double h100_small_batch_penalty_cycles = 0.0;
  double compute_cycles = 0.0;
  double communication_cycles = 0.0;
  double q_broadcast_cycles = 0.0;
  double bank_to_pc_communication_cycles = 0.0;
  double pc_to_global_communication_cycles = 0.0;
  double global_to_npu_communication_cycles = 0.0;
  double q_broadcast_critical_bank_bytes = 0.0;
  double bank_to_pc_total_bytes = 0.0;
  double bank_to_pc_critical_bank_bytes = 0.0;
  double bank_to_pc_critical_pc_bytes = 0.0;
  double pc_to_global_total_bytes = 0.0;
  double pc_to_global_critical_pc_bytes = 0.0;
  double global_to_npu_bytes = 0.0;
  double pc_reducer_input_groups = 0.0;
  double pc_reducer_output_groups = 0.0;
  double global_reducer_input_groups = 0.0;
  double global_reducer_output_groups = 0.0;
  double critical_path_cycles = 0.0;
  double critical_path_latency_ns = 0.0;
  double traffic_stall_fraction = 0.0;
  std::string bottleneck_stage = "none";
  double bottleneck_cycles = 0.0;
  double bottleneck_fraction = 0.0;
  double total_cycles = 0.0;
  double latency_ns = 0.0;
  double near_bank_pe_utilization = 0.0;
  double reducer_utilization = 0.0;
  int active_banks = 0;
  int active_pseudo_channels = 0;
  int score_bottleneck_bank = -1;
  int message_bottleneck_bank = -1;
  int cached_kv_bottleneck_bank = -1;
  GraphSanity graph_sanity;
  PlacementValidation placement_validation;
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
  return {{"smoke", 5, 15, "uniform", 1.2, 0.0},
          {"small", 16, 64, "uniform", 1.2, 0.0},
          {"medium", 64, 512, "uniform", 1.2, 0.0},
          {"large", 128, 2048, "uniform", 1.2, 0.0},
          {"skewed_high_degree", 128, 2048, "power_law", 1.2, 1.5}};
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
  config.suite.allow_duplicate_edges =
      ReadScalar<bool>(suite, "allow_duplicate_edges", false);
  const YAML::Node workloads = suite["workloads"];
  if (workloads && workloads.IsSequence()) {
    for (const auto& item : workloads) {
      WorkloadCase workload;
      workload.name = ReadScalar<std::string>(item, "name", "workload");
      workload.sampled_nodes = ReadScalar<int>(item, "sampled_nodes", 16);
      workload.sampled_edges = ReadScalar<int>(item, "sampled_edges", 64);
      workload.degree_distribution =
          ReadScalar<std::string>(item, "degree_distribution", "uniform");
      workload.power_law_exponent =
          ReadScalar<double>(item, "power_law_exponent", 1.2);
      workload.destination_skew =
          ReadScalar<double>(item, "destination_skew", 0.0);
      config.suite.workloads.push_back(workload);
    }
  }
  if (config.suite.workloads.empty()) {
    config.suite.workloads = DefaultWorkloads();
  }

  const YAML::Node hybrid = root["hybrid_placement"];
  config.hybrid.hot_dst_degree_threshold =
      ReadScalar<int>(hybrid, "hot_dst_degree_threshold", 64);
  config.hybrid.target_edges_per_bank =
      ReadScalar<int>(hybrid, "target_edges_per_bank", 32);
  config.hybrid.max_banks_per_destination =
      ReadScalar<int>(hybrid, "max_banks_per_destination", 16);
  config.hybrid.locality_weight =
      ReadScalar<double>(hybrid, "locality_weight", 0.25);
  config.hybrid.bank_balance_weight =
      ReadScalar<double>(hybrid, "bank_balance_weight", 1.0);
  config.hybrid.pseudo_channel_balance_weight = ReadScalar<double>(
      hybrid, "pseudo_channel_balance_weight", 1.0);

  const YAML::Node communication = root["communication"];
  config.communication.q_broadcast_bandwidth_bytes_per_cycle_per_bank =
      ReadScalar<double>(
          communication,
          "q_broadcast_bandwidth_bytes_per_cycle_per_bank", 64.0);
  config.communication.q_broadcast_startup_cycles =
      ReadScalar<double>(communication, "q_broadcast_startup_cycles", 16.0);
  config.communication.bank_to_pc_bandwidth_bytes_per_cycle_per_bank =
      ReadScalar<double>(
          communication,
          "bank_to_pc_bandwidth_bytes_per_cycle_per_bank", 32.0);
  config.communication.bank_to_pc_startup_cycles =
      ReadScalar<double>(communication, "bank_to_pc_startup_cycles", 16.0);
  config.communication.pc_to_global_bandwidth_bytes_per_cycle_per_pc =
      ReadScalar<double>(
          communication,
          "pc_to_global_bandwidth_bytes_per_cycle_per_pc", 64.0);
  config.communication.pc_to_global_startup_cycles = ReadScalar<double>(
      communication, "pc_to_global_startup_cycles", 32.0);
  config.communication.global_to_npu_bandwidth_bytes_per_cycle =
      ReadScalar<double>(communication,
                         "global_to_npu_bandwidth_bytes_per_cycle", 256.0);
  config.communication.global_to_npu_startup_cycles = ReadScalar<double>(
      communication, "global_to_npu_startup_cycles", 32.0);

  const YAML::Node reducer = root["reducer"];
  config.reducer.pc_lanes_per_pseudo_channel = ReadScalar<int>(
      reducer, "pc_lanes_per_pseudo_channel", 16);
  config.reducer.pc_throughput_groups_per_cycle_per_lane =
      ReadScalar<double>(reducer,
                         "pc_throughput_groups_per_cycle_per_lane", 1.0);
  config.reducer.pc_input_bandwidth_bytes_per_cycle = ReadScalar<double>(
      reducer, "pc_input_bandwidth_bytes_per_cycle", 512.0);
  config.reducer.pc_concurrent_destination_groups = ReadScalar<int>(
      reducer, "pc_concurrent_destination_groups", 64);
  config.reducer.global_units =
      ReadScalar<int>(reducer, "global_units", 1);
  config.reducer.global_lanes_per_unit =
      ReadScalar<int>(reducer, "global_lanes_per_unit", 64);
  config.reducer.global_throughput_groups_per_cycle_per_lane =
      ReadScalar<double>(
          reducer, "global_throughput_groups_per_cycle_per_lane", 1.0);
  config.reducer.global_input_bandwidth_bytes_per_cycle = ReadScalar<double>(
      reducer, "global_input_bandwidth_bytes_per_cycle", 1024.0);
  config.reducer.global_concurrent_destination_groups = ReadScalar<int>(
      reducer, "global_concurrent_destination_groups", 256);

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
      config.suite.num_queries_per_workload <= 0 ||
      config.hybrid.hot_dst_degree_threshold <= 0 ||
      config.hybrid.target_edges_per_bank <= 0 ||
      config.hybrid.max_banks_per_destination <= 0 ||
      config.hybrid.locality_weight < 0.0 ||
      config.hybrid.bank_balance_weight < 0.0 ||
      config.hybrid.pseudo_channel_balance_weight < 0.0 ||
      config.communication.q_broadcast_bandwidth_bytes_per_cycle_per_bank <=
          0.0 ||
      config.communication.bank_to_pc_bandwidth_bytes_per_cycle_per_bank <=
          0.0 ||
      config.communication.pc_to_global_bandwidth_bytes_per_cycle_per_pc <=
          0.0 ||
      config.communication.global_to_npu_bandwidth_bytes_per_cycle <= 0.0 ||
      config.communication.q_broadcast_startup_cycles < 0.0 ||
      config.communication.bank_to_pc_startup_cycles < 0.0 ||
      config.communication.pc_to_global_startup_cycles < 0.0 ||
      config.communication.global_to_npu_startup_cycles < 0.0 ||
      config.reducer.pc_lanes_per_pseudo_channel <= 0 ||
      config.reducer.pc_throughput_groups_per_cycle_per_lane <= 0.0 ||
      config.reducer.pc_input_bandwidth_bytes_per_cycle <= 0.0 ||
      config.reducer.pc_concurrent_destination_groups <= 0 ||
      config.reducer.global_units <= 0 ||
      config.reducer.global_lanes_per_unit <= 0 ||
      config.reducer.global_throughput_groups_per_cycle_per_lane <= 0.0 ||
      config.reducer.global_input_bandwidth_bytes_per_cycle <= 0.0 ||
      config.reducer.global_concurrent_destination_groups <= 0) {
    throw std::runtime_error("Invalid analytical PIM config");
  }
  for (const auto& workload : config.suite.workloads) {
    if ((workload.degree_distribution != "uniform" &&
         workload.degree_distribution != "power_law") ||
        workload.sampled_nodes < 2 || workload.sampled_edges < 0 ||
        workload.power_law_exponent < 0.0 ||
        workload.destination_skew < 0.0) {
      throw std::runtime_error("Invalid workload suite config: " +
                               workload.name);
    }
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
  std::cout << "hybrid_placement: hot_dst_degree_threshold="
            << config.hybrid.hot_dst_degree_threshold
            << ", target_edges_per_bank="
            << config.hybrid.target_edges_per_bank
            << ", max_banks_per_destination="
            << config.hybrid.max_banks_per_destination
            << ", locality_weight=" << std::setprecision(2)
            << config.hybrid.locality_weight
            << ", bank_balance_weight=" << config.hybrid.bank_balance_weight
            << ", pseudo_channel_balance_weight="
            << config.hybrid.pseudo_channel_balance_weight << "\n";
  std::cout << "communication_bandwidth_bytes_per_cycle: q_per_bank="
            << config.communication
                   .q_broadcast_bandwidth_bytes_per_cycle_per_bank
            << ", bank_to_pc_per_bank="
            << config.communication
                   .bank_to_pc_bandwidth_bytes_per_cycle_per_bank
            << ", pc_to_global_per_pc="
            << config.communication
                   .pc_to_global_bandwidth_bytes_per_cycle_per_pc
            << ", global_to_npu="
            << config.communication.global_to_npu_bandwidth_bytes_per_cycle
            << "\n";
  std::cout << "reducer: pc_lanes="
            << config.reducer.pc_lanes_per_pseudo_channel
            << ", pc_input_bytes_per_cycle="
            << config.reducer.pc_input_bandwidth_bytes_per_cycle
            << ", global_units=" << config.reducer.global_units
            << ", global_lanes_per_unit="
            << config.reducer.global_lanes_per_unit
            << ", global_input_bytes_per_cycle="
            << config.reducer.global_input_bandwidth_bytes_per_cycle << "\n";
}

void AddUniqueNode(std::vector<int>& nodes, std::vector<bool>& mask, int node) {
  if (!mask[node]) {
    mask[node] = true;
    nodes.push_back(node);
  }
}

uint32_t StableStringHash(const std::string& value) {
  uint32_t hash = 2166136261u;
  for (unsigned char ch : value) {
    hash = (hash ^ ch) * 16777619u;
  }
  return hash;
}

struct WeightedEdgeCandidate {
  int src_idx = 0;
  int dst_idx = 0;
  double weight = 1.0;
  double key = 0.0;
};

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

  const uint32_t query_seed =
      static_cast<uint32_t>(config.suite.seed) ^ StableStringHash(workload.name) ^
      (0x9e3779b9u * static_cast<uint32_t>(query_id + 1));
  std::mt19937 rng(query_seed);

  std::vector<bool> node_mask(config.suite.full_graph_nodes, false);
  AddUniqueNode(query.sampled_nodes, node_mask, query.target_node);
  std::uniform_int_distribution<int> node_distribution(
      0, config.suite.full_graph_nodes - 1);
  for (int idx = 1; idx < sampled_nodes; ++idx) {
    int node = node_distribution(rng);
    while (node_mask[node]) {
      node = node_distribution(rng);
    }
    AddUniqueNode(query.sampled_nodes, node_mask, node);
  }

  const int local_nodes = static_cast<int>(query.sampled_nodes.size());
  const int max_unique_edges = local_nodes * (local_nodes - 1);
  if (!config.suite.allow_duplicate_edges &&
      sampled_edges > max_unique_edges) {
    throw std::runtime_error("Workload " + workload.name + " requests " +
                             std::to_string(sampled_edges) +
                             " unique edges, but only " +
                             std::to_string(max_unique_edges) +
                             " directed non-self edges are possible");
  }

  std::vector<int> source_order(local_nodes);
  std::vector<int> destination_order(local_nodes);
  std::iota(source_order.begin(), source_order.end(), 0);
  std::iota(destination_order.begin(), destination_order.end(), 0);
  std::shuffle(source_order.begin(), source_order.end(), rng);
  std::shuffle(destination_order.begin(), destination_order.end(), rng);
  std::vector<int> source_rank(local_nodes, 0);
  std::vector<int> destination_rank(local_nodes, 0);
  for (int rank = 0; rank < local_nodes; ++rank) {
    source_rank[source_order[rank]] = rank + 1;
    destination_rank[destination_order[rank]] = rank + 1;
  }

  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<WeightedEdgeCandidate> candidates;
  candidates.reserve(max_unique_edges);
  for (int src_idx = 0; src_idx < local_nodes; ++src_idx) {
    for (int dst_idx = 0; dst_idx < local_nodes; ++dst_idx) {
      if (src_idx == dst_idx) {
        continue;
      }
      double weight = 1.0;
      if (workload.degree_distribution == "power_law") {
        weight *= std::pow(source_rank[src_idx],
                           -workload.power_law_exponent);
        weight *= std::pow(destination_rank[dst_idx],
                           -workload.power_law_exponent);
      }
      if (workload.destination_skew > 0.0) {
        weight *=
            std::pow(destination_rank[dst_idx], -workload.destination_skew);
      }
      const double sample = std::max(uniform(rng), 1e-15);
      candidates.push_back(
          {src_idx, dst_idx, weight, -std::log(sample) / weight});
    }
  }

  std::vector<WeightedEdgeCandidate> selected_edges;
  selected_edges.reserve(sampled_edges);
  if (config.suite.allow_duplicate_edges) {
    std::vector<double> weights;
    weights.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      weights.push_back(candidate.weight);
    }
    std::discrete_distribution<size_t> edge_distribution(weights.begin(),
                                                         weights.end());
    for (int edge_id = 0; edge_id < sampled_edges; ++edge_id) {
      selected_edges.push_back(candidates[edge_distribution(rng)]);
    }
  } else {
    if (sampled_edges < static_cast<int>(candidates.size())) {
      std::nth_element(candidates.begin(), candidates.begin() + sampled_edges,
                       candidates.end(), [](const auto& lhs, const auto& rhs) {
                         return lhs.key < rhs.key;
                       });
    }
    candidates.resize(sampled_edges);
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& lhs, const auto& rhs) {
                return lhs.key < rhs.key;
              });
    selected_edges = std::move(candidates);
  }

  query.node_degree.assign(config.suite.full_graph_nodes, 0);
  query.edges.reserve(sampled_edges);
  for (int edge_id = 0; edge_id < sampled_edges; ++edge_id) {
    const int src_idx = selected_edges[edge_id].src_idx;
    const int dst_idx = selected_edges[edge_id].dst_idx;
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

GraphSanity DiagnoseGraph(const QuerySample& query,
                          const SimulationConfig& config) {
  GraphSanity sanity;
  std::set<std::pair<int, int>> unique_edges;
  std::vector<int> in_degree(query.full_graph_nodes, 0);
  std::vector<int> out_degree(query.full_graph_nodes, 0);
  std::set<int> source_banks;
  for (const auto& edge : query.edges) {
    unique_edges.insert({edge.src, edge.dst});
    in_degree[edge.dst]++;
    out_degree[edge.src]++;
    source_banks.insert(NodeBankHash(edge.src, config));
  }

  sanity.unique_src_dst_count = unique_edges.size();
  sanity.duplicate_edge_count =
      query.edges.size() - static_cast<double>(unique_edges.size());
  sanity.duplicate_edge_ratio =
      query.edges.empty() ? 0.0
                          : sanity.duplicate_edge_count / query.edges.size();
  sanity.unique_source_banks = source_banks.size();

  std::vector<double> in_values;
  std::vector<double> out_values;
  in_values.reserve(query.sampled_nodes.size());
  out_values.reserve(query.sampled_nodes.size());
  for (int node : query.sampled_nodes) {
    in_values.push_back(in_degree[node]);
    out_values.push_back(out_degree[node]);
    const int degree = in_degree[node];
    if (degree == 0) {
      sanity.dst_degree_hist_0++;
    } else if (degree == 1) {
      sanity.dst_degree_hist_1++;
    } else if (degree <= 3 && degree > 1) {
      sanity.dst_degree_hist_2_3++;
    } else if (degree <= 7 && degree > 3) {
      sanity.dst_degree_hist_4_7++;
    } else if (degree <= 15 && degree > 7) {
      sanity.dst_degree_hist_8_15++;
    } else if (degree <= 31 && degree > 15) {
      sanity.dst_degree_hist_16_31++;
    } else if (degree <= 63 && degree > 31) {
      sanity.dst_degree_hist_32_63++;
    } else if (degree >= 64) {
      sanity.dst_degree_hist_64_plus++;
    }
  }
  sanity.in_degree_mean = Mean(in_values);
  sanity.in_degree_p50 = Percentile(in_values, 0.50);
  sanity.in_degree_p95 = Percentile(in_values, 0.95);
  sanity.in_degree_max =
      in_values.empty() ? 0.0
                        : *std::max_element(in_values.begin(), in_values.end());
  sanity.out_degree_mean = Mean(out_values);
  sanity.out_degree_p50 = Percentile(out_values, 0.50);
  sanity.out_degree_p95 = Percentile(out_values, 0.95);
  sanity.out_degree_max =
      out_values.empty()
          ? 0.0
          : *std::max_element(out_values.begin(), out_values.end());
  return sanity;
}

std::string FormatActiveHistogram(const std::vector<int>& counts) {
  std::ostringstream histogram;
  bool first = true;
  for (size_t index = 0; index < counts.size(); ++index) {
    if (counts[index] == 0) {
      continue;
    }
    if (!first) {
      histogram << "|";
    }
    first = false;
    histogram << index << ":" << counts[index];
  }
  return histogram.str();
}

PlacementValidation ValidatePlacement(const QuerySample& query,
                                      const SimulationConfig& config) {
  PlacementValidation validation;
  std::vector<int> bank_counts(config.topology.total_banks(), 0);
  std::vector<int> pc_counts(config.topology.total_pseudo_channels(), 0);
  std::map<int, std::set<int>> destination_banks;
  int changed_from_hash = 0;
  for (const auto& edge : query.edges) {
    bank_counts[edge.bank]++;
    pc_counts[edge.pseudo_channel]++;
    destination_banks[edge.dst].insert(edge.bank);
    if (edge.bank != NodeBankHash(edge.src, config)) {
      changed_from_hash++;
    }
  }
  validation.mapping_difference_ratio_vs_hash =
      query.edges.empty() ? 0.0 : 1.0 * changed_from_hash / query.edges.size();
  validation.active_bank_histogram = FormatActiveHistogram(bank_counts);
  validation.active_pc_histogram = FormatActiveHistogram(pc_counts);

  std::vector<double> active_bank_counts;
  std::vector<double> active_pc_counts;
  for (int count : bank_counts) {
    if (count > 0) {
      active_bank_counts.push_back(count);
    }
  }
  for (int count : pc_counts) {
    if (count > 0) {
      active_pc_counts.push_back(count);
    }
  }
  validation.edge_active_banks = active_bank_counts.size();
  validation.edge_active_pseudo_channels = active_pc_counts.size();
  validation.bank_edge_count_mean = Mean(active_bank_counts);
  validation.bank_edge_count_p50 = Percentile(active_bank_counts, 0.50);
  validation.bank_edge_count_p95 = Percentile(active_bank_counts, 0.95);
  validation.bank_edge_count_max =
      active_bank_counts.empty()
          ? 0.0
          : *std::max_element(active_bank_counts.begin(),
                              active_bank_counts.end());
  validation.pc_edge_count_mean = Mean(active_pc_counts);
  validation.pc_edge_count_p50 = Percentile(active_pc_counts, 0.50);
  validation.pc_edge_count_p95 = Percentile(active_pc_counts, 0.95);
  validation.pc_edge_count_max =
      active_pc_counts.empty()
          ? 0.0
          : *std::max_element(active_pc_counts.begin(), active_pc_counts.end());

  std::vector<double> shards_per_destination;
  for (const auto& item : destination_banks) {
    const double shards = item.second.size();
    shards_per_destination.push_back(shards);
    validation.total_destination_shards += shards;
    if (shards > 1.0) {
      validation.sharded_destination_count++;
    }
  }
  validation.average_shards_per_destination = Mean(shards_per_destination);
  validation.max_shards_per_destination =
      shards_per_destination.empty()
          ? 0.0
          : *std::max_element(shards_per_destination.begin(),
                              shards_per_destination.end());
  return validation;
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

void ApplyHybridPlacement(QuerySample& query,
                          const SimulationConfig& config) {
  const int total_banks = config.topology.total_banks();
  const int total_pcs = config.topology.total_pseudo_channels();
  const int banks_per_pc = config.topology.banks_per_pseudo_channel;
  std::map<int, std::vector<size_t>> edges_by_dst;
  for (size_t edge_idx = 0; edge_idx < query.edges.size(); ++edge_idx) {
    edges_by_dst[query.edges[edge_idx].dst].push_back(edge_idx);
  }

  std::vector<int> destinations;
  destinations.reserve(edges_by_dst.size());
  for (const auto& item : edges_by_dst) {
    destinations.push_back(item.first);
  }
  std::sort(destinations.begin(), destinations.end(), [&](int lhs, int rhs) {
    const size_t lhs_edges = edges_by_dst.at(lhs).size();
    const size_t rhs_edges = edges_by_dst.at(rhs).size();
    return lhs_edges == rhs_edges ? lhs < rhs : lhs_edges > rhs_edges;
  });

  std::vector<int> bank_load(total_banks, 0);
  std::vector<int> pc_load(total_pcs, 0);
  for (int dst : destinations) {
    const auto& edge_indices = edges_by_dst.at(dst);
    const int dst_edges = static_cast<int>(edge_indices.size());
    int shard_count = 1;
    if (dst_edges >= config.hybrid.hot_dst_degree_threshold) {
      shard_count = CeilDiv(dst_edges, config.hybrid.target_edges_per_bank);
    }
    shard_count = std::max(
        1, std::min({shard_count, config.hybrid.max_banks_per_destination,
                     total_banks, dst_edges}));

    const int anchor_bank = NodeBankHash(dst, config);
    const int anchor_pc = config.topology.bank_to_pseudo_channel(anchor_bank);
    std::vector<int> shard_banks;
    shard_banks.reserve(shard_count);
    std::set<int> selected_banks;
    for (int shard = 0; shard < shard_count; ++shard) {
      const int shard_edges =
          dst_edges / shard_count + (shard < dst_edges % shard_count ? 1 : 0);
      int best_bank = -1;
      double best_score = std::numeric_limits<double>::infinity();
      int best_distance = total_banks;
      for (int bank = 0; bank < total_banks; ++bank) {
        if (selected_banks.count(bank) != 0) {
          continue;
        }
        const int pc = config.topology.bank_to_pseudo_channel(bank);
        const double bank_pressure =
            1.0 * (bank_load[bank] + shard_edges) /
            config.hybrid.target_edges_per_bank;
        const double pc_pressure =
            1.0 * (pc_load[pc] + shard_edges) /
            (config.hybrid.target_edges_per_bank * banks_per_pc);
        const double locality_penalty =
            bank == anchor_bank ? 0.0 : (pc == anchor_pc ? 0.5 : 1.0);
        const double score =
            config.hybrid.bank_balance_weight * bank_pressure +
            config.hybrid.pseudo_channel_balance_weight * pc_pressure +
            config.hybrid.locality_weight * locality_penalty;
        const int distance = PositiveModulo(bank - anchor_bank, total_banks);
        if (score < best_score - 1e-12 ||
            (std::abs(score - best_score) <= 1e-12 &&
             distance < best_distance)) {
          best_bank = bank;
          best_score = score;
          best_distance = distance;
        }
      }
      selected_banks.insert(best_bank);
      shard_banks.push_back(best_bank);
      bank_load[best_bank] += shard_edges;
      pc_load[config.topology.bank_to_pseudo_channel(best_bank)] += shard_edges;
    }

    size_t edge_offset = 0;
    for (int shard = 0; shard < shard_count; ++shard) {
      const int shard_edges =
          dst_edges / shard_count + (shard < dst_edges % shard_count ? 1 : 0);
      const int bank = shard_banks[shard];
      for (int idx = 0; idx < shard_edges; ++idx) {
        Edge& edge = query.edges[edge_indices[edge_offset++]];
        edge.bank = bank;
        edge.pseudo_channel = config.topology.bank_to_pseudo_channel(bank);
      }
    }
  }
}

QuerySample ApplyPlacement(const QuerySample& input,
                           const SimulationConfig& config,
                           const std::string& placement) {
  QuerySample query = input;
  const auto degree_balanced = BuildDegreeBalancedNodePlacement(query, config);
  if (placement == kPlacementHybrid) {
    ApplyHybridPlacement(query, config);
    return query;
  }
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

void SetBottleneck(
    QueryResult& result,
    const std::vector<std::pair<std::string, double>>& stage_cycles) {
  for (const auto& stage : stage_cycles) {
    if (stage.second > result.bottleneck_cycles) {
      result.bottleneck_stage = stage.first;
      result.bottleneck_cycles = stage.second;
    }
  }
  result.bottleneck_fraction =
      result.total_cycles == 0.0
          ? 0.0
          : result.bottleneck_cycles / result.total_cycles;
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
  result.graph_sanity = DiagnoseGraph(query, config);
  result.placement_validation = ValidatePlacement(query, config);
  result.diagnosis = DiagnoseLocalCombine(query, config);

  if (baseline == kH100Ideal || baseline == kH100Realistic) {
    result.gnn_score_cycles = score_groups * config.pe.h100_group_cycles;
    result.gnn_message_cycles = message_groups * config.pe.h100_group_cycles;
    result.cached_kv_cycles = cached_kv_groups * config.pe.h100_group_cycles;
    double total = result.gnn_score_cycles + result.gnn_message_cycles +
                   result.cached_kv_cycles;
    if (baseline == kH100Realistic) {
      const double efficiency =
          std::min(1.0, std::max(0.01, config.pe.h100_small_batch_efficiency));
      result.h100_cache_read_cycles =
          result.selected_kv_bytes /
          std::max(1.0, config.pe.h100_cache_bytes_per_cycle);
      result.h100_irregular_gather_penalty_cycles =
          result.h100_cache_read_cycles *
          std::max(0.0, config.pe.h100_irregular_gather_penalty - 1.0);
      result.h100_small_batch_penalty_cycles =
          (result.h100_cache_read_cycles +
           result.h100_irregular_gather_penalty_cycles) *
          (1.0 / efficiency - 1.0);
      result.h100_int2_unpack_cycles =
          cached_kv_groups * config.pe.h100_int2_unpack_group_cycles;
      result.h100_scale_dequant_cycles =
          cached_kv_groups * config.pe.h100_scale_dequant_group_cycles;
      result.h100_layout_conversion_cycles =
          cached_kv_groups * config.pe.h100_layout_conversion_group_cycles;
      total += result.h100_cache_read_cycles +
               result.h100_irregular_gather_penalty_cycles +
               result.h100_small_batch_penalty_cycles +
               result.h100_int2_unpack_cycles +
               result.h100_scale_dequant_cycles +
               result.h100_layout_conversion_cycles;
    }
    result.compute_cycles = total;
    result.total_cycles = total;
    result.latency_ns = total * config.pe.clock_ns;
    result.critical_path_cycles = result.total_cycles;
    result.critical_path_latency_ns = result.latency_ns;
    SetBottleneck(
        result,
        {{"h100_gnn_score_compute", result.gnn_score_cycles},
         {"h100_gnn_value_compute", result.gnn_message_cycles},
         {"h100_cached_kv_compute", result.cached_kv_cycles},
         {"h100_cache_read", result.h100_cache_read_cycles},
         {"h100_int2_unpack", result.h100_int2_unpack_cycles},
         {"h100_scale_dequant", result.h100_scale_dequant_cycles},
         {"h100_layout_conversion", result.h100_layout_conversion_cycles},
         {"h100_irregular_gather_penalty",
          result.h100_irregular_gather_penalty_cycles},
         {"h100_small_batch_penalty",
          result.h100_small_batch_penalty_cycles}});
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
    const double bank_q8k8_vdot_cycles =
        bank_score_groups * config.pe.q8k8_group_cycles /
        config.topology.pe_per_bank;
    const double bank_score_scale_cycles =
        bank_score_groups * config.pe.scale_group_cycles /
        config.topology.pe_per_bank;
    const double bank_p8v8_vmul_cycles =
        bank_message_groups * config.pe.p8v8_group_cycles /
        config.topology.pe_per_bank;
    const double bank_message_scale_cycles =
        bank_message_groups * config.pe.scale_group_cycles /
        config.topology.pe_per_bank;
    const double bank_local_vadd_cycles =
        baseline == kPIMLocalCombine
            ? bank_message_groups * config.pe.vadd_group_cycles /
                  config.topology.pe_per_bank
            : 0.0;
    const double bank_q8k2_lut_cycles =
        bank_kv_groups * config.pe.q8k2_lut_group_cycles /
        config.topology.pe_per_bank;
    const double bank_p8v2_lut_cycles =
        bank_kv_groups * config.pe.p8v2_lut_group_cycles /
        config.topology.pe_per_bank;
    const double bank_kv_scale_cycles =
        bank_kv_groups * 2.0 * config.pe.scale_group_cycles /
        config.topology.pe_per_bank;
    if (bank_score_cycles > score_cycles) {
      result.score_bottleneck_bank = bank;
    }
    if (bank_message_cycles > message_cycles) {
      result.message_bottleneck_bank = bank;
    }
    if (bank_kv_cycles > kv_cycles) {
      result.cached_kv_bottleneck_bank = bank;
    }
    score_cycles = std::max(score_cycles, bank_score_cycles);
    message_cycles = std::max(message_cycles, bank_message_cycles);
    kv_cycles = std::max(kv_cycles, bank_kv_cycles);
    result.q8k8_vdot_cycles =
        std::max(result.q8k8_vdot_cycles, bank_q8k8_vdot_cycles);
    result.gnn_score_scale_cycles =
        std::max(result.gnn_score_scale_cycles, bank_score_scale_cycles);
    result.p8v8_vmul_cycles =
        std::max(result.p8v8_vmul_cycles, bank_p8v8_vmul_cycles);
    result.gnn_value_scale_cycles =
        std::max(result.gnn_value_scale_cycles, bank_message_scale_cycles);
    result.local_vadd_cycles =
        std::max(result.local_vadd_cycles, bank_local_vadd_cycles);
    result.q8k2_lut_cycles =
        std::max(result.q8k2_lut_cycles, bank_q8k2_lut_cycles);
    result.p8v2_lut_cycles =
        std::max(result.p8v2_lut_cycles, bank_p8v2_lut_cycles);
    result.cached_kv_scale_cycles =
        std::max(result.cached_kv_scale_cycles, bank_kv_scale_cycles);
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

  if (baseline != kPIMNoLocalCombine && baseline != kPIMLocalCombine) {
    throw std::runtime_error("Unknown baseline: " + baseline);
  }
  const double group_multiplier =
      1.0 * config.model.memory_tokens * config.model.gnn_heads *
      groups_per_head;
  const double message_bytes_per_group =
      config.tile.channel_group * config.precision.partial_msg_bytes;
  std::vector<std::set<int>> destinations_by_bank(
      config.topology.total_banks());
  std::vector<std::set<int>> destinations_by_pc(
      config.topology.total_pseudo_channels());
  std::set<int> global_destinations;
  for (const auto& edge : query.edges) {
    destinations_by_bank[edge.bank].insert(edge.dst);
    destinations_by_pc[edge.pseudo_channel].insert(edge.dst);
    global_destinations.insert(edge.dst);
  }

  std::vector<double> bank_to_pc_bytes_by_bank(
      config.topology.total_banks(), 0.0);
  std::vector<double> pc_input_groups_by_pc(
      config.topology.total_pseudo_channels(), 0.0);
  for (int bank = 0; bank < config.topology.total_banks(); ++bank) {
    const double raw_messages =
        baseline == kPIMLocalCombine ? destinations_by_bank[bank].size()
                                     : edge_count_by_bank[bank];
    const double groups = raw_messages * group_multiplier;
    const double bytes = groups * message_bytes_per_group;
    bank_to_pc_bytes_by_bank[bank] = bytes;
    pc_input_groups_by_pc[config.topology.bank_to_pseudo_channel(bank)] +=
        groups;
  }

  std::vector<double> pc_output_groups_by_pc(
      config.topology.total_pseudo_channels(), 0.0);
  std::vector<double> pc_to_global_bytes_by_pc(
      config.topology.total_pseudo_channels(), 0.0);
  for (int pc = 0; pc < config.topology.total_pseudo_channels(); ++pc) {
    pc_output_groups_by_pc[pc] =
        destinations_by_pc[pc].size() * group_multiplier;
    pc_to_global_bytes_by_pc[pc] =
        pc_output_groups_by_pc[pc] * message_bytes_per_group;
  }

  result.pc_reducer_input_groups =
      std::accumulate(pc_input_groups_by_pc.begin(),
                      pc_input_groups_by_pc.end(), 0.0);
  result.pc_reducer_output_groups =
      std::accumulate(pc_output_groups_by_pc.begin(),
                      pc_output_groups_by_pc.end(), 0.0);
  result.global_reducer_input_groups = result.pc_reducer_output_groups;
  result.global_reducer_output_groups =
      global_destinations.size() * group_multiplier;

  result.q_broadcast_critical_bank_bytes =
      active_banks.empty()
          ? 0.0
          : 1.0 * num_token_tiles * config.model.gnn_heads *
                config.model.head_dim * config.precision.q8_bytes;
  if (result.q_broadcast_bytes > 0.0) {
    result.q_broadcast_cycles =
        num_token_tiles * config.communication.q_broadcast_startup_cycles +
        result.q_broadcast_critical_bank_bytes /
            config.communication
                .q_broadcast_bandwidth_bytes_per_cycle_per_bank;
  }

  result.bank_to_pc_total_bytes =
      std::accumulate(bank_to_pc_bytes_by_bank.begin(),
                      bank_to_pc_bytes_by_bank.end(), 0.0);
  result.bank_to_pc_critical_bank_bytes =
      bank_to_pc_bytes_by_bank.empty()
          ? 0.0
          : *std::max_element(bank_to_pc_bytes_by_bank.begin(),
                              bank_to_pc_bytes_by_bank.end());
  std::vector<double> bank_to_pc_bytes_by_pc(
      config.topology.total_pseudo_channels(), 0.0);
  for (int bank = 0; bank < config.topology.total_banks(); ++bank) {
    bank_to_pc_bytes_by_pc[config.topology.bank_to_pseudo_channel(bank)] +=
        bank_to_pc_bytes_by_bank[bank];
  }
  result.bank_to_pc_critical_pc_bytes =
      bank_to_pc_bytes_by_pc.empty()
          ? 0.0
          : *std::max_element(bank_to_pc_bytes_by_pc.begin(),
                              bank_to_pc_bytes_by_pc.end());
  if (result.bank_to_pc_total_bytes > 0.0) {
    result.bank_to_pc_communication_cycles =
        num_token_tiles * config.communication.bank_to_pc_startup_cycles +
        std::max(
            result.bank_to_pc_critical_bank_bytes /
                config.communication
                    .bank_to_pc_bandwidth_bytes_per_cycle_per_bank,
            result.bank_to_pc_critical_pc_bytes /
                config.reducer.pc_input_bandwidth_bytes_per_cycle);
  }

  result.pc_to_global_total_bytes =
      std::accumulate(pc_to_global_bytes_by_pc.begin(),
                      pc_to_global_bytes_by_pc.end(), 0.0);
  result.pc_to_global_critical_pc_bytes =
      pc_to_global_bytes_by_pc.empty()
          ? 0.0
          : *std::max_element(pc_to_global_bytes_by_pc.begin(),
                              pc_to_global_bytes_by_pc.end());
  if (result.pc_to_global_total_bytes > 0.0) {
    result.pc_to_global_communication_cycles =
        num_token_tiles * config.communication.pc_to_global_startup_cycles +
        std::max(
            result.pc_to_global_critical_pc_bytes /
                config.communication
                    .pc_to_global_bandwidth_bytes_per_cycle_per_pc,
            result.pc_to_global_total_bytes /
                config.reducer.global_input_bandwidth_bytes_per_cycle);
  }

  result.global_to_npu_bytes =
      result.global_reducer_output_groups * message_bytes_per_group;
  if (result.global_to_npu_bytes > 0.0) {
    result.global_to_npu_communication_cycles =
        num_token_tiles * config.communication.global_to_npu_startup_cycles +
        result.global_to_npu_bytes /
            config.communication.global_to_npu_bandwidth_bytes_per_cycle;
  }

  const double max_pc_input_groups =
      pc_input_groups_by_pc.empty()
          ? 0.0
          : *std::max_element(pc_input_groups_by_pc.begin(),
                              pc_input_groups_by_pc.end());
  const int pc_effective_lanes = std::min(
      config.reducer.pc_lanes_per_pseudo_channel,
      config.reducer.pc_concurrent_destination_groups);
  result.pc_reduce_cycles =
      max_pc_input_groups /
      (pc_effective_lanes *
       config.reducer.pc_throughput_groups_per_cycle_per_lane);
  const int global_effective_lanes = std::min(
      config.reducer.global_units * config.reducer.global_lanes_per_unit,
      config.reducer.global_concurrent_destination_groups);
  result.global_reduce_cycles =
      result.global_reducer_input_groups /
      (global_effective_lanes *
       config.reducer.global_throughput_groups_per_cycle_per_lane);

  result.message_reduce_traffic_bytes =
      result.bank_to_pc_total_bytes + result.pc_to_global_total_bytes;
  result.gnn_score_cycles = score_cycles;
  result.gnn_message_cycles = message_cycles;
  result.cached_kv_cycles = kv_cycles;
  result.compute_cycles = score_cycles + message_cycles + kv_cycles;
  result.reducer_cycles =
      result.pc_reduce_cycles + result.global_reduce_cycles;
  result.communication_cycles =
      result.q_broadcast_cycles + result.bank_to_pc_communication_cycles +
      result.pc_to_global_communication_cycles +
      result.global_to_npu_communication_cycles;
  result.scheduling_cycles =
      num_token_tiles * config.pe.scheduling_overhead_per_tile_cycles;
  result.total_cycles = result.compute_cycles + result.communication_cycles +
                        result.reducer_cycles + result.scheduling_cycles;
  result.latency_ns = result.total_cycles * config.pe.clock_ns;
  result.critical_path_cycles = result.total_cycles;
  result.critical_path_latency_ns = result.latency_ns;
  result.traffic_stall_fraction =
      result.total_cycles == 0.0
          ? 0.0
          : result.communication_cycles / result.total_cycles;
  const double near_bank_cycles = result.compute_cycles;
  result.near_bank_pe_utilization =
      near_bank_cycles == 0.0
          ? 0.0
          : pe_work_cycles /
                (near_bank_cycles * config.topology.total_banks() *
                 config.topology.pe_per_bank);
  result.reducer_utilization =
      result.reducer_cycles == 0.0
          ? 0.0
          : (result.pc_reducer_input_groups +
             result.global_reducer_input_groups) /
                (result.pc_reduce_cycles *
                     std::max(1.0,
                              result.placement_validation
                                  .edge_active_pseudo_channels) *
                     pc_effective_lanes *
                     config.reducer
                         .pc_throughput_groups_per_cycle_per_lane +
                 result.global_reduce_cycles * global_effective_lanes *
                     config.reducer
                         .global_throughput_groups_per_cycle_per_lane);
  SetBottleneck(
      result,
      {{"pim_q8k8_vdot", result.q8k8_vdot_cycles},
       {"pim_gnn_score_scale", result.gnn_score_scale_cycles},
       {"pim_p8v8_vmul", result.p8v8_vmul_cycles},
       {"pim_gnn_value_scale", result.gnn_value_scale_cycles},
       {"pim_local_vadd", result.local_vadd_cycles},
       {"pim_q8k2_lut", result.q8k2_lut_cycles},
       {"pim_p8v2_lut", result.p8v2_lut_cycles},
       {"pim_cached_kv_scale", result.cached_kv_scale_cycles},
       {"pim_pc_reduce", result.pc_reduce_cycles},
       {"pim_global_reduce", result.global_reduce_cycles},
       {"q_broadcast_communication", result.q_broadcast_cycles},
       {"bank_to_pc_communication",
        result.bank_to_pc_communication_cycles},
       {"pc_to_global_communication",
        result.pc_to_global_communication_cycles},
       {"global_to_npu_communication",
        result.global_to_npu_communication_cycles},
       {"scheduling", result.scheduling_cycles}});
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
         "pseudo_channel_imbalance,q8k8_vdot_cycles,"
         "gnn_score_scale_cycles,p8v8_vmul_cycles,"
         "gnn_value_scale_cycles,local_vadd_cycles,q8k2_lut_cycles,"
         "p8v2_lut_cycles,cached_kv_scale_cycles,pc_reduce_cycles,"
         "global_reduce_cycles,h100_cache_read_cycles,"
         "h100_int2_unpack_cycles,h100_scale_dequant_cycles,"
         "h100_layout_conversion_cycles,"
         "h100_irregular_gather_penalty_cycles,"
         "h100_small_batch_penalty_cycles,bottleneck_stage,"
         "bottleneck_cycles,bottleneck_fraction,score_bottleneck_bank,"
         "message_bottleneck_bank,cached_kv_bottleneck_bank,"
         "duplicate_edge_count,duplicate_edge_ratio,unique_src_dst_count,"
         "in_degree_mean,in_degree_p50,in_degree_p95,in_degree_max,"
         "out_degree_mean,out_degree_p50,out_degree_p95,out_degree_max,"
         "dst_degree_hist_0,dst_degree_hist_1,dst_degree_hist_2_3,"
         "dst_degree_hist_4_7,"
         "dst_degree_hist_8_15,dst_degree_hist_16_31,"
         "dst_degree_hist_32_63,dst_degree_hist_64_plus,"
         "unique_source_banks,mapping_difference_ratio_vs_hash,"
         "edge_active_banks,edge_active_pseudo_channels,"
         "bank_edge_count_mean,bank_edge_count_p50,bank_edge_count_p95,"
         "bank_edge_count_max,pc_edge_count_mean,pc_edge_count_p50,"
         "pc_edge_count_p95,pc_edge_count_max,sharded_destination_count,"
         "total_destination_shards,average_shards_per_destination,"
         "max_shards_per_destination,active_bank_histogram,"
         "active_pc_histogram,compute_cycles,communication_cycles,"
         "q_broadcast_cycles,bank_to_pc_communication_cycles,"
         "pc_to_global_communication_cycles,"
         "global_to_npu_communication_cycles,critical_path_cycles,"
         "critical_path_latency_ns,traffic_stall_fraction,"
         "q_broadcast_critical_bank_bytes,bank_to_pc_total_bytes,"
         "bank_to_pc_critical_bank_bytes,bank_to_pc_critical_pc_bytes,"
         "pc_to_global_total_bytes,pc_to_global_critical_pc_bytes,"
         "global_to_npu_bytes,pc_reducer_input_groups,"
         "pc_reducer_output_groups,global_reducer_input_groups,"
         "global_reducer_output_groups\n";
  csv << std::fixed << std::setprecision(6);
  for (const auto& r : results) {
    const Diagnosis& d = r.diagnosis;
    const GraphSanity& g = r.graph_sanity;
    const PlacementValidation& p = r.placement_validation;
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
        << "," << d.pseudo_channel_imbalance << "," << r.q8k8_vdot_cycles
        << "," << r.gnn_score_scale_cycles << "," << r.p8v8_vmul_cycles
        << "," << r.gnn_value_scale_cycles << "," << r.local_vadd_cycles
        << "," << r.q8k2_lut_cycles << "," << r.p8v2_lut_cycles << ","
        << r.cached_kv_scale_cycles << "," << r.pc_reduce_cycles << ","
        << r.global_reduce_cycles << "," << r.h100_cache_read_cycles << ","
        << r.h100_int2_unpack_cycles << ","
        << r.h100_scale_dequant_cycles << ","
        << r.h100_layout_conversion_cycles << ","
        << r.h100_irregular_gather_penalty_cycles << ","
        << r.h100_small_batch_penalty_cycles << "," << r.bottleneck_stage
        << "," << r.bottleneck_cycles << "," << r.bottleneck_fraction << ","
        << r.score_bottleneck_bank << "," << r.message_bottleneck_bank << ","
        << r.cached_kv_bottleneck_bank << "," << g.duplicate_edge_count << ","
        << g.duplicate_edge_ratio << "," << g.unique_src_dst_count << ","
        << g.in_degree_mean << "," << g.in_degree_p50 << ","
        << g.in_degree_p95 << "," << g.in_degree_max << ","
        << g.out_degree_mean << "," << g.out_degree_p50 << ","
        << g.out_degree_p95 << "," << g.out_degree_max << ","
        << g.dst_degree_hist_0 << "," << g.dst_degree_hist_1 << ","
        << g.dst_degree_hist_2_3 << ","
        << g.dst_degree_hist_4_7 << "," << g.dst_degree_hist_8_15 << ","
        << g.dst_degree_hist_16_31 << "," << g.dst_degree_hist_32_63 << ","
        << g.dst_degree_hist_64_plus << "," << g.unique_source_banks << ","
        << p.mapping_difference_ratio_vs_hash << "," << p.edge_active_banks
        << "," << p.edge_active_pseudo_channels << ","
        << p.bank_edge_count_mean << "," << p.bank_edge_count_p50 << ","
        << p.bank_edge_count_p95 << "," << p.bank_edge_count_max << ","
        << p.pc_edge_count_mean << "," << p.pc_edge_count_p50 << ","
        << p.pc_edge_count_p95 << "," << p.pc_edge_count_max << ","
        << p.sharded_destination_count << "," << p.total_destination_shards
        << "," << p.average_shards_per_destination << ","
        << p.max_shards_per_destination << "," << p.active_bank_histogram
        << "," << p.active_pc_histogram << "," << r.compute_cycles << ","
        << r.communication_cycles << "," << r.q_broadcast_cycles << ","
        << r.bank_to_pc_communication_cycles << ","
        << r.pc_to_global_communication_cycles << ","
        << r.global_to_npu_communication_cycles << ","
        << r.critical_path_cycles << "," << r.critical_path_latency_ns << ","
        << r.traffic_stall_fraction << ","
        << r.q_broadcast_critical_bank_bytes << ","
        << r.bank_to_pc_total_bytes << ","
        << r.bank_to_pc_critical_bank_bytes << ","
        << r.bank_to_pc_critical_pc_bytes << ","
        << r.pc_to_global_total_bytes << ","
        << r.pc_to_global_critical_pc_bytes << "," << r.global_to_npu_bytes
        << "," << r.pc_reducer_input_groups << ","
        << r.pc_reducer_output_groups << ","
        << r.global_reducer_input_groups << ","
        << r.global_reducer_output_groups << "\n";
  }
}

double MeanResultField(const std::vector<const QueryResult*>& group,
                       double QueryResult::*field) {
  std::vector<double> values;
  values.reserve(group.size());
  for (const auto* result : group) {
    values.push_back(result->*field);
  }
  return Mean(values);
}

template <typename Getter>
double MeanResultValue(const std::vector<const QueryResult*>& group,
                       Getter getter) {
  std::vector<double> values;
  values.reserve(group.size());
  for (const auto* result : group) {
    values.push_back(getter(*result));
  }
  return Mean(values);
}

double MeanGraphField(const std::vector<const QueryResult*>& group,
                      double GraphSanity::*field) {
  return MeanResultValue(group,
                         [field](const QueryResult& result) {
                           return result.graph_sanity.*field;
                         });
}

double MeanPlacementField(const std::vector<const QueryResult*>& group,
                          double PlacementValidation::*field) {
  return MeanResultValue(group,
                         [field](const QueryResult& result) {
                           return result.placement_validation.*field;
                         });
}

std::pair<std::string, double> DominantBottleneck(
    const std::vector<const QueryResult*>& group) {
  std::map<std::string, int> counts;
  for (const auto* result : group) {
    counts[result->bottleneck_stage]++;
  }
  std::pair<std::string, int> dominant{"none", 0};
  for (const auto& item : counts) {
    if (item.second > dominant.second) {
      dominant = item;
    }
  }
  return {dominant.first,
          group.empty() ? 0.0 : 1.0 * dominant.second / group.size()};
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
         "message_traffic_after_local_combine,mean_gnn_score_cycles,"
         "mean_gnn_message_cycles,mean_cached_kv_cycles,"
         "mean_reducer_cycles,mean_scheduling_cycles,"
         "mean_q8k8_vdot_cycles,mean_gnn_score_scale_cycles,"
         "mean_p8v8_vmul_cycles,mean_gnn_value_scale_cycles,"
         "mean_local_vadd_cycles,mean_q8k2_lut_cycles,"
         "mean_p8v2_lut_cycles,mean_cached_kv_scale_cycles,"
         "mean_pc_reduce_cycles,mean_global_reduce_cycles,"
         "mean_h100_cache_read_cycles,mean_h100_int2_unpack_cycles,"
         "mean_h100_scale_dequant_cycles,"
         "mean_h100_layout_conversion_cycles,"
         "mean_h100_irregular_gather_penalty_cycles,"
         "mean_h100_small_batch_penalty_cycles,dominant_bottleneck_stage,"
         "bottleneck_stage_query_fraction,mean_bottleneck_cycles,"
         "mean_bottleneck_fraction,duplicate_edge_count,"
         "duplicate_edge_ratio,unique_src_dst_count,in_degree_mean,"
         "in_degree_p50,in_degree_p95,in_degree_max,out_degree_mean,"
         "out_degree_p50,out_degree_p95,out_degree_max,dst_degree_hist_0,"
         "dst_degree_hist_1,dst_degree_hist_2_3,dst_degree_hist_4_7,"
         "dst_degree_hist_8_15,"
         "dst_degree_hist_16_31,dst_degree_hist_32_63,"
         "dst_degree_hist_64_plus,unique_source_banks,"
         "mapping_difference_ratio_vs_hash,edge_active_banks,"
         "edge_active_pseudo_channels,bank_edge_count_mean,"
         "bank_edge_count_p50,bank_edge_count_p95,bank_edge_count_max,"
         "pc_edge_count_mean,pc_edge_count_p50,pc_edge_count_p95,"
         "pc_edge_count_max,sharded_destination_count,"
         "total_destination_shards,average_shards_per_destination,"
         "max_shards_per_destination,mean_compute_cycles,"
         "mean_communication_cycles,mean_q_broadcast_cycles,"
         "mean_bank_to_pc_communication_cycles,"
         "mean_pc_to_global_communication_cycles,"
         "mean_global_to_npu_communication_cycles,"
         "mean_critical_path_cycles,mean_critical_path_latency_ns,"
         "mean_traffic_stall_fraction,"
         "mean_q_broadcast_critical_bank_bytes,"
         "mean_bank_to_pc_total_bytes,"
         "mean_bank_to_pc_critical_bank_bytes,"
         "mean_bank_to_pc_critical_pc_bytes,"
         "mean_pc_to_global_total_bytes,"
         "mean_pc_to_global_critical_pc_bytes,mean_global_to_npu_bytes,"
         "mean_pc_reducer_input_groups,mean_pc_reducer_output_groups,"
         "mean_global_reducer_input_groups,"
         "mean_global_reducer_output_groups\n";
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
          const auto dominant_bottleneck = DominantBottleneck(group);
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
              << Mean(traffic_after) << ","
              << MeanResultField(group, &QueryResult::gnn_score_cycles) << ","
              << MeanResultField(group, &QueryResult::gnn_message_cycles)
              << ","
              << MeanResultField(group, &QueryResult::cached_kv_cycles) << ","
              << MeanResultField(group, &QueryResult::reducer_cycles) << ","
              << MeanResultField(group, &QueryResult::scheduling_cycles) << ","
              << MeanResultField(group, &QueryResult::q8k8_vdot_cycles) << ","
              << MeanResultField(group, &QueryResult::gnn_score_scale_cycles)
              << ","
              << MeanResultField(group, &QueryResult::p8v8_vmul_cycles) << ","
              << MeanResultField(group, &QueryResult::gnn_value_scale_cycles)
              << ","
              << MeanResultField(group, &QueryResult::local_vadd_cycles) << ","
              << MeanResultField(group, &QueryResult::q8k2_lut_cycles) << ","
              << MeanResultField(group, &QueryResult::p8v2_lut_cycles) << ","
              << MeanResultField(group, &QueryResult::cached_kv_scale_cycles)
              << ","
              << MeanResultField(group, &QueryResult::pc_reduce_cycles) << ","
              << MeanResultField(group, &QueryResult::global_reduce_cycles)
              << ","
              << MeanResultField(group, &QueryResult::h100_cache_read_cycles)
              << ","
              << MeanResultField(group, &QueryResult::h100_int2_unpack_cycles)
              << ","
              << MeanResultField(group,
                                 &QueryResult::h100_scale_dequant_cycles)
              << ","
              << MeanResultField(
                     group, &QueryResult::h100_layout_conversion_cycles)
              << ","
              << MeanResultField(
                     group,
                     &QueryResult::h100_irregular_gather_penalty_cycles)
              << ","
              << MeanResultField(
                     group, &QueryResult::h100_small_batch_penalty_cycles)
              << "," << dominant_bottleneck.first << ","
              << dominant_bottleneck.second << ","
              << MeanResultField(group, &QueryResult::bottleneck_cycles) << ","
              << MeanResultField(group, &QueryResult::bottleneck_fraction)
              << ","
              << MeanGraphField(group, &GraphSanity::duplicate_edge_count)
              << ","
              << MeanGraphField(group, &GraphSanity::duplicate_edge_ratio)
              << ","
              << MeanGraphField(group, &GraphSanity::unique_src_dst_count)
              << "," << MeanGraphField(group, &GraphSanity::in_degree_mean)
              << "," << MeanGraphField(group, &GraphSanity::in_degree_p50)
              << "," << MeanGraphField(group, &GraphSanity::in_degree_p95)
              << "," << MeanGraphField(group, &GraphSanity::in_degree_max)
              << "," << MeanGraphField(group, &GraphSanity::out_degree_mean)
              << "," << MeanGraphField(group, &GraphSanity::out_degree_p50)
              << "," << MeanGraphField(group, &GraphSanity::out_degree_p95)
              << "," << MeanGraphField(group, &GraphSanity::out_degree_max)
              << "," << MeanGraphField(group, &GraphSanity::dst_degree_hist_0)
              << "," << MeanGraphField(group, &GraphSanity::dst_degree_hist_1)
              << ","
              << MeanGraphField(group, &GraphSanity::dst_degree_hist_2_3)
              << ","
              << MeanGraphField(group, &GraphSanity::dst_degree_hist_4_7)
              << ","
              << MeanGraphField(group, &GraphSanity::dst_degree_hist_8_15)
              << ","
              << MeanGraphField(group, &GraphSanity::dst_degree_hist_16_31)
              << ","
              << MeanGraphField(group, &GraphSanity::dst_degree_hist_32_63)
              << ","
              << MeanGraphField(group, &GraphSanity::dst_degree_hist_64_plus)
              << ","
              << MeanGraphField(group, &GraphSanity::unique_source_banks)
              << ","
              << MeanPlacementField(
                     group,
                     &PlacementValidation::mapping_difference_ratio_vs_hash)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::edge_active_banks)
              << ","
              << MeanPlacementField(
                     group, &PlacementValidation::edge_active_pseudo_channels)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::bank_edge_count_mean)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::bank_edge_count_p50)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::bank_edge_count_p95)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::bank_edge_count_max)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::pc_edge_count_mean)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::pc_edge_count_p50)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::pc_edge_count_p95)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::pc_edge_count_max)
              << ","
              << MeanPlacementField(
                     group, &PlacementValidation::sharded_destination_count)
              << ","
              << MeanPlacementField(group,
                                    &PlacementValidation::total_destination_shards)
              << ","
              << MeanPlacementField(
                     group,
                     &PlacementValidation::average_shards_per_destination)
              << ","
              << MeanPlacementField(
                     group, &PlacementValidation::max_shards_per_destination)
              << "," << MeanResultField(group, &QueryResult::compute_cycles)
              << ","
              << MeanResultField(group, &QueryResult::communication_cycles)
              << ","
              << MeanResultField(group, &QueryResult::q_broadcast_cycles)
              << ","
              << MeanResultField(
                     group, &QueryResult::bank_to_pc_communication_cycles)
              << ","
              << MeanResultField(
                     group, &QueryResult::pc_to_global_communication_cycles)
              << ","
              << MeanResultField(
                     group, &QueryResult::global_to_npu_communication_cycles)
              << ","
              << MeanResultField(group, &QueryResult::critical_path_cycles)
              << ","
              << MeanResultField(group,
                                 &QueryResult::critical_path_latency_ns)
              << ","
              << MeanResultField(group,
                                 &QueryResult::traffic_stall_fraction)
              << ","
              << MeanResultField(
                     group, &QueryResult::q_broadcast_critical_bank_bytes)
              << ","
              << MeanResultField(group,
                                 &QueryResult::bank_to_pc_total_bytes)
              << ","
              << MeanResultField(
                     group, &QueryResult::bank_to_pc_critical_bank_bytes)
              << ","
              << MeanResultField(
                     group, &QueryResult::bank_to_pc_critical_pc_bytes)
              << ","
              << MeanResultField(group,
                                 &QueryResult::pc_to_global_total_bytes)
              << ","
              << MeanResultField(
                     group, &QueryResult::pc_to_global_critical_pc_bytes)
              << ","
              << MeanResultField(group, &QueryResult::global_to_npu_bytes)
              << ","
              << MeanResultField(group,
                                 &QueryResult::pc_reducer_input_groups)
              << ","
              << MeanResultField(group,
                                 &QueryResult::pc_reducer_output_groups)
              << ","
              << MeanResultField(group,
                                 &QueryResult::global_reducer_input_groups)
              << ","
              << MeanResultField(group,
                                 &QueryResult::global_reducer_output_groups)
              << "\n";
        }
      }
    }
  }
}

void PrintSummary(const SimulationConfig& config,
                  const std::vector<QueryResult>& results) {
  std::cout << "Analytical PIM Patch 4 communication-aware run\n";
  std::cout << "workloads=" << config.suite.workloads.size()
            << ", placements=" << config.placements.size()
            << ", baselines=" << config.baselines.size()
            << ", queries_per_workload="
            << config.suite.num_queries_per_workload << "\n";
  for (const auto& workload : config.suite.workloads) {
    for (const auto& placement : config.placements) {
      std::vector<double> local_reduction;
      std::vector<double> latency;
      std::vector<double> compute;
      std::vector<double> communication;
      std::vector<double> reducer;
      std::vector<double> traffic_stall;
      std::vector<const QueryResult*> group;
      for (const auto& r : results) {
        if (r.workload == workload.name && r.placement == placement &&
            r.baseline == kPIMLocalCombine && r.memory_token_tile == 4) {
          local_reduction.push_back(
              r.diagnosis.local_combine_reduction_ratio);
          latency.push_back(r.latency_ns);
          compute.push_back(r.compute_cycles);
          communication.push_back(r.communication_cycles);
          reducer.push_back(r.reducer_cycles);
          traffic_stall.push_back(r.traffic_stall_fraction);
          group.push_back(&r);
        }
      }
      if (!latency.empty()) {
        std::cout << workload.name << "/" << placement
                  << " local-combine T=4 mean_latency_ns=" << std::fixed
                  << std::setprecision(2) << Mean(latency)
                  << " compute_cycles=" << Mean(compute)
                  << " communication_cycles=" << Mean(communication)
                  << " reducer_cycles=" << Mean(reducer)
                  << " traffic_stall=" << std::setprecision(4)
                  << Mean(traffic_stall)
                  << " mean_local_reduction=" << std::setprecision(4)
                  << Mean(local_reduction)
                  << " bottleneck=" << DominantBottleneck(group).first << "\n";
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
