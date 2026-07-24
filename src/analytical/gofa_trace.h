#ifndef LLM_SYSTEM_ANALYTICAL_GOFA_TRACE_H_
#define LLM_SYSTEM_ANALYTICAL_GOFA_TRACE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llm_system {
namespace analytical {

struct GOFAShape {
  std::vector<int64_t> dims;

  int64_t Numel() const;
  int64_t Dim(size_t index) const;
};

struct GOFATraceTaskConfig {
  std::string name;
  std::string directory;
};

struct GOFATraceLoadConfig {
  std::string root;
  std::vector<GOFATraceTaskConfig> tasks;
  std::vector<std::string> splits{"validation", "test"};
  int expected_queries_per_split = 100;
  int expected_batch_size = 1;
  std::string expected_cache_mode = "memory_kv";
  std::string expected_selection_policy = "target_1hop";
  int expected_memory_bits = 4;
  int expected_key_bits = 2;
  int expected_value_bits = 2;
  int expected_hidden_size = 4096;
  int expected_attention_heads = 32;
  int expected_kv_heads = 8;
  int expected_head_dim = 128;
  int expected_memory_tokens = 128;
  std::vector<int> expected_suffix_layer_ids{26, 27, 28, 29, 30, 31};
  bool validate_summary = true;
};

struct GOFATraceModel {
  int hidden_size = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  int memory_tokens = 0;
  int gnn_start_layer = 0;
  std::vector<int> suffix_layer_ids;
};

struct GOFATraceKVLayer {
  int layer_id = 0;
  GOFAShape key_shape;
  GOFAShape value_shape;
  GOFAShape key_scale_shape;
  GOFAShape value_scale_shape;
};

struct GOFATraceCacheItem {
  int item_index = 0;
  std::string item_type;
  std::string cache_key;
  bool cache_eligible = false;
  bool is_nog = false;
  int sequence_length = 0;
  int valid_text_tokens = 0;
  int memory_bits = 0;
  int key_bits = 0;
  int value_bits = 0;
  GOFAShape memory_shape;
  GOFAShape memory_scale_shape;
  std::vector<GOFATraceKVLayer> layers;
  int64_t memory_logical_bytes = 0;
  int64_t full_key_logical_bytes = 0;
  int64_t full_value_logical_bytes = 0;
};

struct GOFATraceRuntimeItem {
  int runtime_item_index = 0;
  int item_index = 0;
  GOFAShape q_projection_input_shape;
  GOFAShape q_projection_output_shape;
  GOFAShape qk_shape;
  GOFAShape softmax_probability_shape;
  GOFAShape pv_shape;
  GOFAShape attention_output_shape;
  GOFAShape mlp_input_shape;
  GOFAShape mlp_output_shape;
};

struct GOFATraceRuntimeLayer {
  int layer_id = 0;
  std::vector<GOFATraceRuntimeItem> items;
  GOFAShape gnn_node_input_shape;
  GOFAShape gnn_node_output_shape;
  GOFAShape gnn_edge_input_shape;
};

struct GOFATraceTraffic {
  std::string byte_accounting;
  int64_t memory_cache_bytes = 0;
  int64_t selected_key_bytes = 0;
  int64_t selected_value_bytes = 0;
  int64_t full_key_bytes = 0;
  int64_t full_value_bytes = 0;
  int64_t edge_cache_bytes = 0;
  int64_t persistent_cache_bytes = 0;
  int64_t runtime_loaded_cache_bytes = 0;
  int64_t persistent_scale_bytes = 0;
  int64_t runtime_loaded_scale_bytes = 0;
  int64_t persistent_gather_index_metadata_bytes = 0;
  int64_t runtime_gather_index_metadata_bytes = 0;
  int64_t persistent_total_bytes = 0;
  int64_t runtime_loaded_total_bytes = 0;
  int nog_online_item_count = 0;
};

struct GOFATraceQuery {
  int trace_order = 0;
  std::string query_id;
  std::string task_name;
  std::string dataset_name;
  std::string split;
  std::string source_path;
  int runtime_query_index = 0;
  int batch_size = 0;
  std::string cache_mode;
  std::string selection_policy;
  GOFATraceModel model;

  int num_graph_nodes = 0;
  int num_node_text_items = 0;
  int num_edge_text_items = 0;
  int num_structural_edges = 0;
  std::vector<int> target_indices;
  std::vector<int> question_indices;
  std::vector<int> node_map;
  std::vector<int> edge_map;
  std::vector<int> edge_sources;
  std::vector<int> edge_destinations;

  std::vector<GOFATraceCacheItem> cache_items;
  std::vector<int> eligible_item_indices;
  std::vector<int> selected_key_item_indices;
  std::vector<int> selected_value_item_indices;
  std::vector<std::vector<int>> selected_key_items_by_layer;
  std::vector<std::vector<int>> selected_value_items_by_layer;
  std::vector<int> runtime_item_order;
  std::vector<GOFATraceRuntimeLayer> runtime_layers;
  GOFATraceTraffic traffic;

  int total_item_count = 0;
  int cacheable_item_count = 0;
  int nog_count = 0;
  double selected_key_ratio = 0.0;
  double selected_value_ratio = 0.0;
  double selected_kv_ratio = 0.0;
};

struct GOFATraceLoadResult {
  std::vector<GOFATraceQuery> queries;
  int validation_query_count = 0;
  int test_query_count = 0;
};

GOFATraceLoadResult LoadAndValidateGOFATraces(
    const GOFATraceLoadConfig &config);

}  // namespace analytical
}  // namespace llm_system

#endif  // LLM_SYSTEM_ANALYTICAL_GOFA_TRACE_H_
