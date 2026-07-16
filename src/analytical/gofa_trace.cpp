#include "analytical/gofa_trace.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace llm_system {
namespace analytical {

int64_t GOFAShape::Numel() const {
  if (dims.empty()) {
    return 0;
  }
  return std::accumulate(dims.begin(), dims.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

int64_t GOFAShape::Dim(size_t index) const {
  if (index >= dims.size()) {
    throw std::out_of_range("GOFA shape dimension index out of range");
  }
  return dims[index];
}

namespace {

namespace fs = std::filesystem;

[[noreturn]] void Fail(const std::string &source, const std::string &message) {
  throw std::runtime_error(source + ": " + message);
}

void Require(bool condition, const std::string &source,
             const std::string &message) {
  if (!condition) {
    Fail(source, message);
  }
}

YAML::Node RequireMap(const YAML::Node &parent, const std::string &key,
                      const std::string &source) {
  const YAML::Node node = parent[key];
  Require(node && node.IsMap(), source, key + " must be an object");
  return node;
}

YAML::Node RequireSequence(const YAML::Node &parent, const std::string &key,
                           const std::string &source) {
  const YAML::Node node = parent[key];
  Require(node && node.IsSequence(), source, key + " must be an array");
  return node;
}

template <typename T>
T Required(const YAML::Node &parent, const std::string &key,
           const std::string &source) {
  const YAML::Node node = parent[key];
  Require(static_cast<bool>(node), source, "missing field " + key);
  try {
    return node.as<T>();
  } catch (const YAML::Exception &ex) {
    Fail(source, "invalid field " + key + ": " + ex.what());
  }
}

std::vector<int> IntVector(const YAML::Node &node, const std::string &source) {
  Require(node && node.IsSequence(), source, "expected integer array");
  std::vector<int> result;
  result.reserve(node.size());
  for (size_t index = 0; index < node.size(); ++index) {
    try {
      result.push_back(node[index].as<int>());
    } catch (const YAML::Exception &ex) {
      Fail(source, "invalid integer at index " + std::to_string(index) + ": " +
                       ex.what());
    }
  }
  return result;
}

std::vector<bool> BoolVector(const YAML::Node &node,
                             const std::string &source) {
  Require(node && node.IsSequence(), source, "expected boolean array");
  std::vector<bool> result;
  result.reserve(node.size());
  for (size_t index = 0; index < node.size(); ++index) {
    try {
      result.push_back(node[index].as<bool>());
    } catch (const YAML::Exception &ex) {
      Fail(source, "invalid boolean at index " + std::to_string(index) + ": " +
                       ex.what());
    }
  }
  return result;
}

void RequireNoCachePayload(const YAML::Node &node, const std::string &source) {
  static const std::set<std::string> forbidden = {"q", "q_packed", "tensor",
                                                  "scale", "zero_point"};
  if (node.IsMap()) {
    for (const auto &entry : node) {
      const std::string key = entry.first.as<std::string>();
      Require(forbidden.count(key) == 0, source,
              "cache inventory contains forbidden tensor payload field " + key);
      RequireNoCachePayload(entry.second, source + "." + key);
    }
  } else if (node.IsSequence()) {
    for (size_t index = 0; index < node.size(); ++index) {
      RequireNoCachePayload(node[index],
                            source + "[" + std::to_string(index) + "]");
    }
  }
}

GOFAShape Shape(const YAML::Node &node, const std::string &source) {
  Require(node && node.IsSequence() && node.size() > 0, source,
          "shape must be a non-empty array");
  GOFAShape result;
  result.dims.reserve(node.size());
  for (size_t index = 0; index < node.size(); ++index) {
    int64_t dim = 0;
    try {
      dim = node[index].as<int64_t>();
    } catch (const YAML::Exception &ex) {
      Fail(source, "invalid shape dimension: " + std::string(ex.what()));
    }
    Require(dim >= 0, source, "shape dimensions must be non-negative");
    result.dims.push_back(dim);
  }
  return result;
}

bool SameShape(const GOFAShape &lhs, const GOFAShape &rhs) {
  return lhs.dims == rhs.dims;
}

int64_t QuantizedBytes(const GOFAShape &shape, int bits) {
  return (shape.Numel() * bits + 7) / 8;
}

std::string NormalizeSplit(const std::string &split,
                           const std::string &source) {
  if (split == "val" || split == "valid" || split == "validation") {
    return "validation";
  }
  if (split == "test") {
    return "test";
  }
  Fail(source, "unsupported split " + split);
}

bool Contains(const std::vector<int> &values, int value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

void RequireUniqueIndices(const std::vector<int> &values, int upper_bound,
                          const std::string &source) {
  std::set<int> unique;
  for (int value : values) {
    Require(value >= 0 && value < upper_bound, source,
            "item index out of range: " + std::to_string(value));
    Require(unique.insert(value).second, source,
            "duplicate item index: " + std::to_string(value));
  }
}

void RequireSorted(const std::vector<int> &values, const std::string &source) {
  Require(std::is_sorted(values.begin(), values.end()), source,
          "item indices must be sorted");
}

std::vector<int> SetIntersection(const std::vector<int> &lhs,
                                 const std::vector<int> &rhs) {
  std::vector<int> result;
  std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                        std::back_inserter(result));
  return result;
}

std::vector<int> SetDifference(const std::vector<int> &lhs,
                               const std::vector<int> &rhs) {
  std::vector<int> result;
  std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                      std::back_inserter(result));
  return result;
}

std::vector<int> SetUnion(const std::vector<int> &lhs,
                          const std::vector<int> &rhs) {
  std::vector<int> result;
  std::set_union(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                 std::back_inserter(result));
  return result;
}

double Mean(const std::vector<double> &values) {
  if (values.empty()) {
    return 0.0;
  }
  return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double Percentile(std::vector<double> values, double probability) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double position = (values.size() - 1) * probability;
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  if (lower == upper) {
    return values[lower];
  }
  const double weight = position - lower;
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void RequireNear(double actual, double expected, const std::string &source,
                 const std::string &field) {
  const double tolerance = std::max(1e-9 * std::abs(expected), 1e-12);
  Require(std::abs(actual - expected) <= tolerance, source,
          field + " mismatch: actual=" + std::to_string(actual) +
              ", expected=" + std::to_string(expected));
}

const GOFATraceKVLayer &FindKVLayer(const GOFATraceCacheItem &item,
                                    int layer_id, const std::string &source) {
  for (const auto &layer : item.layers) {
    if (layer.layer_id == layer_id) {
      return layer;
    }
  }
  Fail(source, "cache item " + std::to_string(item.item_index) +
                   " has no layer " + std::to_string(layer_id));
}

std::vector<int> LayerSelection(const YAML::Node &by_layer, int layer_id,
                                int total_items, const std::string &source) {
  const std::string key = std::to_string(layer_id);
  Require(static_cast<bool>(by_layer[key]), source,
          "missing layer selection " + key);
  std::vector<int> values = IntVector(by_layer[key], source + "." + key);
  RequireUniqueIndices(values, total_items, source + "." + key);
  return values;
}

GOFATraceQuery LoadQuery(const fs::path &path, int trace_order,
                         const GOFATraceLoadConfig &config) {
  YAML::Node trace;
  try {
    trace = YAML::LoadFile(path.string());
  } catch (const YAML::Exception &ex) {
    Fail(path.string(), "invalid JSON: " + std::string(ex.what()));
  }
  const std::string source = path.string();
  Require(trace.IsMap(), source, "top-level value must be an object");
  Require(Required<std::string>(trace, "trace_format", source) ==
              "gofa_query_trace",
          source, "trace_format must be gofa_query_trace");
  Require(Required<int>(trace, "trace_version", source) == 1, source,
          "trace_version must be 1");
  Required<std::string>(trace, "created_at", source);
  Required<std::string>(trace, "repository_commit_sha", source);

  GOFATraceQuery query;
  query.trace_order = trace_order;
  query.source_path = source;
  query.query_id = Required<std::string>(trace, "query_id", source);
  query.task_name = Required<std::string>(trace, "task_name", source);
  query.dataset_name = Required<std::string>(trace, "dataset_name", source);
  query.split =
      NormalizeSplit(Required<std::string>(trace, "split", source), source);
  query.runtime_query_index =
      Required<int>(trace, "runtime_query_index", source);
  query.batch_size = Required<int>(trace, "batch_size", source);
  query.cache_mode = Required<std::string>(trace, "cache_mode", source);
  Required<std::string>(trace, "cache_tag", source);
  Require(query.batch_size == config.expected_batch_size, source,
          "batch_size mismatch");
  Require(query.cache_mode == config.expected_cache_mode, source,
          "cache_mode mismatch");

  const YAML::Node model = RequireMap(trace, "model_configuration", source);
  query.model.hidden_size = Required<int>(model, "hidden_size", source);
  query.model.num_attention_heads =
      Required<int>(model, "num_attention_heads", source);
  query.model.num_key_value_heads =
      Required<int>(model, "num_key_value_heads", source);
  query.model.head_dim = Required<int>(model, "head_dim", source);
  query.model.memory_tokens = Required<int>(model, "mem_size", source);
  query.model.gnn_start_layer = Required<int>(model, "gnn_start_layer", source);
  query.model.suffix_layer_ids =
      IntVector(RequireSequence(model, "suffix_layer_ids", source), source);
  Require(query.model.hidden_size == config.expected_hidden_size, source,
          "hidden_size mismatch");
  Require(query.model.num_attention_heads == config.expected_attention_heads,
          source, "num_attention_heads mismatch");
  Require(query.model.num_key_value_heads == config.expected_kv_heads, source,
          "num_key_value_heads mismatch");
  Require(query.model.head_dim == config.expected_head_dim, source,
          "head_dim mismatch");
  Require(query.model.memory_tokens == config.expected_memory_tokens, source,
          "mem_size mismatch");
  Require(query.model.suffix_layer_ids == config.expected_suffix_layer_ids,
          source, "suffix_layer_ids mismatch");
  Require(query.model.suffix_layer_ids.front() == query.model.gnn_start_layer,
          source, "gnn_start_layer mismatch");

  const YAML::Node graph = RequireMap(trace, "query_graph_structure", source);
  query.num_graph_nodes = Required<int>(graph, "num_graph_nodes", source);
  query.num_node_text_items =
      Required<int>(graph, "num_node_text_items", source);
  query.num_edge_text_items =
      Required<int>(graph, "num_edge_text_items", source);
  query.num_structural_edges =
      Required<int>(graph, "num_structural_edges", source);
  Require(query.num_graph_nodes > 0 && query.num_node_text_items > 0 &&
              query.num_edge_text_items >= 0 && query.num_structural_edges >= 0,
          source, "invalid graph or inventory counts");
  query.target_indices = IntVector(
      RequireSequence(graph, "target_index", source), source + ".target_index");
  query.question_indices =
      IntVector(RequireSequence(graph, "question_index", source),
                source + ".question_index");
  query.node_map = IntVector(RequireSequence(graph, "node_map", source),
                             source + ".node_map");
  query.edge_map = IntVector(RequireSequence(graph, "edge_map", source),
                             source + ".edge_map");
  Require(!query.target_indices.empty(), source,
          "target_index must not be empty");
  Require(Required<std::string>(graph, "node_map_semantics", source) ==
              "graph_local_node_to_encoder_text_item",
          source, "node_map_semantics mismatch");
  Require(Required<std::string>(graph, "edge_map_semantics", source) ==
              "structural_edge_to_edge_text_item",
          source, "edge_map_semantics mismatch");
  Require(static_cast<int>(query.node_map.size()) == query.num_graph_nodes,
          source, "node_map length mismatch");
  Require(static_cast<int>(query.edge_map.size()) == query.num_structural_edges,
          source, "edge_map length mismatch");
  for (int index : query.node_map) {
    Require(index >= 0 && index < query.num_node_text_items, source,
            "node_map item index out of range");
  }
  for (int index : query.edge_map) {
    Require(index >= 0 && index < query.num_edge_text_items, source,
            "edge_map item index out of range");
  }
  Require(query.question_indices.size() == 1, source,
          "formal trace must contain one NOG question index");
  const std::vector<int> nog_indices =
      IntVector(RequireSequence(graph, "nog_local_indices", source), source);
  Require(nog_indices == query.question_indices, source,
          "NOG local indices mismatch");
  Require(Required<int>(graph, "nog_local_index", source) ==
              query.question_indices.front(),
          source, "NOG local index mismatch");
  const std::vector<int> batch =
      IntVector(RequireSequence(graph, "batch", source), source);
  const std::vector<int> ptr =
      IntVector(RequireSequence(graph, "ptr", source), source);
  Require(batch == std::vector<int>(query.num_graph_nodes, 0), source,
          "batch vector mismatch");
  Require(ptr == std::vector<int>({0, query.num_graph_nodes}), source,
          "ptr vector mismatch");
  const YAML::Node edge_index = RequireSequence(graph, "edge_index", source);
  Require(edge_index.size() == 2, source, "edge_index must have two rows");
  query.edge_sources = IntVector(edge_index[0], source + ".edge_index[0]");
  query.edge_destinations = IntVector(edge_index[1], source + ".edge_index[1]");
  Require(static_cast<int>(query.edge_sources.size()) ==
                  query.num_structural_edges &&
              query.edge_sources.size() == query.edge_destinations.size(),
          source, "edge_index length mismatch");
  RequireUniqueIndices(query.target_indices, query.num_graph_nodes,
                       source + ".target_index");
  for (size_t index = 0; index < query.edge_sources.size(); ++index) {
    Require(query.edge_sources[index] >= 0 &&
                query.edge_sources[index] < query.num_graph_nodes &&
                query.edge_destinations[index] >= 0 &&
                query.edge_destinations[index] < query.num_graph_nodes,
            source, "edge_index node out of range");
  }

  const YAML::Node inventory =
      RequireSequence(trace, "cache_item_inventory", source);
  RequireNoCachePayload(inventory, source + ".cache_item_inventory");
  Require(static_cast<int>(inventory.size()) ==
              query.num_node_text_items + query.num_edge_text_items,
          source, "cache inventory length mismatch");
  int nog_count = 0;
  for (size_t item_index = 0; item_index < inventory.size(); ++item_index) {
    const YAML::Node node = inventory[item_index];
    Require(node.IsMap(), source, "cache inventory item must be an object");
    GOFATraceCacheItem item;
    item.item_index = Required<int>(node, "item_index", source);
    Require(item.item_index == static_cast<int>(item_index), source,
            "cache item_index mismatch");
    item.item_type = Required<std::string>(node, "item_type", source);
    item.cache_key = Required<std::string>(node, "cache_key", source);
    Require(!item.cache_key.empty(), source, "cache key must not be empty");
    item.cache_eligible = Required<bool>(node, "cache_eligible", source);
    item.is_nog = Required<bool>(node, "is_nog", source);
    item.sequence_length = Required<int>(node, "sequence_length", source);
    item.valid_text_tokens = Required<int>(node, "text_length", source);
    item.memory_bits = Required<int>(node, "memory_bits", source);
    item.key_bits = Required<int>(node, "key_bits", source);
    item.value_bits = Required<int>(node, "value_bits", source);
    Require(item.memory_bits == config.expected_memory_bits &&
                item.key_bits == config.expected_key_bits &&
                item.value_bits == config.expected_value_bits,
            source, "cache item precision mismatch");
    Require(item.valid_text_tokens >= 0 &&
                item.sequence_length >= item.valid_text_tokens,
            source, "invalid valid/stored token counts");
    if (item.is_nog) {
      ++nog_count;
      Require(item.item_type == "NOG" && !item.cache_eligible, source,
              "NOG cache eligibility mismatch");
    } else {
      const std::string expected_type =
          item.item_index < query.num_node_text_items ? "node" : "edge";
      Require(item.item_type == expected_type && item.cache_eligible, source,
              "cache item type/eligibility mismatch");
    }
    item.memory_shape = Shape(node["memory_shape"], source + ".memory_shape");
    Require(item.memory_shape.dims ==
                std::vector<int64_t>(
                    {query.model.memory_tokens, query.model.hidden_size}),
            source, "memory shape mismatch");
    item.memory_logical_bytes =
        QuantizedBytes(item.memory_shape, item.memory_bits);
    const YAML::Node layers = RequireSequence(node, "text_kv_shapes", source);
    Require(layers.size() == query.model.suffix_layer_ids.size(), source,
            "text_kv layer count mismatch");
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
      GOFATraceKVLayer layer;
      layer.layer_id = Required<int>(layers[layer_index], "layer_id", source);
      Require(layer.layer_id == query.model.suffix_layer_ids[layer_index],
              source, "text_kv layer id mismatch");
      layer.key_shape =
          Shape(layers[layer_index]["key_shape"], source + ".key_shape");
      layer.value_shape =
          Shape(layers[layer_index]["value_shape"], source + ".value_shape");
      const std::vector<int64_t> expected_shape = {
          query.model.num_key_value_heads, item.valid_text_tokens,
          query.model.head_dim};
      Require(layer.key_shape.dims == expected_shape &&
                  layer.value_shape.dims == expected_shape,
              source, "text_kv shape mismatch");
      item.full_key_logical_bytes +=
          QuantizedBytes(layer.key_shape, item.key_bits);
      item.full_value_logical_bytes +=
          QuantizedBytes(layer.value_shape, item.value_bits);
      item.layers.push_back(std::move(layer));
    }
    query.cache_items.push_back(std::move(item));
  }
  Require(nog_count == 1, source, "formal trace must contain one NOG item");
  const int question_node = query.question_indices.front();
  Require(question_node >= 0 && question_node < query.num_graph_nodes, source,
          "question index out of range");
  Require(query.cache_items[query.node_map[question_node]].is_nog, source,
          "question node must map to NOG cache item");

  const YAML::Node access = RequireMap(trace, "selective_kv_access", source);
  query.selection_policy = Required<std::string>(access, "policy_name", source);
  Require(query.selection_policy == config.expected_selection_policy, source,
          "selection policy mismatch");
  const int total_items = query.cache_items.size();
  query.eligible_item_indices = IntVector(
      RequireSequence(access, "eligible_item_indices", source), source);
  query.selected_key_item_indices = IntVector(
      RequireSequence(access, "selected_key_item_indices", source), source);
  query.selected_value_item_indices = IntVector(
      RequireSequence(access, "selected_value_item_indices", source), source);
  RequireUniqueIndices(query.eligible_item_indices, total_items, source);
  RequireUniqueIndices(query.selected_key_item_indices, total_items, source);
  RequireUniqueIndices(query.selected_value_item_indices, total_items, source);
  RequireSorted(query.eligible_item_indices, source + ".eligible_item_indices");
  RequireSorted(query.selected_key_item_indices,
                source + ".selected_key_item_indices");
  RequireSorted(query.selected_value_item_indices,
                source + ".selected_value_item_indices");
  std::vector<int> expected_eligible;
  for (const auto &item : query.cache_items) {
    if (item.cache_eligible) {
      expected_eligible.push_back(item.item_index);
    }
  }
  Require(query.eligible_item_indices == expected_eligible, source,
          "eligible item inventory mismatch");
  for (int index : query.selected_key_item_indices) {
    Require(Contains(query.eligible_item_indices, index), source,
            "selected K item is not cache eligible");
  }
  for (int index : query.selected_value_item_indices) {
    Require(Contains(query.eligible_item_indices, index), source,
            "selected V item is not cache eligible");
  }
  const std::vector<bool> key_mask =
      BoolVector(RequireSequence(access, "key_item_mask", source),
                 source + ".key_item_mask");
  const std::vector<bool> value_mask =
      BoolVector(RequireSequence(access, "value_item_mask", source),
                 source + ".value_item_mask");
  Require(static_cast<int>(key_mask.size()) == total_items &&
              static_cast<int>(value_mask.size()) == total_items,
          source, "K/V item mask length mismatch");
  std::vector<int> key_mask_indices;
  std::vector<int> value_mask_indices;
  for (int index = 0; index < total_items; ++index) {
    if (key_mask[index]) {
      key_mask_indices.push_back(index);
    }
    if (value_mask[index]) {
      value_mask_indices.push_back(index);
    }
  }
  Require(key_mask_indices == query.selected_key_item_indices, source,
          "key_item_mask does not match selected K items");
  Require(value_mask_indices == query.selected_value_item_indices, source,
          "value_item_mask does not match selected V items");

  const std::vector<int> complete_items = SetIntersection(
      query.selected_key_item_indices, query.selected_value_item_indices);
  const std::vector<int> key_only_items = SetDifference(
      query.selected_key_item_indices, query.selected_value_item_indices);
  const std::vector<int> value_only_items = SetDifference(
      query.selected_value_item_indices, query.selected_key_item_indices);
  std::vector<int> all_item_indices(total_items);
  std::iota(all_item_indices.begin(), all_item_indices.end(), 0);
  const std::vector<int> skipped_items = SetDifference(
      all_item_indices, SetUnion(query.selected_key_item_indices,
                                 query.selected_value_item_indices));
  Require(IntVector(RequireSequence(access, "complete_kv_item_indices", source),
                    source) == complete_items,
          source, "complete K/V items mismatch");
  Require(IntVector(RequireSequence(access, "k_only_item_indices", source),
                    source) == key_only_items,
          source, "K-only items mismatch");
  Require(IntVector(RequireSequence(access, "v_only_item_indices", source),
                    source) == value_only_items,
          source, "V-only items mismatch");
  Require(IntVector(RequireSequence(access, "skipped_item_indices", source),
                    source) == skipped_items,
          source, "skipped items mismatch");
  const YAML::Node by_key =
      RequireMap(access, "effective_key_items_by_layer", source);
  const YAML::Node by_value =
      RequireMap(access, "effective_value_items_by_layer", source);
  for (int layer_id : query.model.suffix_layer_ids) {
    query.selected_key_items_by_layer.push_back(LayerSelection(
        by_key, layer_id, total_items, source + ".effective_key"));
    query.selected_value_items_by_layer.push_back(LayerSelection(
        by_value, layer_id, total_items, source + ".effective_value"));
  }
  for (const auto &values : query.selected_key_items_by_layer) {
    Require(values == query.selected_key_item_indices, source,
            "K selection must be shared across suffix layers");
  }
  for (const auto &values : query.selected_value_items_by_layer) {
    Require(values == query.selected_value_item_indices, source,
            "V selection must be shared across suffix layers");
  }
  Require(Required<bool>(access, "selection_is_shared_across_suffix_layers",
                         source),
          source, "selection sharing flag must be true");

  const YAML::Node runtime =
      RequireMap(trace, "runtime_operation_shapes", source);
  query.runtime_item_order =
      IntVector(RequireSequence(runtime, "item_order", source), source);
  std::vector<int> expected_runtime_item_order = query.node_map;
  for (int item_index = query.num_node_text_items; item_index < total_items;
       ++item_index) {
    expected_runtime_item_order.push_back(item_index);
  }
  Require(query.runtime_item_order == expected_runtime_item_order, source,
          "runtime item_order does not match node_map plus edge inventory");
  const YAML::Node runtime_layers = RequireSequence(runtime, "layers", source);
  Require(runtime_layers.size() == query.model.suffix_layer_ids.size(), source,
          "runtime layer count mismatch");
  for (size_t layer_index = 0; layer_index < runtime_layers.size();
       ++layer_index) {
    const YAML::Node layer_node = runtime_layers[layer_index];
    GOFATraceRuntimeLayer layer;
    layer.layer_id = Required<int>(layer_node, "layer_id", source);
    Require(layer.layer_id == query.model.suffix_layer_ids[layer_index], source,
            "runtime layer id mismatch");
    const YAML::Node items = RequireSequence(layer_node, "items", source);
    Require(items.size() == query.runtime_item_order.size(), source,
            "runtime item count mismatch");
    for (size_t runtime_index = 0; runtime_index < items.size();
         ++runtime_index) {
      GOFATraceRuntimeItem item;
      item.runtime_item_index =
          Required<int>(items[runtime_index], "runtime_item_index", source);
      item.item_index =
          Required<int>(items[runtime_index], "item_index", source);
      Require(item.runtime_item_index == static_cast<int>(runtime_index) &&
                  item.item_index == query.runtime_item_order[runtime_index],
              source, "runtime item mapping mismatch");
      Require(item.item_index >= 0 && item.item_index < total_items, source,
              "runtime source item out of range");
      item.q_projection_input_shape =
          Shape(items[runtime_index]["q_projection_input_shape"], source);
      item.q_projection_output_shape =
          Shape(items[runtime_index]["q_projection_output_shape"], source);
      item.qk_shape = Shape(items[runtime_index]["qk_shape"], source);
      item.softmax_probability_shape =
          Shape(items[runtime_index]["softmax_probability_shape"], source);
      item.pv_shape = Shape(items[runtime_index]["pv_shape"], source);
      item.attention_output_shape =
          Shape(items[runtime_index]["attention_output_shape"], source);
      item.mlp_input_shape =
          Shape(items[runtime_index]["mlp_input_shape"], source);
      item.mlp_output_shape =
          Shape(items[runtime_index]["mlp_output_shape"], source);
      const std::vector<int64_t> hidden_shape = {1, query.model.memory_tokens,
                                                 query.model.hidden_size};
      Require(item.q_projection_input_shape.dims == hidden_shape &&
                  item.q_projection_output_shape.dims == hidden_shape &&
                  item.attention_output_shape.dims == hidden_shape &&
                  item.mlp_input_shape.dims == hidden_shape &&
                  item.mlp_output_shape.dims == hidden_shape,
              source, "runtime hidden-state shape mismatch");
      Require(SameShape(item.qk_shape, item.softmax_probability_shape), source,
              "QK and softmax shapes mismatch");
      Require(item.qk_shape.dims.size() == 4 && item.qk_shape.Dim(0) == 1 &&
                  item.qk_shape.Dim(1) == query.model.num_attention_heads &&
                  item.qk_shape.Dim(2) == query.model.memory_tokens &&
                  item.qk_shape.Dim(3) >= query.model.memory_tokens,
              source, "runtime QK shape mismatch");
      const bool has_selected_key = Contains(
          query.selected_key_items_by_layer[layer_index], item.item_index);
      const bool has_online_nog = query.cache_items[item.item_index].is_nog;
      const int expected_qk_tokens =
          query.model.memory_tokens +
          (has_selected_key || has_online_nog
               ? query.cache_items[item.item_index].valid_text_tokens
               : 0);
      Require(item.qk_shape.Dim(3) == expected_qk_tokens, source,
              "runtime QK token dimension does not match selected/online K");
      Require(item.pv_shape.dims ==
                  std::vector<int64_t>({1, query.model.num_attention_heads,
                                        query.model.memory_tokens,
                                        query.model.head_dim}),
              source, "runtime PV shape mismatch");
      layer.items.push_back(std::move(item));
    }
    const YAML::Node gnn = RequireMap(layer_node, "gnn", source);
    layer.gnn_node_input_shape = Shape(gnn["node_input_shape"], source);
    layer.gnn_node_output_shape = Shape(gnn["node_output_shape"], source);
    layer.gnn_edge_input_shape = Shape(gnn["edge_input_shape"], source);
    Require(
        layer.gnn_node_input_shape.dims ==
                std::vector<int64_t>({query.num_graph_nodes,
                                      query.model.memory_tokens,
                                      query.model.hidden_size}) &&
            SameShape(layer.gnn_node_input_shape, layer.gnn_node_output_shape),
        source, "runtime GNN node shape mismatch");
    Require(layer.gnn_edge_input_shape.dims ==
                std::vector<int64_t>({query.num_structural_edges,
                                      query.model.memory_tokens,
                                      query.model.hidden_size}),
            source, "runtime GNN edge shape mismatch");
    Require(!gnn["edge_output_shape"] || gnn["edge_output_shape"].IsNull(),
            source, "runtime GNN edge output must be null");
    Require(Required<std::string>(gnn, "edge_output_status", source) ==
                "not_produced_by_gofa_gnn_layer",
            source, "runtime GNN edge output status mismatch");
    query.runtime_layers.push_back(std::move(layer));
  }

  int64_t memory_bytes = 0;
  int64_t full_key_bytes = 0;
  int64_t full_value_bytes = 0;
  int64_t edge_cache_bytes = 0;
  for (const auto &item : query.cache_items) {
    if (!item.cache_eligible) {
      continue;
    }
    memory_bytes += item.memory_logical_bytes;
    full_key_bytes += item.full_key_logical_bytes;
    full_value_bytes += item.full_value_logical_bytes;
    if (item.item_type == "edge") {
      edge_cache_bytes += item.memory_logical_bytes +
                          item.full_key_logical_bytes +
                          item.full_value_logical_bytes;
    }
  }
  int64_t selected_key_bytes = 0;
  int64_t selected_value_bytes = 0;
  for (size_t layer_index = 0;
       layer_index < query.model.suffix_layer_ids.size(); ++layer_index) {
    const int layer_id = query.model.suffix_layer_ids[layer_index];
    for (int item_index : query.selected_key_items_by_layer[layer_index]) {
      const auto &item = query.cache_items[item_index];
      selected_key_bytes += QuantizedBytes(
          FindKVLayer(item, layer_id, source).key_shape, item.key_bits);
    }
    for (int item_index : query.selected_value_items_by_layer[layer_index]) {
      const auto &item = query.cache_items[item_index];
      selected_value_bytes += QuantizedBytes(
          FindKVLayer(item, layer_id, source).value_shape, item.value_bits);
    }
  }
  const YAML::Node traffic = RequireMap(trace, "traffic_metadata", source);
  Require(Required<std::string>(traffic, "byte_accounting", source) ==
              "quantized_data_only_excluding_scale_and_container_metadata",
          source, "traffic byte accounting mode mismatch");
  query.traffic.memory_cache_bytes =
      Required<int64_t>(traffic, "memory_cache_bytes", source);
  query.traffic.selected_key_bytes =
      Required<int64_t>(traffic, "selected_key_bytes", source);
  query.traffic.selected_value_bytes =
      Required<int64_t>(traffic, "selected_value_bytes", source);
  query.traffic.full_key_bytes =
      Required<int64_t>(traffic, "full_key_bytes", source);
  query.traffic.full_value_bytes =
      Required<int64_t>(traffic, "full_value_bytes", source);
  query.traffic.edge_cache_bytes =
      Required<int64_t>(traffic, "edge_cache_bytes", source);
  query.traffic.nog_online_item_count =
      Required<int>(traffic, "nog_online_item_count", source);
  query.traffic.persistent_cache_bytes =
      Required<int64_t>(traffic, "persistent_cache_bytes", source);
  query.traffic.runtime_loaded_cache_bytes =
      Required<int64_t>(traffic, "runtime_loaded_cache_bytes", source);
  Require(query.traffic.memory_cache_bytes == memory_bytes &&
              query.traffic.selected_key_bytes == selected_key_bytes &&
              query.traffic.selected_value_bytes == selected_value_bytes &&
              query.traffic.full_key_bytes == full_key_bytes &&
              query.traffic.full_value_bytes == full_value_bytes &&
              query.traffic.edge_cache_bytes == edge_cache_bytes &&
              query.traffic.nog_online_item_count == nog_count &&
              query.traffic.persistent_cache_bytes ==
                  memory_bytes + full_key_bytes + full_value_bytes &&
              query.traffic.runtime_loaded_cache_bytes ==
                  memory_bytes + selected_key_bytes + selected_value_bytes,
          source, "trace traffic metadata mismatch");

  const YAML::Node summary = RequireMap(trace, "summary", source);
  query.total_item_count = Required<int>(summary, "total_item_count", source);
  query.cacheable_item_count =
      Required<int>(summary, "cacheable_item_count", source);
  query.nog_count = Required<int>(summary, "NOG_count", source);
  Require(Required<std::string>(summary, "selection_ratio_basis", source) ==
              "cacheable_item_layer_accesses",
          source, "selection ratio basis mismatch");
  query.selected_key_ratio =
      Required<double>(summary, "selected_K_ratio", source);
  query.selected_value_ratio =
      Required<double>(summary, "selected_V_ratio", source);
  query.selected_kv_ratio =
      Required<double>(summary, "selected_KV_ratio", source);
  Require(
      query.total_item_count == total_items &&
          query.cacheable_item_count ==
              static_cast<int>(query.eligible_item_indices.size()) &&
          query.nog_count == nog_count &&
          Required<int64_t>(summary, "persistent_cache_bytes", source) ==
              query.traffic.persistent_cache_bytes &&
          Required<int64_t>(summary, "runtime_loaded_cache_bytes", source) ==
              query.traffic.runtime_loaded_cache_bytes,
      source, "query summary count/byte mismatch");
  const double denominator = 1.0 * query.eligible_item_indices.size() *
                             query.model.suffix_layer_ids.size();
  double key_accesses = 0.0;
  double value_accesses = 0.0;
  double complete_accesses = 0.0;
  for (size_t layer_index = 0;
       layer_index < query.model.suffix_layer_ids.size(); ++layer_index) {
    const auto &keys = query.selected_key_items_by_layer[layer_index];
    const auto &values = query.selected_value_items_by_layer[layer_index];
    key_accesses += keys.size();
    value_accesses += values.size();
    std::vector<int> intersection;
    std::set_intersection(keys.begin(), keys.end(), values.begin(),
                          values.end(), std::back_inserter(intersection));
    complete_accesses += intersection.size();
  }
  RequireNear(query.selected_key_ratio,
              denominator == 0.0 ? 0.0 : key_accesses / denominator, source,
              "selected_K_ratio");
  RequireNear(query.selected_value_ratio,
              denominator == 0.0 ? 0.0 : value_accesses / denominator, source,
              "selected_V_ratio");
  RequireNear(query.selected_kv_ratio,
              denominator == 0.0 ? 0.0 : complete_accesses / denominator,
              source, "selected_KV_ratio");
  return query;
}

void ValidateIndexEntry(const YAML::Node &entry, const GOFATraceQuery &query,
                        const std::string &source) {
  Require(Required<std::string>(entry, "query_id", source) == query.query_id,
          source, "query_id mismatch");
  Require(Required<std::string>(entry, "task", source) == query.task_name,
          source, "task mismatch");
  Require(NormalizeSplit(Required<std::string>(entry, "split", source),
                         source) == query.split,
          source, "split mismatch");
  Require(Required<int>(entry, "num_nodes", source) == query.num_graph_nodes &&
              Required<int>(entry, "num_edges", source) ==
                  query.num_structural_edges &&
              Required<int>(entry, "cacheable_items", source) ==
                  query.cacheable_item_count &&
              Required<int>(entry, "selected_K_items", source) ==
                  static_cast<int>(query.selected_key_item_indices.size()) &&
              Required<int>(entry, "selected_V_items", source) ==
                  static_cast<int>(query.selected_value_item_indices.size()) &&
              Required<int64_t>(entry, "runtime_loaded_bytes", source) ==
                  query.traffic.runtime_loaded_cache_bytes,
          source, "index characterization mismatch");
}

std::vector<GOFATraceQuery> LoadTask(const GOFATraceLoadConfig &config,
                                     const GOFATraceTaskConfig &task,
                                     int *next_trace_order) {
  const fs::path directory = fs::path(config.root) / task.directory;
  Require(fs::is_directory(directory), directory.string(),
          "trace task directory is missing");
  const fs::path index_path = directory / "trace_index.jsonl";
  Require(fs::is_regular_file(index_path), index_path.string(),
          "trace index is missing");
  std::ifstream index_file(index_path);
  Require(index_file.is_open(), index_path.string(), "cannot open trace index");

  std::vector<GOFATraceQuery> queries;
  std::set<std::string> query_ids;
  std::set<fs::path> indexed_paths;
  std::string line;
  int line_number = 0;
  while (std::getline(index_file, line)) {
    ++line_number;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }
    YAML::Node entry;
    const std::string source =
        index_path.string() + ":" + std::to_string(line_number);
    try {
      entry = YAML::Load(line);
    } catch (const YAML::Exception &ex) {
      Fail(source, "invalid JSONL entry: " + std::string(ex.what()));
    }
    Require(entry.IsMap(), source, "index entry must be an object");
    const std::string trace_path_value =
        Required<std::string>(entry, "trace_path", source);
    fs::path trace_path(trace_path_value);
    if (trace_path.is_relative()) {
      trace_path = directory / trace_path;
    }
    trace_path = trace_path.lexically_normal();
    Require(fs::is_regular_file(trace_path), source,
            "indexed query trace is missing: " + trace_path.string());
    Require(indexed_paths.insert(trace_path).second, source,
            "duplicate trace_path in index");
    GOFATraceQuery query = LoadQuery(trace_path, (*next_trace_order)++, config);
    Require(trace_path.stem().string() == query.query_id, source,
            "trace filename does not match query_id");
    Require(query.task_name == task.name, source,
            "trace task_name does not match configured task");
    Require(query_ids.insert(query.query_id).second, source,
            "duplicate query_id in index");
    ValidateIndexEntry(entry, query, source);
    queries.push_back(std::move(query));
  }

  std::set<fs::path> trace_files;
  for (const auto &entry : fs::directory_iterator(directory)) {
    const std::string filename = entry.path().filename().string();
    if (entry.is_regular_file() && filename.rfind("query_", 0) == 0 &&
        entry.path().extension() == ".json") {
      trace_files.insert(entry.path().lexically_normal());
    }
  }
  Require(trace_files == indexed_paths, directory.string(),
          "trace_index.jsonl does not exactly enumerate query_*.json files");
  Require(static_cast<int>(queries.size()) ==
              config.expected_queries_per_split *
                  static_cast<int>(config.splits.size()),
          directory.string(), "unexpected total query count");
  for (size_t split_index = 0; split_index < config.splits.size();
       ++split_index) {
    const std::string split =
        NormalizeSplit(config.splits[split_index], directory.string());
    const int count = std::count_if(
        queries.begin(), queries.end(),
        [&split](const GOFATraceQuery &query) { return query.split == split; });
    Require(count == config.expected_queries_per_split, directory.string(),
            "unexpected " + split + " query count");
    const size_t begin = split_index * config.expected_queries_per_split;
    const size_t end = begin + config.expected_queries_per_split;
    for (size_t query_index = begin; query_index < end; ++query_index) {
      Require(queries[query_index].split == split, directory.string(),
              "trace index split order must follow configured split order");
    }
  }
  return queries;
}

void ValidateDistribution(const YAML::Node &node,
                          const std::vector<double> &values,
                          const std::string &source) {
  Require(node && node.IsMap(), source, "distribution must be an object");
  const double minimum =
      values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
  const double maximum =
      values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
  RequireNear(Required<double>(node, "min", source), minimum, source, "min");
  RequireNear(Required<double>(node, "mean", source), Mean(values), source,
              "mean");
  RequireNear(Required<double>(node, "p50", source), Percentile(values, 0.50),
              source, "p50");
  RequireNear(Required<double>(node, "p95", source), Percentile(values, 0.95),
              source, "p95");
  RequireNear(Required<double>(node, "max", source), maximum, source, "max");
}

void ValidateTotalMean(const YAML::Node &node,
                       const std::vector<double> &values,
                       const std::string &source) {
  Require(node && node.IsMap(), source, "total/mean field must be an object");
  const double total = std::accumulate(values.begin(), values.end(), 0.0);
  RequireNear(Required<double>(node, "total", source), total, source, "total");
  RequireNear(Required<double>(node, "mean", source), Mean(values), source,
              "mean");
}

void ValidateTaskSummary(const fs::path &task_directory,
                         const std::string &task_name,
                         const std::vector<GOFATraceQuery> &queries) {
  const fs::path summary_path = task_directory / "summary.json";
  Require(fs::is_regular_file(summary_path), summary_path.string(),
          "task summary is missing");
  YAML::Node root;
  try {
    root = YAML::LoadFile(summary_path.string());
  } catch (const YAML::Exception &ex) {
    Fail(summary_path.string(),
         "invalid summary JSON: " + std::string(ex.what()));
  }
  const std::string source = summary_path.string();
  Require(Required<std::string>(root, "trace_format", source) ==
              "gofa_query_trace_summary_v1",
          source, "summary trace_format mismatch");
  const YAML::Node tasks = RequireMap(root, "tasks", source);
  Require(tasks.size() == 1, source,
          "task summary must contain exactly one task");
  const YAML::Node task = tasks[task_name];
  Require(task && task.IsMap(), source,
          "summary does not contain task " + task_name);
  Require(Required<int>(task, "query_count", source) ==
              static_cast<int>(queries.size()),
          source, "summary query_count mismatch");

  std::vector<double> nodes;
  std::vector<double> edges;
  std::vector<double> node_items;
  std::vector<double> edge_items;
  std::vector<double> selected_k;
  std::vector<double> selected_v;
  std::vector<double> selected_kv;
  std::vector<double> persistent;
  std::vector<double> loaded;
  std::vector<double> nog;
  for (const auto &query : queries) {
    nodes.push_back(query.num_graph_nodes);
    edges.push_back(query.num_structural_edges);
    node_items.push_back(query.num_node_text_items);
    edge_items.push_back(query.num_edge_text_items);
    selected_k.push_back(query.selected_key_ratio);
    selected_v.push_back(query.selected_value_ratio);
    selected_kv.push_back(query.selected_kv_ratio);
    persistent.push_back(query.traffic.persistent_cache_bytes);
    loaded.push_back(query.traffic.runtime_loaded_cache_bytes);
    nog.push_back(query.traffic.nog_online_item_count);
  }
  ValidateDistribution(task["node_count"], nodes, source + ".node_count");
  ValidateDistribution(task["edge_count"], edges, source + ".edge_count");
  ValidateTotalMean(task["node_text_item_count"], node_items,
                    source + ".node_text_item_count");
  ValidateTotalMean(task["edge_text_item_count"], edge_items,
                    source + ".edge_text_item_count");
  RequireNear(Required<double>(task, "selected_K_ratio", source),
              Mean(selected_k), source, "selected_K_ratio");
  RequireNear(Required<double>(task, "selected_V_ratio", source),
              Mean(selected_v), source, "selected_V_ratio");
  RequireNear(Required<double>(task, "selected_KV_ratio", source),
              Mean(selected_kv), source, "selected_KV_ratio");
  ValidateTotalMean(task["persistent_cache_bytes"], persistent,
                    source + ".persistent_cache_bytes");
  ValidateTotalMean(task["loaded_cache_bytes"], loaded,
                    source + ".loaded_cache_bytes");
  ValidateTotalMean(task["NOG_online_count"], nog,
                    source + ".NOG_online_count");
}

}  // namespace

GOFATraceLoadResult LoadAndValidateGOFATraces(
    const GOFATraceLoadConfig &config) {
  Require(!config.root.empty(), "trace_workload.root",
          "trace root must not be empty");
  Require(!config.tasks.empty(), config.root,
          "at least one trace task is required");
  Require(config.expected_queries_per_split > 0, config.root,
          "expected_queries_per_split must be positive");
  GOFATraceLoadResult result;
  int next_trace_order = 0;
  for (const auto &task : config.tasks) {
    std::vector<GOFATraceQuery> task_queries =
        LoadTask(config, task, &next_trace_order);
    if (config.validate_summary) {
      ValidateTaskSummary(fs::path(config.root) / task.directory, task.name,
                          task_queries);
    }
    for (auto &query : task_queries) {
      if (query.split == "validation") {
        ++result.validation_query_count;
      } else if (query.split == "test") {
        ++result.test_query_count;
      }
      result.queries.push_back(std::move(query));
    }
  }
  return result;
}

}  // namespace analytical
}  // namespace llm_system
