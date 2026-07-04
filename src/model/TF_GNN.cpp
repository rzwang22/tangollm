#include "model/tf_gnn.h"

#include "module/decoder.h"
#include "module/embedding.h"
#include "module/layer.h"
#include "module/layernorm.h"
#include "module/communication.h"

namespace llm_system {

//////////////////////////////////////////////////////////////
// TF-GNN Main Model
//////////////////////////////////////////////////////////////

TFGNN::TFGNN(const ModelConfig& model_config,
             Cluster::Ptr cluster,
             Scheduler::Ptr scheduler,
             Device::Ptr device)
    : Module("", "TFGNN", device),
      model_config(model_config) {

  std::vector<int> device_list;

  set_device_list(device_list,
                  0,
                  cluster->num_total_device);

  //////////////////////////////////////////////////////
  // Embedding
  //////////////////////////////////////////////////////

  auto embedding =
      Embedding::Create(
          module_map_name,
          "embedding",
          model_config,
          device_list,
          device);

  add_module(embedding);

  //////////////////////////////////////////////////////
  // Transformer Encoder
  //////////////////////////////////////////////////////

  for (int layer = 0;
       layer < model_config.num_layers;
       layer++) {

    auto decoder =
        Decoder::Create(
            module_map_name,
            "transformer_" + std::to_string(layer),
            model_config,
            scheduler,
            device_list,
            device);

    add_module(decoder);
  }

  auto pooling_layer = Pooling::Create(module_map_name, "pooling_layer",
                                       model_config.hidden_dim,
                                       device_list, device);
  add_module(pooling_layer);

  //////////////////////////////////////////////////////
  // GNN Aggregation
  //////////////////////////////////////////////////////

  auto gnn_layer = GNN::Create(module_map_name, "gnn_layer",
                               model_config.hidden_dim,
                               128, 
                               device_list, device);
  add_module(gnn_layer);

  //////////////////////////////////////////////////////
  // Final LayerNorm
  //////////////////////////////////////////////////////

  auto output_norm =
      LayerNorm::Create(
          module_map_name,
          "output_norm",
          model_config.hidden_dim,
          device_list,
          device);

  add_module(output_norm);
}

//////////////////////////////////////////////////////////////
// Forward
//////////////////////////////////////////////////////////////

Tensor::Ptr TFGNN::forward(
    const Tensor::Ptr input,
    BatchedSequence::Ptr sequences_metadata) {

  //////////////////////////////////////////////////////
  // Embedding
  //////////////////////////////////////////////////////

  Module::Ptr embedding =
      get_module("embedding");

  Tensor::Ptr out =
      (*embedding)(input,
                   sequences_metadata);

  //////////////////////////////////////////////////////
  // Transformer Encoding
  //////////////////////////////////////////////////////

  for (int layer = 0;
       layer < model_config.num_layers;
       layer++) {

    Module::Ptr decoder =
        get_module("transformer_" +
                   std::to_string(layer));

    out =
        (*decoder)(out,
                   sequences_metadata);
  }

  Module::Ptr pooling_layer = get_module("pooling_layer");
  Tensor::Ptr pooled_out = (*pooling_layer)(out, sequences_metadata);

  //////////////////////////////////////////////////////
  // Graph Aggregation
  //////////////////////////////////////////////////////

  Module::Ptr gnn_layer = get_module("gnn_layer");
  Tensor::Ptr gnn_out = (*gnn_layer)(pooled_out, sequences_metadata);

  //////////////////////////////////////////////////////
  // Output LayerNorm
  //////////////////////////////////////////////////////

  Module::Ptr output_norm =
      get_module("output_norm");

  out =
      (*output_norm)(gnn_out,
                     sequences_metadata);

  return out;
}

GNN::GNN(std::string& prefix, std::string& name,
         int hidden_dim, int max_nodes, 
         std::vector<int> device_list,
         Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      hidden_dim(hidden_dim),
      max_nodes(max_nodes) {
  
  // 定义基础形状 [max_nodes, hidden_dim]
  std::vector<int> shape = {max_nodes, hidden_dim};

  // 1. 注册中间缓存 (Intermediate Cache)
  // 用于 GNN Phase 1: 消息聚合时临时存放拉取的邻居特征
  Tensor::Ptr intermediate_cache = Tensor::Create(
      "gnn_cache", shape, "cache", device, device->model_config.precision_byte);
  add_tensor(intermediate_cache);

  // 2. 注册输出 Tensor
  // 用于存放 GNN Phase 2: 特征变换后的最终节点特征
  Tensor::Ptr output = Tensor::Create(
      "gnn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

// 前向传播：负责获取实际数据大小、构建硬件指令上下文并触发 execution
Tensor::Ptr GNN::forward(const Tensor::Ptr input,
                         BatchedSequence::Ptr sequences_metadata) {
  
  // 提取当前输入的实际节点数
  Graph& graph = sequences_metadata->graph;
  int num_nodes = graph.num_nodes;

  // 动态调整当前轮次的输出形状
  std::vector<int> current_shape = {num_nodes, hidden_dim};
  Tensor::Ptr output_tensor = get_activation("gnn_output", current_shape);

  Tensor::Ptr cache_tensor = get_cache("gnn_cache");
  cache_tensor->shape = current_shape;

  // 构建传递给底层的硬件/算子配置
  LayerInfo layer_info;

  layer_info.processor_type = {device->config.low_processor_type};
  layer_info.parallel_execution = false;
  output_tensor->setPerformLow();

  // 适配 GNNExecutionLogic 的维度计算:
  // 底层解析规则是 hidden_dim = head_dim * num_heads
  layer_info.head_dim = hidden_dim;
  layer_info.num_heads = 1; 

  // 组装张量列表，顺序必须与 GNNExecutionLogic 中 tensor.at(0) 和 at(1) 对应
  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.push_back(input);                               // at(0): 经过编码的输入节点特征
  tensor_list.push_back(get_cache("gnn_cache"));             // at(1): 聚合时的中间缓存

  // 触发硬件执行器
  // 注意：需要确保底层的 device->execution() 路由中心已经注册了 LayerType::GNN，
  // 并将其指向你新写的 GNNExecutionLogic 模拟函数。
  device->execution(LayerType::GRAPH_AGGREGATION, tensor_list, sequences_metadata, layer_info);

  return output_tensor;
}

Pooling::Pooling(std::string& prefix, std::string& name, int hidden_dim,
                 std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      hidden_dim(hidden_dim) {
  // 仿照 LayerNorm，在构造时先预分配一个基础形状的占位 Tensor
  std::vector<int> shape = {hidden_dim, 1};
  Tensor::Ptr pooling_out = Tensor::Create("pooling_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(pooling_out);
}

Tensor::Ptr Pooling::forward(const Tensor::Ptr input,
                             BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0]; // 输入的 Token 总数 (等于所有节点的 token_count 之和)
  int k = input->shape[1]; // hidden_dim

  Graph& graph = sequences_metadata->graph;
  int num_nodes = graph.num_nodes;

  // 动态获取并调整输出 Activation 的形状为 [num_nodes, hidden_dim]
  std::vector<int> output_shape = {num_nodes, k};
  Tensor::Ptr output = get_activation("pooling_output", output_shape);

  long size = input->getSize();
  if (size == 0 || num_nodes == 0) {
    return output;
  }

  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  // =========================================================
  // 开销建模计算 (内嵌)
  // =========================================================
  double flops, memory_size;

  // FLOPs: 把 m 个 Token 聚合为 num_nodes 个节点特征。
  // 每个元素经历一次 Reduction 操作（如 Sum/Mean/Max，均约 1 次算术操作）
  flops = 1.0 * m * k; 

  // Memory Size: 
  // 1. 读取所有 Token 的特征图: m * k
  // 2. 写入池化后的节点特征图: num_nodes * k
  memory_size = (1.0 * m * k + 1.0 * num_nodes * k) * input->precision_byte;

  // 计算延迟 (纳秒)
  time_ns compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  time_ns total_time = std::max(compute_duration, memory_duration);

  // =========================================================
  // 更新模拟器硬件状态时间 (完全对齐 LayerNorm)
  // =========================================================
  if (input->parallel_execution) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

}  // namespace llm_system