#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
class DRAMRequest;
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

ExecStatus GNNExecutionGPU(Device_Ptr device,
                           std::vector<Tensor_Ptr> tensor,
                           BatchedSequence::Ptr sequences_metadata,
                           LayerInfo layer_info,
                           bool use_ramulator) {
  Tensor_Ptr encoded_node_features = tensor.at(0);
  Tensor_Ptr intermediate_cache = tensor.at(1);

  auto config = device->config;
  // GPU 直接读取峰值算力和带宽，通常不根据数据精度翻倍算力（除非特定 Tensor Core 逻辑，这里保持与 Attention 基础模拟一致）
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int hidden_dim = layer_info.head_dim * layer_info.num_heads;
  
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  ExecStatus exec_status;

  Graph* graph = &(sequences_metadata->graph);
  if (!graph || graph->num_nodes == 0) {
    return exec_status;
  }

  // ==========================================
  // Phase 1: 消息聚合 (Graph Aggregation)
  // ==========================================
  int accumul_len_agg = 0;
  time_ns accumul_compute_duration_agg = 0;
  time_ns accumul_memory_duration_agg = 0;

  for (int i = 0; i < graph->num_nodes; i++) {
    for (int neighbor : graph->adj[i]) {
      // 聚合操作为加法: 1 FLOP per element
      flops = hidden_dim * 1.0; 
      total_flops += flops;

      // GPU: 随机读取邻居特征
      memory_size = hidden_dim * encoded_node_features->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      accumul_compute_duration_agg += compute_duration;
      exec_status.compute_duration += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration_agg += memory_duration;

      // GPU 无需严格的 PIM 4-byte 对齐
      accumul_len_agg += hidden_dim;
    }
  }

  if (use_ramulator) {
    intermediate_cache->setShape({accumul_len_agg, 1});
    // GPU 发送普通读指令
    ExecStatus temp = issueRamulator(device, LayerType::GRAPH_AGGREGATION, ProcessorType::GPU,
                                     DRAMRequestType::kRead, PIMOperandType::kDRAM, intermediate_cache);
    exec_status += temp;
    accumul_memory_duration_agg = temp.memory_duration;
  } else {
    intermediate_cache->setShape({accumul_len_agg, 1});
    ExecStatus temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, intermediate_cache);
    exec_status += temp;
  }

  // GPU 遵循 Roofline 模型：取计算与访存开销的最大值
  exec_status.total_duration += std::max(accumul_compute_duration_agg, accumul_memory_duration_agg);

  // ==========================================
  // Phase 2: 特征变换 (Node-wise Update / FFN)
  // ==========================================
  time_ns accumul_compute_duration_update = 0;
  time_ns accumul_memory_duration_update = 0;

  // GPU 必须从显存加载全连接层权重 W (假设 Batch 内复用一次)
  double weight_memory_size = hidden_dim * hidden_dim * encoded_node_features->precision_byte;
  total_memory_size += weight_memory_size;
  accumul_memory_duration_update += (weight_memory_size / memory_bandwidth * 1000 * 1000 * 1000);

  for (int i = 0; i < graph->num_nodes; i++) {
    // 矩阵向量乘法 MACs: 2 FLOPs per element
    flops = hidden_dim * hidden_dim * 2.0; 
    total_flops += flops;

    // GPU: 需要读取输入节点特征 x，并写入输出特征 x'
    memory_size = (hidden_dim + hidden_dim) * encoded_node_features->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    accumul_compute_duration_update += compute_duration;
    exec_status.compute_duration += compute_duration;

    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
    accumul_memory_duration_update += memory_duration;
  }

  std::vector<int> org_shape = encoded_node_features->getShape();
  encoded_node_features->setShape({graph->num_nodes, hidden_dim});

  if (use_ramulator) {
    // GPU 在 FFN 阶段依然发送普通 kRead (或相应的普通读写指令)
    ExecStatus temp = issueRamulator(device, LayerType::GRAPH_FFN, ProcessorType::GPU,
                                     DRAMRequestType::kRead, PIMOperandType::kDRAM, encoded_node_features);
    exec_status += temp;
    accumul_memory_duration_update = temp.memory_duration;
  } else {
    ExecStatus temp = getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, encoded_node_features);
    exec_status += temp;
  }

  encoded_node_features->setShape(org_shape);

  // GPU FFN 阶段时间重叠
  exec_status.total_duration += std::max(accumul_compute_duration_update, accumul_memory_duration_update);

  exec_status.compute_util = 1e9 * total_flops / compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1e9 * total_memory_size / memory_bandwidth / exec_status.total_duration;
  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
}

ExecStatus GNNExecutionLogic(Device_Ptr device,
                             std::vector<Tensor_Ptr> tensor,
                             BatchedSequence::Ptr sequences_metadata,
                             LayerInfo layer_info,
                             bool use_ramulator) {
  Tensor_Ptr encoded_node_features = tensor.at(0);
  Tensor_Ptr intermediate_cache = tensor.at(1);

  auto config = device->config;
  hw_metric compute_peak_flops = config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  
  if (encoded_node_features->precision_byte == 1) {
    compute_peak_flops *= 2;
  }

  int hidden_dim = layer_info.head_dim * layer_info.num_heads;
  
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  ExecStatus exec_status;

  Graph* graph = &(sequences_metadata->graph);
  if (!graph || graph->num_nodes == 0) {
    return exec_status;
  }

  // ==========================================
  // Phase 1: 消息聚合 (Graph Aggregation)
  // ==========================================
  int accumul_len_agg = 0;
  time_ns accumul_compute_duration_agg = 0;
  time_ns accumul_memory_duration_agg = 0;

  for (int i = 0; i < graph->num_nodes; i++) {
    for (int neighbor : graph->adj[i]) {
      // [修复 1] Flops: 聚合通常是 Sum 或 Mean，只有加法，1 FLOP per element
      flops = hidden_dim * 1.0; 
      total_flops += flops;

      // Memory: 从内存拉取该邻居的特征向量
      memory_size = hidden_dim * encoded_node_features->precision_byte;
      total_memory_size += memory_size;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;
      accumul_compute_duration_agg += compute_duration;

      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration_agg += memory_duration;

      // 4-byte 硬件内存对齐
      int _n = (hidden_dim + 3) / 4 * 4;
      accumul_len_agg += _n;
    }
  }

  if (use_ramulator) {
    intermediate_cache->setShape({accumul_len_agg, 1});
    ExecStatus temp = issueRamulator(device, LayerType::GRAPH_AGGREGATION, ProcessorType::LOGIC,
                                     DRAMRequestType::kRead, PIMOperandType::kDRAM, intermediate_cache);
    exec_status += temp;
    accumul_memory_duration_agg = temp.memory_duration;
  } else {
    intermediate_cache->setShape({accumul_len_agg, 1});
    ExecStatus temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, intermediate_cache);
    exec_status += temp;
  }

  exec_status.total_duration += std::max(accumul_compute_duration_agg, accumul_memory_duration_agg);


  // ==========================================
  // Phase 2: 特征变换 (Node-wise Update / FFN)
  // ==========================================
  time_ns accumul_compute_duration_update = 0;
  time_ns accumul_memory_duration_update = 0;

  // [修复 2] Logic 层架构：需要从主存拉取全连接层的权重矩阵 W (尺寸为 hidden_dim * hidden_dim)
  // 假设权重矩阵在一个 batch 中被拉取一次并缓存在 Logic 层的 SRAM 中复用
  double weight_memory_size = hidden_dim * hidden_dim * encoded_node_features->precision_byte;
  total_memory_size += weight_memory_size;
  accumul_memory_duration_update += (weight_memory_size / memory_bandwidth * 1000 * 1000 * 1000);

  for (int i = 0; i < graph->num_nodes; i++) {
    // Flops: 节点特征的矩阵-向量乘法 (W * x)，乘加操作，2 FLOPs per element
    flops = hidden_dim * hidden_dim * 2.0; 
    total_flops += flops;

    // Memory: 读取当前节点的输入特征向量 x
    memory_size = hidden_dim * encoded_node_features->precision_byte;
    total_memory_size += memory_size;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    exec_status.compute_duration += compute_duration;
    accumul_compute_duration_update += compute_duration;

    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
    accumul_memory_duration_update += memory_duration;
  }

  std::vector<int> org_shape = encoded_node_features->getShape();

  // [修复 3] 保持 2D 结构：通知 Ramulator 这是一个 Batch Size = num_nodes 的 GEMV 操作
  // 加入硬件对齐的维度特征，防止将其降维成 1D 数组导致缓存时序模拟失效
  int aligned_hidden_dim = (hidden_dim + 3) / 4 * 4;
  encoded_node_features->setShape({graph->num_nodes, aligned_hidden_dim});

  if (use_ramulator) {
    ExecStatus temp = issueRamulator(device, LayerType::GRAPH_FFN, ProcessorType::LOGIC,
                                     DRAMRequestType::kGEMV, PIMOperandType::kSrc, encoded_node_features);
    exec_status += temp;
    
    // 如果 Ramulator 返回了更精确的访存时间，则覆盖计算模型的时间
    accumul_memory_duration_update = temp.memory_duration;
  } else {
    ExecStatus temp = getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, encoded_node_features);
    exec_status += temp;
  }

  // 恢复原始形状
  encoded_node_features->setShape(org_shape);

  // Roofline 模型：计算与访存重叠
  exec_status.total_duration += std::max(accumul_compute_duration_update, accumul_memory_duration_update);

  // ==========================================
  // 统计：计算整体利用率
  // ==========================================
  exec_status.compute_util = 1e9 * total_flops / compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1e9 * total_memory_size / memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
}

ExecStatus GNNExecutionPIM(Device_Ptr device,
                           std::vector<Tensor_Ptr> tensor,
                           BatchedSequence::Ptr sequences_metadata,
                           LayerInfo layer_info,
                           bool use_ramulator) {
  Tensor_Ptr encoded_node_features = tensor.at(0);
  Tensor_Ptr intermediate_cache = tensor.at(1);

  auto config = device->config;
  // PIM 算力由带宽和 OPB 推导
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;
  
  // PIM 中低精度计算吞吐量翻倍
  if (encoded_node_features->precision_byte == 1) {
    compute_peak_flops *= 2;
  }

  int hidden_dim = layer_info.head_dim * layer_info.num_heads;
  
  double flops = 0;
  double memory_size = 0;
  double total_flops = 0;
  double total_memory_size = 0;

  time_ns compute_duration;
  time_ns memory_duration;
  ExecStatus exec_status;

  Graph* graph = &(sequences_metadata->graph);
  if (!graph || graph->num_nodes == 0) {
    return exec_status;
  }

  // ==========================================
  // Phase 1: 消息聚合 (Graph Aggregation)
  // ==========================================
  int accumul_len_agg = 0;
  time_ns accumul_memory_duration_agg = 0;
  double phase1_flops = 0;
  double phase1_memory_size = 0;

  for (int i = 0; i < graph->num_nodes; i++) {
    for (int neighbor : graph->adj[i]) {
      flops = hidden_dim * 1.0; 
      phase1_flops += flops;
      total_flops += flops;

      memory_size = hidden_dim * encoded_node_features->precision_byte;
      phase1_memory_size += memory_size;
      total_memory_size += memory_size;

      // 【修复】在循环内部计算理论分析时延 (Analytical Memory Duration)
      memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
      accumul_memory_duration_agg += memory_duration;

      compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
      exec_status.compute_duration += compute_duration;

      int _n = (hidden_dim + 3) / 4 * 4;
      accumul_len_agg += _n;
    }
  }

  if (use_ramulator) {
    intermediate_cache->setShape({accumul_len_agg, 1});
    ExecStatus temp = issueRamulator(device, LayerType::GRAPH_AGGREGATION, ProcessorType::PIM,
                                     DRAMRequestType::kRead, PIMOperandType::kDRAM, intermediate_cache);
    exec_status += temp;
    // 【修复】安全检查：只有当模拟器返回非零时间才覆盖
    if (temp.memory_duration > 0) {
      accumul_memory_duration_agg = temp.memory_duration;
    }
  } else {
    intermediate_cache->setShape({accumul_len_agg, 1});
    ExecStatus temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, intermediate_cache);
    exec_status += temp;
    if (temp.memory_duration > 0) {
      accumul_memory_duration_agg = temp.memory_duration;
    }
    // 如果返回 0，则自动使用 for 循环中累加的 analytical duration
  }

  double opb_agg = (phase1_memory_size > 0) ? (phase1_flops / phase1_memory_size) : 0;
  exec_status.total_duration += accumul_memory_duration_agg * opb_agg;


  // ==========================================
  // Phase 2: 特征变换 (Node-wise Update / FFN)
  // ==========================================
  time_ns accumul_memory_duration_update = 0;
  double phase2_flops = 0;
  double phase2_memory_size = 0;

  for (int i = 0; i < graph->num_nodes; i++) {
    flops = hidden_dim * hidden_dim * 2.0; 
    phase2_flops += flops;
    total_flops += flops;

    memory_size = hidden_dim * encoded_node_features->precision_byte;
    phase2_memory_size += memory_size;
    total_memory_size += memory_size;

    // 【修复】在循环内部计算理论分析时延
    memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;
    accumul_memory_duration_update += memory_duration;

    compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
    exec_status.compute_duration += compute_duration;
  }

  std::vector<int> org_shape = encoded_node_features->getShape();
  int aligned_hidden_dim = (hidden_dim + 3) / 4 * 4;
  encoded_node_features->setShape({graph->num_nodes, aligned_hidden_dim});

  if (use_ramulator) {
    ExecStatus temp = issueRamulator(device, LayerType::GRAPH_FFN, ProcessorType::PIM,
                                     DRAMRequestType::kGEMV, PIMOperandType::kSrc, encoded_node_features);
    exec_status += temp;
    // 【修复】安全检查
    if (temp.memory_duration > 0) {
      accumul_memory_duration_update = temp.memory_duration;
    }
  } else {
    ExecStatus temp = getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, encoded_node_features);
    exec_status += temp;
    if (temp.memory_duration > 0) {
      accumul_memory_duration_update = temp.memory_duration;
    }
  }

  encoded_node_features->setShape(org_shape);

  double opb_update = (phase2_memory_size > 0) ? (phase2_flops / phase2_memory_size) : 0;
  exec_status.total_duration += accumul_memory_duration_update * opb_update;

  exec_status.compute_util = 1e9 * total_flops / compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1e9 * total_memory_size / memory_bandwidth / exec_status.total_duration;
  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
}

}