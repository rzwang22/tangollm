#pragma once

#include "hardware/cluster.h"
#include "model/model_config.h"
#include "model/util.h"
#include "scheduler/scheduler.h"

namespace llm_system {

class TFGNN : public Module {
 public:
  using Ptr = std::shared_ptr<TFGNN>;

  [[nodiscard]] static Ptr Create(
      const ModelConfig& model_config,
      Cluster::Ptr cluster,
      Scheduler::Ptr scheduler,
      Device::Ptr device) {

    Ptr ptr = Ptr(
        new TFGNN(
            model_config,
            cluster,
            scheduler,
            device));

    ptr->set_tensor_module();

    return ptr;
  };

 private:
  TFGNN(const ModelConfig& model_config,
        Cluster::Ptr cluster,
        Scheduler::Ptr scheduler,
        Device::Ptr device);

  Tensor::Ptr forward(
      const Tensor::Ptr input,
      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
};

class GNN : public Module {
 public:
  using Ptr = std::shared_ptr<GNN>;
  
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int hidden_dim, int max_nodes,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new GNN(prefix, name, hidden_dim, max_nodes,
                          device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  GNN(std::string& prefix, std::string& name, int hidden_dim, int max_nodes, 
      std::vector<int> device_list, Device::Ptr device);
  GNN() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int hidden_dim;
  int max_nodes;
};

class Pooling : public Module {
 public:
  using Ptr = std::shared_ptr<Pooling>;
  
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int hidden_dim, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Pooling(prefix, name, hidden_dim, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  Pooling(std::string& prefix, std::string& name, int hidden_dim,
          std::vector<int> device_list, Device::Ptr device);
  Pooling() = default;

  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int hidden_dim;
};

}  // namespace llm_system