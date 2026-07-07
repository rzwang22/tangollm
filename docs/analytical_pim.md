# Analytical PIM Simulator

This target is an architecture-level analytical simulator for the HBM3
near-bank PIM design. It is independent from Ramulator and does not model
cycle-accurate DRAM timing or real tensor values.

## Patch 2 Scope

Patch 2 adds a workload suite and local-combine diagnosis on top of the Patch 1
calibrated stage model:

- Fixed total memory tokens `M = 128`.
- `memory_token_tile` is a tile size, not the total token count.
- `num_token_tiles = ceil(M / memory_token_tile)`.
- GNN score/value compute is independent of tile size.
- Tile size affects scheduling overhead, q broadcast, buffer occupancy, and
  reducer active groups.
- Query-level synthetic sampled subgraphs are used instead of full-graph
  execution.
- Placement policies are swept independently from workload size.
- No Ramulator, no cycle-accurate DRAM, no real tensor values.

## Sanity Check

The simulator prints model dimensions and text-side KV bytes per item at
startup. With the default config:

```text
memory_tokens=128
hidden_dim=4096
gnn_heads=32
head_dim=128
channel_group=16
groups_per_head=8
suffix_layers=6
text_len=256
kv_heads=8
kv_bits=2
```

The text-side K/V payload per selected item is:

```text
6 * 2 * 256 * 8 * 128 * 2 / 8 = 786432 bytes
```

Scale metadata is controlled by:

```yaml
model:
  include_scale_metadata: false
```

## Workload Suite

The default synthetic query-level suite is:

- `smoke`: approximately 5 sampled nodes and 15 sampled edges.
- `small`: approximately 16 sampled nodes and 64 sampled edges.
- `medium`: approximately 64 sampled nodes and 512 sampled edges.
- `large`: approximately 128 sampled nodes and 2048 sampled edges.
- `skewed_high_degree`: 128 sampled nodes and 2048 dst-skewed edges.

Each workload emits `num_queries_per_workload` independent query samples. The
selective KV policy is:

```text
target node union target 1-hop neighbors union high-degree sampled nodes
```

The CSV reports both `selected_kv_ratio_vs_sampled_nodes` and
`selected_kv_ratio_vs_full_graph`.

## Placement Sweep

Patch 2 supports three placement policies:

- `hash`: source-node hash placement.
- `degree_balanced`: sampled nodes are assigned to the least-loaded bank by
  descending degree.
- `source_dst_locality`: edge execution is placed by destination-node hash to
  improve local combine opportunities.

The output includes bank and pseudo-channel imbalance, local-combine reduction,
latency, and traffic for each placement.

## Baselines

Patch 2 implements four baselines:

- `h100_ideal_compute_only`
- `h100_realistic_cache_path`
- `pim_selective_kv_no_local_combine`
- `pim_selective_kv_local_combine`

The realistic H100 path includes configurable INT2 K/V read, unpack,
scale/dequant, layout conversion, irregular gather penalty, and small-batch
efficiency penalty.

## Local-Combine Diagnosis

The per-query and aggregate CSV files include:

- `edge_message_count_before_local_combine`
- `bank_local_group_count_after_combine`
- `pc_group_count_after_pc_reduce`
- `local_combine_reduction_ratio`
- `pc_reduction_ratio`
- `avg_edges_per_bank_dst`
- `p95_edges_per_bank_dst`
- `max_edges_per_bank_dst`
- `message_traffic_before_local_combine`
- `message_traffic_after_local_combine`
- `bank_imbalance`
- `pseudo_channel_imbalance`

These fields are intended to diagnose whether a placement creates enough
same-bank and same-pseudo-channel destination reuse for local combine to help.

## Build

```bash
cd build
cmake ..
make -j analytical_pim
```

## Run

```bash
cd build
./analytical_pim analytical_pim_config.yaml
```

Outputs:

```text
../data/analytical_pim_per_query.csv
../data/analytical_pim_aggregate.csv
```

With the default config, the aggregate CSV has:

```text
5 workloads * 3 placements * 4 baselines * 3 tile sizes = 180 data rows
```

The per-query CSV has:

```text
5 workloads * 16 queries * 3 placements * 4 baselines * 3 tile sizes = 2880 data rows
```
