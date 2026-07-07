# Analytical PIM Simulator

This is the first architecture-level analytical simulator for the HBM3
near-bank PIM design. It is independent from Ramulator and does not model
cycle-accurate DRAM timing.

## Patch 1 Scope

Patch 1 calibrates the skeleton model into a query-level stage model:

- Fixed total memory tokens `M = 128`.
- `memory_token_tile` is a tile size, not the total token count.
- `num_token_tiles = ceil(M / memory_token_tile)`.
- GNN score/value compute is independent of tile size.
- Tile size affects scheduling overhead, q broadcast amortization, buffer
  occupancy, and reducer active groups.
- Query-level sampled subgraphs are used instead of full-graph execution.
- Selective KV policy:
  target node union target 1-hop neighbors union high-degree sampled nodes.
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

## Baselines

Patch 1 implements exactly three baselines:

- `h100_selective_kv`
- `pim_selective_kv_no_local_combine`
- `pim_selective_kv_local_combine`

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

## Query Graph Input

The default config uses a synthetic graph plus query sampled subgraphs. A
schema graph can be supplied with:

```yaml
query_graph:
  source: schema
  num_nodes: 4
  edge_list:
    - [0, 1, 0]
    - [2, 1, 1]
  node_degree: [1, 2, 1, 0]
  high_degree_nodes: [1]
  node_to_bank: [0, 1, 2, 3]
  edge_to_bank: [0, 2]
```

If `edge_to_bank` is omitted, the source node bank is used. If only
`edge_to_pseudo_channel` is provided, the first bank in that pseudo-channel is
used.

## Workload Config

```yaml
workload:
  mode: query_sampled_subgraph
  num_queries: 16
  hop: 2
  fanout: 16
  high_degree_top_percent: 0.05
  seed: 777
```

## Aggregate CSV

The aggregate CSV reports one row per baseline and tile size:

- `mean_latency_ns`
- `p50_latency_ns`
- `p95_latency_ns`
- `selected_kv_count`
- `q_broadcast_bytes`
- `score_traffic_bytes`
- `p_return_traffic_bytes`
- `message_reduce_traffic_bytes`
- `local_combine_buffer_max_bytes`
- `pc_reducer_buffer_max_bytes`
- `active_banks`
- `active_pseudo_channels`
- `near_bank_pe_utilization`
- `reducer_utilization`
