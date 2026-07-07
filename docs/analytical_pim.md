# Analytical PIM Simulator

This is the first architecture-level analytical simulator for the HBM3
near-bank PIM design. It is intentionally independent from Ramulator and does
not model cycle-accurate DRAM timing.

## Scope

- Single-query stage-level timing and traffic model.
- Logical HBM3-like topology:
  stack -> channel -> pseudo-channel -> bank -> near-bank PE.
- Three-level reducer model:
  bank-local combine, pseudo-channel reducer, base-die global reducer.
- Tile sweep over `memory_token_tile`, with `channel_group = 16` and
  `head_tile = 1` by default.
- Selective KV loading:
  target nodes, 1-hop neighbors of targets, and high-degree nodes.

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

The default output is:

```text
../data/analytical_pim_tiles.csv
```

## Query Graph Input

The default config uses a synthetic graph. A schema graph can be supplied in
the config with:

```yaml
query_graph:
  source: schema
  num_nodes: 4
  edge_list:
    - [0, 1, 0]
    - [2, 1, 1]
  target_nodes: [1]
  node_degree: [1, 2, 1, 0]
  high_degree_nodes: [1]
  node_to_bank: [0, 1, 2, 3]
  edge_to_bank: [0, 2]
  selected_kv_mask: [true, true, true, false]
```

If `selected_kv_mask` is omitted, the simulator builds it from:

```text
target nodes union target 1-hop neighbors union high-degree nodes
```

If `edge_to_bank` is omitted, the source node bank is used. If only
`edge_to_pseudo_channel` is provided, the first bank in that pseudo-channel is
used.

## Key Outputs

- `q_broadcast_bytes`
- `score_traffic_bytes`
- `p_return_traffic_bytes`
- `message_reduce_traffic_bytes`
- `local_combine_buffer_*`
- `pc_reducer_buffer_*`
- `gnn_score_cycles`
- `gnn_message_cycles`
- `cached_kv_cycles`
- `reducer_cycles`
- `near_bank_pe_utilization`
- `reducer_utilization`
- `selected_kv_count`
- `selected_kv_ratio`
- `kv_reduction_vs_all_nodes`
- `kv_reduction_vs_all_edges`
