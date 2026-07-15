# Analytical PIM Simulator

This target is an architecture-level analytical simulator for the HBM3
near-bank PIM design. It is independent from Ramulator and does not model
cycle-accurate DRAM timing or real tensor values.

## Patch 4 Scope

Patch 4 adds workload correctness, fair reducer paths, and communication-aware
timing on top of the Patch 3 hybrid placement model:

- Fixed total memory tokens `M = 128`.
- `memory_token_tile` is a tile size, not the total token count.
- `num_token_tiles = ceil(M / memory_token_tile)`.
- GNN score/value compute is independent of tile size.
- Tile size affects scheduling overhead, q broadcast, buffer occupancy, and
  reducer active groups.
- Query-level synthetic sampled subgraphs are used instead of full-graph
  execution.
- Synthetic edges are sampled without replacement by default, so duplicate
  `(src,dst)` edges are prohibited.
- Uniform and power-law degree distributions and destination skew are
  configurable and reproducible with a fixed random seed.
- Placement policies are swept independently from workload size.
- Hybrid placement shards only hot destinations and balances both bank and
  pseudo-channel pressure.
- Primitive-level PIM and H100 cache-path cycles are reported separately.
- Both PIM baselines traverse the pseudo-channel and global reducers.
- Q broadcast, bank-to-PC, PC-to-global, and global-to-NPU traffic contribute
  configurable communication latency.
- Reducer throughput is parameterized by lanes, units, input bandwidth, and
  concurrent destination groups.
- Every query reports its dominant modeled bottleneck and bottleneck fraction.
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
- `skewed_high_degree`: 128 sampled nodes and 2048 power-law, dst-skewed edges.

The default generator configuration is:

```yaml
workload_suite:
  seed: 777
  allow_duplicate_edges: false
  workloads:
    - name: large
      degree_distribution: uniform
      power_law_exponent: 1.2
      destination_skew: 0.0
    - name: skewed_high_degree
      degree_distribution: power_law
      power_law_exponent: 1.2
      destination_skew: 1.5
```

Edges are selected from all directed non-self candidates using reproducible
weighted sampling without replacement. The per-query and aggregate CSV files
report duplicate counts, unique edge counts, in/out-degree statistics, fixed
destination-degree histogram bins, and the number of source hash banks.

Each workload emits `num_queries_per_workload` independent query samples. The
selective KV policy is:

```text
target node union target 1-hop neighbors union high-degree sampled nodes
```

The CSV reports both `selected_kv_ratio_vs_sampled_nodes` and
`selected_kv_ratio_vs_full_graph`.

## Placement Sweep

Patch 4 supports four placement policies:

- `hash`: source-node hash placement.
- `degree_balanced`: sampled nodes are assigned to the least-loaded bank by
  descending degree.
- `source_dst_locality`: edge execution is placed by destination-node hash to
  improve local combine opportunities.
- `hybrid_locality_balanced`: ordinary destinations retain single-bank
  locality, while hot destinations are split across a bounded number of banks.
  A greedy score balances bank pressure, pseudo-channel pressure, and distance
  from the destination's hash anchor.

Hybrid placement is configurable:

```yaml
hybrid_placement:
  hot_dst_degree_threshold: 64
  target_edges_per_bank: 32
  max_banks_per_destination: 16
  locality_weight: 0.25
  bank_balance_weight: 1.0
  pseudo_channel_balance_weight: 1.0
```

The number of shards for a hot destination is:

```text
min(max_banks_per_destination, ceil(in_degree / target_edges_per_bank))
```

The output includes bank and pseudo-channel imbalance, local-combine reduction,
latency, and traffic for each placement.

Placement validation additionally reports mapping difference versus source
hash, active bank and pseudo-channel histograms, edge-count distributions, and
the number and distribution of destination shards.

## Baselines

The simulator retains four diagnostic baselines:

- `h100_ideal_compute_only`
- `h100_realistic_cache_path`
- `pim_selective_kv_no_local_combine`
- `pim_selective_kv_local_combine`

The realistic H100 path includes configurable INT2 K/V read, unpack,
scale/dequant, layout conversion, irregular gather penalty, and small-batch
efficiency penalty.

Patch 4 does not report PIM/H100 speedup. The H100 throughput parameters still
require external calibration before cross-architecture performance claims.

## Fair Reducer Paths

Both PIM paths execute the same hierarchy:

```text
bank output -> pseudo-channel reducer -> global reducer -> NPU
```

Without local combine, every edge partial enters the pseudo-channel reducer.
With local combine, one `(bank,destination)` group enters it and the bank pays
the additional VADD work. Both paths produce one `(pseudo-channel,destination)`
group for the global reducer, so their global input is intentionally identical
after a complete pseudo-channel reduction.

## Communication-Aware Timing

The four modeled links are configured independently:

```yaml
communication:
  q_broadcast_bandwidth_bytes_per_cycle_per_bank: 64
  q_broadcast_startup_cycles: 16
  bank_to_pc_bandwidth_bytes_per_cycle_per_bank: 32
  bank_to_pc_startup_cycles: 16
  pc_to_global_bandwidth_bytes_per_cycle_per_pc: 64
  pc_to_global_startup_cycles: 32
  global_to_npu_bandwidth_bytes_per_cycle: 256
  global_to_npu_startup_cycles: 32
```

Bank-to-PC timing is the maximum of the busiest bank-link serialization and
the busiest pseudo-channel aggregate input. PC-to-global timing is the maximum
of the busiest PC link and the global aggregate input. Startup cost is charged
per memory-token tile.

Reducer parameters are:

```yaml
reducer:
  pc_lanes_per_pseudo_channel: 16
  pc_throughput_groups_per_cycle_per_lane: 1
  pc_input_bandwidth_bytes_per_cycle: 512
  pc_concurrent_destination_groups: 64
  global_units: 1
  global_lanes_per_unit: 64
  global_throughput_groups_per_cycle_per_lane: 1
  global_input_bandwidth_bytes_per_cycle: 1024
  global_concurrent_destination_groups: 256
```

The architecture-level critical path is sequential in Patch 4:

```text
critical_path_cycles = compute_cycles + communication_cycles
                     + reducer_cycles + scheduling_cycles
```

No stage overlap or cycle-accurate contention is modeled yet.

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

## Bottleneck Breakdown

The existing stage totals are retained:

- `compute_cycles`
- `communication_cycles`
- `gnn_score_cycles`
- `gnn_message_cycles`
- `cached_kv_cycles`
- `reducer_cycles`
- `scheduling_cycles`
- `critical_path_cycles`
- `critical_path_latency_ns`
- `traffic_stall_fraction`

PIM rows additionally split these totals into Q8xK8 VDOT, score scale, P8xV8
VMUL, value scale, local VADD, Q8xK2 LUT, P8xV2 LUT, cached-KV scale,
pseudo-channel reduce, and global reduce cycles. The three PE stages also report
the bank that determines their latency.

The realistic H100 rows split the cache path into native INT2 cache read,
unpack, scale/dequant, layout conversion, irregular-gather penalty, and
small-batch penalty cycles.

Each per-query row includes:

- `bottleneck_stage`
- `bottleneck_cycles`
- `bottleneck_fraction`

The aggregate CSV reports the mean of every cycle component plus the dominant
bottleneck stage, the fraction of queries for which it dominates, and mean
bottleneck contribution.

Communication output includes total and critical bank/PC bytes for every link,
along with PC/global reducer input and output group counts. This makes it
possible to distinguish a total-traffic reduction from a critical-path traffic
reduction.

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
5 workloads * 4 placements * 4 baselines * 3 tile sizes = 240 data rows
```

The per-query CSV has:

```text
5 workloads * 16 queries * 4 placements * 4 baselines * 3 tile sizes = 3840 data rows
```

Patch 4 currently emits 121 per-query columns and 108 aggregate columns.
