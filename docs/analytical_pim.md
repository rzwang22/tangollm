# Analytical PIM Simulator

This target is an architecture-level analytical simulator for the HBM3
near-bank PIM design. It is independent from Ramulator and does not model
cycle-accurate DRAM timing or real tensor values.

## Patch 5 Scope

Patch 5 adds a trace-driven execution mode for the formal GOFA workloads while
retaining the Patch 4 synthetic suite as a regression target. It remains an
architecture-level analytical model: no Ramulator, cycle-accurate timing, or
tensor payload execution is introduced.

The formal config reads these tasks from the configured trace root:

- `cora_node_formal_v1`
- `cora_link_formal_v1`
- `pubmed_node_formal_v1`
- `wikics_formal_v1`
- `arxiv_formal_v1`

Queries are loaded in `trace_index.jsonl` order. Validation and test rows are
never aggregated together: validation rows have
`evaluation_role=config_selection`, while test rows have
`evaluation_role=final_evaluation`. Configuration choices must be made from
validation output and then applied unchanged to test output.

## Frozen Trace Evaluation

After the validation sweep, the selected PIM-internal configuration is frozen
in `analytical_pim_trace_final_config.yaml`:

```text
graph compute placement = source_dst_locality
KV storage placement = balanced
memory token tile = 4
design = local combine
comparator = no local combine
```

The final config retains validation for distribution-shift reporting and test
for final evaluation, but it contains no remaining placement or tile sweep. It
produces 2,000 per-query rows and 20 aggregate rows for the five formal tasks.
Run it from the build directory and validate the result with:

```bash
./analytical_pim analytical_pim_trace_final_config.yaml \
  | tee final_trace_evaluation.log

python3 ../tools/analyze_gofa_trace_final.py \
  --per-query ../data/analytical_pim_trace_final_per_query.csv \
  --aggregate ../data/analytical_pim_trace_final_aggregate.csv \
  --output ../data/analytical_pim_trace_final_summary.csv \
  --require-final-only
```

The analyzer checks row counts, evaluation roles, trace ordering, equality of
the query sets compared by the two PIM baselines, aggregate/per-query
consistency, corrected VADD semantics, and local-buffer capacity. It reports
test mean/p50/p95 latency, local-combine traffic and reducer reductions,
validation-to-test shift, buffer use, traffic stalls, and the dominant modeled
bottleneck. The `__all__` row computes exact percentiles over all 500 test
queries rather than averaging per-workload percentiles. The report is a
PIM-internal comparison and must not be presented as PIM/H100 speedup.

## Patch 6 Calibration Envelope

Patch 6 performs validation-only sensitivity analysis around the frozen Patch
5 design. `analytical_pim_patch6_config.yaml` still loads and validates all
1,000 formal traces, but this field prevents test queries from entering the
simulation or any sensitivity CSV:

```yaml
trace_workload:
  splits: [validation, test]
  simulation_splits: [validation]
```

The legacy `scale_group_cycles` remains the fallback for compatibility. Patch
6 separates its two architectural uses so cached-KV dequantization can be
varied without changing GNN score/value scaling:

```yaml
near_bank_pe:
  scale_group_cycles: 1
  gnn_scale_group_cycles: 1
  cached_kv_scale_group_cycles: 1
  cached_kv_lut_scale_overlap: 0
```

Cached-KV LUT selection and group-wise scale/dequant can overlap:

```text
effective_cycles = lut_cycles + scale_cycles
                 - overlap * min(lut_cycles, scale_cycles)
```

`overlap=0` preserves the Patch 5 serial model, `0.5` hides half of the
shorter stage, and `1` represents full pipeline overlap. The CSV retains the
raw LUT and scale costs, their unoverlapped sum, and hidden cycles.

The attribution fields separate raw stage composition from effective latency:

- `cached_kv_raw_lut_fraction` and `cached_kv_raw_scale_fraction` sum to one;
- `cached_kv_overlap_hidden_fraction` is the fraction of serial cached-KV
  cycles hidden by overlap;
- `cached_kv_pipeline_compute_fraction` and
  `cached_kv_pipeline_critical_path_fraction` report the effective cached-KV
  contribution after overlap.

When overlap is nonzero, the effective bottleneck is reported as
`pim_cached_kv_lut_scale_pipeline`. It is not attributed to LUT or scale alone.

The executable accepts a strict whitelist of runtime overrides. This avoids
rewriting YAML with string substitutions and rejects unknown or non-positive
parameters:

```bash
./analytical_pim analytical_pim_patch6_config.yaml \
  --set near_bank_pe.cached_kv_scale_group_cycles=2 \
  --set output.per_query_csv=/tmp/patch6_per_query.csv \
  --set output.aggregate_csv=/tmp/patch6_aggregate.csv
```

Run the complete Patch 6 envelope from the build directory:

```bash
python3 ../tools/run_patch6_sensitivity.py \
  --binary ./analytical_pim \
  --config ./analytical_pim_patch6_config.yaml \
  --output-directory ../data/patch6 \
  | tee patch6_sensitivity.log
```

The 17 scenarios include the base point, 50% and 100% cached-KV LUT/scale
overlap, fast/slow single-factor sweeps for cached-KV scale, INT2 LUT, GNN
pipeline, VADD, communication, and reducer, plus optimistic/conservative joint
envelopes. Cycle costs use `0.5x/1x/2x` and bandwidth or throughput uses
`2x/1x/0.5x` for optimistic/base/conservative assumptions.

Patch 6 writes:

- `patch6_parameter_manifest.csv`: parameter units, envelopes, and calibration
  status for 16 parameters;
- `patch6_sensitivity_by_workload.csv`: 85 scenario/workload rows;
- `patch6_sensitivity_overall.csv`: 17 exact 500-query summaries;
- `raw/`: per-scenario simulator CSV and logs.

After the single-factor envelope, run the cached-KV joint calibration grid:

```bash
python3 ../tools/run_patch6_cached_kv_grid.py \
  --binary ./analytical_pim \
  --config ./analytical_pim_patch6_config.yaml \
  --output-directory ../data/patch6_cached_kv_grid \
  | tee patch6_cached_kv_grid.log
```

The default grid ties Q8xK2 and P8xV2 LUT throughput and sweeps:

```text
INT2 LUT cycles/group = {0.5, 1, 2}
scale cycles/group    = {0.25, 0.5, 1, 2}
LUT/scale overlap     = {0, 0.5, 1}
```

It produces 36 validation-only points, 180 per-workload rows, and a 36-row
scenario manifest. `LUT=1, scale=1, overlap=0` is the reference point. Each row
reports mean/p95 latency, change from the reference, effective cached-KV path
share, raw LUT/scale composition, hidden fraction, local-combine gain, and
bottleneck stage/share. These are uncalibrated sensitivity points; the grid
does not select a hardware configuration or evaluate test traces.

The next analysis maps local-combine break-even conditions:

```bash
python3 ../tools/run_patch6_local_combine_break_even.py \
  --binary ./analytical_pim \
  --config ./analytical_pim_patch6_config.yaml \
  --output-directory ../data/patch6_local_break_even \
  | tee patch6_local_break_even.log
```

Three explicitly uncalibrated cached-KV anchors are evaluated:

```text
conservative_serial  = LUT 2,   scale 2,    overlap 0
reference_serial     = LUT 1,   scale 1,    overlap 0
optimistic_pipelined = LUT 0.5, scale 0.25, overlap 1
```

For each anchor, the default grid sweeps VADD cycles/group over
`{0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 4}` and bank-to-PC bandwidth over
`{8, 16, 32, 64, 128}` bytes/cycle/bank. The 120 points produce 600
per-workload rows. Linear interpolation between the nearest winning and losing
VADD points estimates a break-even threshold for every
`anchor x workload x bandwidth` group. The runner also verifies that cached-KV
anchors do not change the absolute local/no-local saving; anchors only change
its relative importance in total latency.

The interpolation criterion is equality of local and no-local mean latency over
the selected validation queries. Both paths' p95 latency remains available in
the grid CSV but is not used to define the threshold.

Within each cached-KV anchor, `VADD=1` and bank-to-PC bandwidth `32` define the
latency reference. Custom sweeps must retain that point.

The output includes the grid manifest, overall and per-workload metrics, raw
scenario files, and `patch6_local_combine_break_even.csv` with 90 threshold
rows. Test traces remain excluded.

Every parameter is labeled `uncalibrated_assumption`. These sweeps measure
model sensitivity and bottleneck transitions; they are not hardware
calibration evidence. Test results remain frozen from Patch 5, and Patch 6 does
not compute PIM/H100 speedup or use Ramulator.

For formal traces, the simulator directly consumes:

- sampled graph structure and NOG metadata;
- memory-state and text-K/V inventory;
- selected K and V item masks for every suffix layer;
- quantized logical bytes excluding scale/container metadata;
- valid and stored token counts;
- runtime QK, PV, GNN node, and GNN edge shapes.

The strict validator cross-checks every query against its index entry and
cross-checks all indexed queries against `summary.json`. It also validates
model dimensions, graph mappings, NOG placement, inventory types and
precision, selection sets/masks, per-layer sharing, runtime item order,
QK/PV token dimensions, logical-byte accounting, selection ratios, split
counts, split order, and exact enumeration of `query_*.json` files. Tensor
payload fields are rejected.

Formal trace execution does not use the synthetic
`selected_count * 786432` estimate. Cached-KV work and bytes come from the
per-item, per-layer K/V shapes in each trace.

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

## Selected-KV Placement Diagnosis

Selected-KV storage placement is diagnosed independently from edge execution
placement. The per-query and aggregate CSV files include:

- `selected_kv_active_banks`
- `selected_kv_items_per_bank_mean`
- `selected_kv_items_per_bank_p95`
- `selected_kv_items_per_bank_max`
- `selected_kv_bank_imbalance`
- `selected_kv_bank_collision_ratio`

The item-count statistics and imbalance use active banks only. Imbalance is
`max / mean`. Collision ratio is the fraction of selected items left after
assigning one item to each active bank:

```text
(selected_kv_count - selected_kv_active_banks) / selected_kv_count
```

The per-query CSV retains `cached_kv_bottleneck_bank`. The aggregate CSV adds
`dominant_cached_kv_bottleneck_bank` and the fraction of queries mapped to that
bank. These fields explain why total selected-KV count and cached-KV
critical-path latency need not be monotonic.

Patch 5 separates graph compute placement from KV storage placement:

```yaml
graph_compute_placement_sweep:
  - hash
  - degree_balanced
  - source_dst_locality
  - hybrid_locality_balanced

kv_storage_placement_sweep:
  - hash
  - balanced
```

`hash` maps a trace cache key independently of graph node placement.
`balanced` greedily assigns the heaviest QK/PV item to the least-loaded bank.
Synthetic regression uses `legacy_graph_coupled`, which exactly preserves the
Patch 4 mapping and numerical results.

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

The first Patch 5 formal run includes only the two PIM-internal alternatives:

```yaml
baselines:
  - pim_selective_kv_no_local_combine
  - pim_selective_kv_local_combine
```

It is workload characterization and internal design comparison only; it does
not report PIM/H100 speedup.

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

The first partial message for each `(bank,destination)` group initializes its
accumulator without a VADD. Only subsequent partials execute VADD:

```text
VADD input edge groups = all edge message groups
VADD initialization groups = bank-local destination groups
VADD operation groups = input edge groups - initialization groups
```

The CSV reports all three group counts. `local_vadd_cycles` is the critical
path of VADD operation groups on the busiest bank, not the sum across banks.

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

## Near-Bank Local Buffer

The local-combine accumulator is modeled as a per-bank near-bank SRAM or
register-file resource. Its residency is configured independently from edge
multiplicity:

```yaml
local_buffer:
  resident_head_tiles: 1
  concurrent_destinations_per_bank: 1
  capacity_bytes_per_bank: 4096
```

For one resident destination, the required bytes are:

```text
bytes_per_group = memory_token_tile * head_tile
                * channel_group * partial_message_bytes

bytes_per_head_tile = groups_per_head * bytes_per_group

bytes_per_destination = resident_head_tiles * bytes_per_head_tile

required_bytes_per_bank = min(configured_concurrent_destinations,
                              active_destinations_on_busiest_bank)
                        * bytes_per_destination
```

Edges with the same `(bank,destination)` stream through VADD into the same
accumulator. Edge count therefore does not multiply SRAM capacity. The output
fields report per-group, per-head-tile, per-destination, required, configured,
overflow, and utilization values. Capacity overflow is diagnostic in this
patch; it does not yet add spill traffic or serialization latency.

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

For overlapped cached-KV execution, raw LUT and scale fractions describe stage
composition, while the effective pipeline fractions describe latency
contribution. The bottleneck stage is the combined LUT/scale pipeline.

The realistic H100 rows split the cache path into native INT2 cache read,
unpack, scale/dequant, layout conversion, irregular-gather penalty, and
small-batch penalty cycles.

Each per-query row includes:

- `bottleneck_stage`
- `bottleneck_cycles`
- `bottleneck_fraction`

The aggregate CSV reports the mean of every cycle component plus the dominant
bottleneck stage, the fraction of queries for which it dominates, and mean
bottleneck contribution. The stage contribution and query-label fraction are
different quantities: a stage can dominate every query without accounting for
100% of latency.

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

Run the formal GOFA trace suite on the server with:

```bash
./analytical_pim analytical_pim_trace_config.yaml
```

The trace root is configured in `analytical_pim_trace_config.yaml`:

```text
/home/rzwang/data/GOFA/cache_data/gofa_cache_exp/
canonical_seed1_h3_n10_s100/query_traces
```

Outputs:

```text
../data/analytical_pim_per_query.csv
../data/analytical_pim_aggregate.csv
```

Formal trace outputs are separate:

```text
../data/analytical_pim_trace_per_query.csv
../data/analytical_pim_trace_aggregate.csv
```

With the default config, the aggregate CSV has:

```text
5 workloads * 4 placements * 4 baselines * 3 tile sizes = 240 data rows
```

The per-query CSV has:

```text
5 workloads * 16 queries * 4 placements * 4 baselines * 3 tile sizes = 3840 data rows
```

Patch 4 with selected-KV placement diagnosis emits 127 per-query columns and
116 aggregate columns.

Patch 5 with the corrected local-buffer and VADD models emits 184 per-query
columns and 169 aggregate columns. The additional
fields carry trace identity, split/evaluation role, target/question and cache
inventory counts, independent graph/KV placement, native cache bytes including
edge-cache bytes, token counts, QK/PV work, NOG shapes, and per-bank local
buffer residency/capacity diagnostics.

The formal config produces:

```text
1000 queries * 4 graph placements * 2 KV placements
             * 2 PIM baselines * 3 tile sizes = 48000 per-query rows

5 workloads * 2 splits * 4 graph placements * 2 KV placements
            * 2 PIM baselines * 3 tile sizes = 480 aggregate rows
```

The local fixture under `tests/fixtures/gofa_trace_v1` contains one validation
and one test query. It uses nonuniform 3/5/7/11-token items so the end-to-end
test detects any fallback to fixed 256-token or 786432-byte item estimates.
Run its positive and negative validator tests from the repository root:

```bash
python3 tests/test_analytical_pim_trace.py --binary build/analytical_pim
```
