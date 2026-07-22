#!/usr/bin/env python3

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path


def write_config(directory: Path, stacks: int, inconsistent: bool = False) -> Path:
    banks_per_pc = 31 if inconsistent else 32
    config = directory / f"hbm2e_{stacks}stack.yaml"
    config.write_text(
        f"""workload:
  mode: synthetic
model:
  memory_tokens: 128
  hidden_dim: 4096
  gnn_heads: 32
  head_dim: 128
  suffix_layers: 6
  text_len: 256
  kv_heads: 8
  kv_bits: 2
  include_scale_metadata: false
  scale_group_channels: 16
  scale_bytes: 2
hbm2e:
  num_stacks: {stacks}
  dies_per_stack: 8
  channels_per_stack: 8
  pseudo_channels_per_channel: 2
  sid_count: 2
  banks_per_sid_per_pseudo_channel: 16
  banks_per_pseudo_channel: {banks_per_pc}
  pe_per_bank: 1
near_bank_pe:
  q8k8_group_cycles: 1
  p8v8_group_cycles: 1
  q8k2_lut_group_cycles: 1
  p8v2_lut_group_cycles: 1
  vadd_group_cycles: 1
  scale_group_cycles: 1
  cached_kv_lut_scale_overlap: 0
  softmax_group_cycles: 1
  scheduling_overhead_per_tile_cycles: 16
  clock_ns: 1
tile:
  channel_group: 16
  head_tile: 1
  memory_token_tiles: [4]
local_buffer:
  resident_head_tiles: 1
  concurrent_destinations_per_bank: 1
  capacity_bytes_per_bank: 4096
workload_suite:
  full_graph_nodes: 2708
  full_graph_edges: 10556
  num_queries_per_workload: 1
  high_degree_top_percent: 0.05
  seed: 777
  allow_duplicate_edges: false
  workloads:
    - name: topology_test
      sampled_nodes: 64
      sampled_edges: 512
      degree_distribution: uniform
      destination_skew: 0
communication:
  q_broadcast_bandwidth_bytes_per_cycle_per_bank: 64
  q_broadcast_startup_cycles: 16
  bank_to_pc_bandwidth_bytes_per_cycle_per_bank: 32
  bank_to_pc_startup_cycles: 16
  pc_to_global_bandwidth_bytes_per_cycle_per_pc: 64
  pc_to_global_startup_cycles: 32
  global_to_npu_bandwidth_bytes_per_cycle_per_stack: 256
  global_to_npu_aggregate_bandwidth_bytes_per_cycle: {256 * stacks}
  global_to_npu_startup_cycles: 32
reducer:
  pc_lanes_per_pseudo_channel: 16
  pc_throughput_groups_per_cycle_per_lane: 1
  pc_input_bandwidth_bytes_per_cycle: 512
  pc_concurrent_destination_groups: 64
  global_units_per_stack: 1
  global_lanes_per_unit: 64
  global_throughput_groups_per_cycle_per_lane: 1
  global_input_bandwidth_bytes_per_cycle_per_stack: 1024
  global_concurrent_destination_groups: 256
  cross_stack_merge_lanes: 64
  cross_stack_merge_throughput_groups_per_cycle_per_lane: 1
graph_compute_placement_sweep: [hash]
kv_storage_placement_sweep: [balanced]
baselines: [pim_selective_kv_local_combine]
output:
  per_query_csv: {directory / f'per_query_{stacks}.csv'}
  aggregate_csv: {directory / f'aggregate_{stacks}.csv'}
""",
        encoding="utf-8",
    )
    return config


def run(binary: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), str(config)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1, len(rows)
    assert None not in rows[0]
    assert all(value is not None for value in rows[0].values())
    return rows[0]


def assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) < 1e-6 * max(1.0, abs(expected)), (
        actual,
        expected,
    )


def check_topology(binary: Path, directory: Path, stacks: int) -> dict[str, str]:
    completed = run(binary, write_config(directory, stacks))
    assert completed.returncode == 0, completed.stdout
    expected_banks = 512 * stacks
    assert f"memory_topology=hbm2e, stacks={stacks}" in completed.stdout
    assert "banks_per_pseudo_channel=32" in completed.stdout
    assert f"total_banks={expected_banks}" in completed.stdout

    row = read_single_row(directory / f"per_query_{stacks}.csv")
    aggregate = read_single_row(directory / f"aggregate_{stacks}.csv")
    assert int(row["topology_num_stacks"]) == stacks
    assert int(row["topology_banks_per_stack"]) == 512
    assert int(row["topology_banks_per_pseudo_channel"]) == 32
    assert int(row["topology_total_banks"]) == expected_banks
    assert int(row["topology_sid_count"]) == 2
    assert int(aggregate["topology_total_banks"]) == expected_banks
    assert 1 <= int(row["active_stacks"]) <= stacks

    critical_groups = float(row["global_reducer_critical_stack_input_groups"])
    total_groups = float(row["global_reducer_input_groups"])
    assert 0 < critical_groups <= total_groups
    assert_close(float(row["global_reduce_cycles"]), critical_groups / 64.0)
    assert_close(
        float(row["cross_stack_merge_cycles"]),
        float(row["cross_stack_duplicate_output_groups"]) / 64.0,
    )

    startup = int(row["num_token_tiles"]) * int(row["gnn_layer_count"]) * 32
    expected_global_to_npu = startup + max(
        float(row["global_to_npu_critical_stack_bytes"]) / 256.0,
        float(row["global_to_npu_bytes"]) / (256.0 * stacks),
    )
    assert_close(
        float(row["global_to_npu_communication_cycles"]),
        expected_global_to_npu,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()

    with tempfile.TemporaryDirectory(prefix="hbm2e-8hi-") as temporary:
        directory = Path(temporary)
        one_stack = check_topology(binary, directory, 1)
        five_stack = check_topology(binary, directory, 5)
        assert int(float(one_stack["selected_kv_active_stacks"])) == 1
        assert int(float(five_stack["selected_kv_active_stacks"])) > 1

        invalid = run(binary, write_config(directory, 5, inconsistent=True))
        assert invalid.returncode != 0
        assert "banks_per_pseudo_channel must equal" in invalid.stdout

    print("HBM2E 8-Hi topology tests passed: 512 banks/stack, 2560 banks/5 stacks")


if __name__ == "__main__":
    main()
