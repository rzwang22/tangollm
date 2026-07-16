#!/usr/bin/env python3
"""End-to-end and negative tests for the Patch 5 GOFA trace path."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures/gofa_trace_v1"
CONFIG_TEMPLATE = REPOSITORY_ROOT / "tests/analytical_pim_trace_fixture_config.yaml"
TASK_DIRECTORY = "fixture_task_formal_v1"
FINAL_ANALYZER = REPOSITORY_ROOT / "tools/analyze_gofa_trace_final.py"
PATCH6_RUNNER = REPOSITORY_ROOT / "tools/run_patch6_sensitivity.py"


def write_config(directory: Path, trace_root: Path) -> Path:
    config = CONFIG_TEMPLATE.read_text()
    config = config.replace(
        "root: tests/fixtures/gofa_trace_v1", f"root: {trace_root}"
    )
    config = config.replace(
        "/tmp/analytical_pim_trace_fixture_per_query.csv",
        str(directory / "per_query.csv"),
    )
    config = config.replace(
        "/tmp/analytical_pim_trace_fixture_aggregate.csv",
        str(directory / "aggregate.csv"),
    )
    config_path = directory / "config.yaml"
    config_path.write_text(config)
    return config_path


def run_simulator(
    binary: Path, config: Path, overrides: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    command = [str(binary), str(config)]
    for override in overrides:
        command.extend(("--set", override))
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: str, expected: float) -> None:
    assert abs(float(actual) - expected) < 1e-6, (actual, expected)


def check_valid_run(binary: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="patch5-valid-") as temporary:
        directory = Path(temporary)
        config = write_config(directory, FIXTURE_ROOT)
        completed = run_simulator(binary, config)
        assert completed.returncode == 0, completed.stdout
        assert (
            "GOFA trace validation passed: queries=2, validation=1, test=1"
            in completed.stdout
        )

        per_query = load_rows(directory / "per_query.csv")
        aggregate = load_rows(directory / "aggregate.csv")
        assert len(per_query) == 32
        assert len(aggregate) == 32
        assert len(per_query[0]) == 189
        assert len(aggregate[0]) == 174
        assert all(
            None not in row and all(value is not None for value in row.values())
            for row in per_query
        )
        assert all(
            None not in row and all(value is not None for value in row.values())
            for row in aggregate
        )
        assert [int(row["trace_order"]) for row in per_query] == [0] * 16 + [1] * 16
        assert {row["split"] for row in aggregate} == {"validation", "test"}
        assert {row["evaluation_role"] for row in aggregate} == {
            "config_selection",
            "final_evaluation",
        }
        assert all(row["num_queries"] == "1" for row in aggregate)
        assert not any(row["baseline"].startswith("h100") for row in aggregate)

        required_columns = {
            "runtime_query_index",
            "target_node_count",
            "question_node_count",
            "total_item_count",
            "node_text_item_count",
            "edge_text_item_count",
            "memory_cache_bytes",
            "edge_cache_bytes",
            "full_key_bytes",
            "full_value_bytes",
            "selected_key_bytes",
            "selected_value_bytes",
            "selected_key_ratio",
            "selected_value_ratio",
            "selected_kv_ratio",
            "runtime_loaded_cache_bytes",
            "cache_inventory_sequence_tokens",
            "cache_inventory_valid_text_tokens",
            "selected_sequence_tokens",
            "valid_text_tokens",
            "stored_key_tokens",
            "stored_value_tokens",
            "trace_cached_qk_groups",
            "trace_cached_pv_groups",
            "runtime_qk_elements",
            "runtime_pv_output_elements",
            "nog_runtime_qk_elements",
            "nog_runtime_pv_output_elements",
            "local_buffer_bytes_per_group",
            "local_buffer_bytes_per_head_tile",
            "local_buffer_bytes_per_destination",
            "local_buffer_resident_head_tiles",
            "local_buffer_concurrent_destinations",
            "local_buffer_active_destinations_max",
            "local_buffer_capacity_bytes_per_bank",
            "local_buffer_overflow_bytes_per_bank",
            "local_buffer_capacity_utilization",
            "local_buffer_capacity_exceeded",
            "local_vadd_input_edge_groups",
            "local_vadd_initialization_groups",
            "local_vadd_operation_groups",
            "cached_kv_lut_scale_overlap",
            "cached_kv_critical_lut_cycles",
            "cached_kv_critical_scale_cycles",
            "cached_kv_unoverlapped_cycles",
            "cached_kv_overlap_hidden_cycles",
        }
        assert required_columns <= set(per_query[0])

        row = per_query[0]
        assert row["trace_query_id"] == "query_000000"
        assert row["selection_policy"] == "target_1hop"
        assert row["runtime_query_index"] == "0"
        assert row["target_node_count"] == "1"
        assert row["question_node_count"] == "1"
        assert row["total_item_count"] == "4"
        assert row["node_text_item_count"] == "3"
        assert row["edge_text_item_count"] == "1"
        assert_close(row["edge_cache_bytes"], 295936)
        assert_close(row["selected_kv_bytes"], 43008)
        assert_close(row["selected_key_bytes"], 21504)
        assert_close(row["selected_value_bytes"], 21504)
        assert_close(row["selected_key_ratio"], 2 / 3)
        assert_close(row["selected_value_ratio"], 2 / 3)
        assert_close(row["selected_kv_ratio"], 2 / 3)
        assert_close(row["runtime_loaded_cache_bytes"], 829440)
        assert_close(row["cache_inventory_sequence_tokens"], 538)
        assert_close(row["cache_inventory_valid_text_tokens"], 26)
        assert_close(row["selected_sequence_tokens"], 270)
        assert_close(row["valid_text_tokens"], 14)
        assert_close(row["stored_key_tokens"], 84)
        assert_close(row["stored_value_tokens"], 84)
        assert_close(row["trace_cached_qk_groups"], 2752512)
        assert_close(row["trace_cached_pv_groups"], 2752512)
        assert_close(row["runtime_qk_elements"], 9953280)
        assert_close(row["runtime_pv_output_elements"], 9437184)
        assert_close(row["nog_runtime_qk_elements"], 3317760)
        assert_close(row["nog_runtime_pv_output_elements"], 3145728)
        assert float(row["selected_kv_bytes"]) != 786432

        def local_buffer_row(tile: str, baseline: str) -> dict[str, str]:
            return next(
                row
                for row in per_query
                if row["split"] == "validation"
                and row["graph_compute_placement"] == "hash"
                and row["kv_storage_placement"] == "balanced"
                and row["baseline"] == baseline
                and row["memory_token_tile"] == tile
            )

        local_t1 = local_buffer_row("1", "pim_selective_kv_local_combine")
        local_t4 = local_buffer_row("4", "pim_selective_kv_local_combine")
        no_local_t4 = local_buffer_row(
            "4", "pim_selective_kv_no_local_combine"
        )
        assert_close(local_t1["local_buffer_bytes_per_group"], 32)
        assert_close(local_t1["local_buffer_bytes_per_destination"], 256)
        assert_close(local_t1["local_combine_buffer_max_bytes"], 256)
        assert_close(local_t4["local_buffer_bytes_per_group"], 128)
        assert_close(local_t4["local_buffer_bytes_per_head_tile"], 1024)
        assert_close(local_t4["local_buffer_bytes_per_destination"], 1024)
        assert local_t4["local_buffer_resident_head_tiles"] == "1"
        assert local_t4["local_buffer_concurrent_destinations"] == "1"
        assert local_t4["local_buffer_active_destinations_max"] == "1"
        assert_close(local_t4["local_buffer_capacity_bytes_per_bank"], 4096)
        assert_close(local_t4["local_combine_buffer_max_bytes"], 1024)
        assert_close(local_t4["local_buffer_overflow_bytes_per_bank"], 0)
        assert_close(local_t4["local_buffer_capacity_utilization"], 0.25)
        assert local_t4["local_buffer_capacity_exceeded"] == "0"
        assert_close(local_t4["local_vadd_input_edge_groups"], 393216)
        assert_close(local_t4["local_vadd_initialization_groups"], 393216)
        assert_close(local_t4["local_vadd_operation_groups"], 0)
        assert_close(local_t4["local_vadd_cycles"], 0)
        assert_close(no_local_t4["local_combine_buffer_max_bytes"], 0)
        assert no_local_t4["local_buffer_concurrent_destinations"] == "0"
        assert_close(no_local_t4["local_vadd_input_edge_groups"], 0)
        assert_close(no_local_t4["local_vadd_initialization_groups"], 0)
        assert_close(no_local_t4["local_vadd_operation_groups"], 0)

        combinations = {
            (row["graph_compute_placement"], row["kv_storage_placement"])
            for row in aggregate
        }
        assert combinations == {
            ("hash", "hash"),
            ("hash", "balanced"),
            ("hybrid_locality_balanced", "hash"),
            ("hybrid_locality_balanced", "balanced"),
        }
        comparison = {
            row["kv_storage_placement"]: row
            for row in aggregate
            if row["split"] == "validation"
            and row["graph_compute_placement"] == "hash"
            and row["baseline"] == "pim_selective_kv_local_combine"
            and row["memory_token_tile"] == "4"
        }
        assert_close(comparison["hash"]["selected_kv_active_banks"], 1)
        assert_close(comparison["balanced"]["selected_kv_active_banks"], 2)
        assert float(comparison["hash"]["mean_cached_kv_cycles"]) > float(
            comparison["balanced"]["mean_cached_kv_cycles"]
        )

        graph_comparison = {
            row["graph_compute_placement"]: row
            for row in aggregate
            if row["split"] == "validation"
            and row["kv_storage_placement"] == "hash"
            and row["baseline"] == "pim_selective_kv_local_combine"
            and row["memory_token_tile"] == "4"
        }
        assert_close(
            graph_comparison["hash"]["mapping_difference_ratio_vs_hash"], 0
        )
        assert_close(
            graph_comparison["hybrid_locality_balanced"][
                "mapping_difference_ratio_vs_hash"
            ],
            1,
        )
        assert_close(graph_comparison["hash"]["selected_kv_active_banks"], 1)
        assert_close(
            graph_comparison["hybrid_locality_balanced"][
                "selected_kv_active_banks"
            ],
            1,
        )
        assert_close(
            graph_comparison["hash"]["mean_cached_kv_cycles"],
            float(
                graph_comparison["hybrid_locality_balanced"][
                    "mean_cached_kv_cycles"
                ]
            ),
        )

        final_config = directory / "final_config.yaml"
        final_config.write_text(
            config.read_text()
            .replace("memory_token_tiles: [1, 4]", "memory_token_tiles: [4]")
            .replace(
                "graph_compute_placement_sweep: [hash, hybrid_locality_balanced]",
                "graph_compute_placement_sweep: [source_dst_locality]",
            )
            .replace(
                "kv_storage_placement_sweep: [hash, balanced]",
                "kv_storage_placement_sweep: [balanced]",
            )
        )
        final_completed = run_simulator(binary, final_config)
        assert final_completed.returncode == 0, final_completed.stdout
        final_per_query = load_rows(directory / "per_query.csv")
        final_aggregate = load_rows(directory / "aggregate.csv")
        assert len(final_per_query) == 4
        assert len(final_aggregate) == 4

        summary_path = directory / "final_summary.csv"
        analyzer_command = [
            sys.executable,
            str(FINAL_ANALYZER),
            "--per-query",
            str(directory / "per_query.csv"),
            "--aggregate",
            str(directory / "aggregate.csv"),
            "--output",
            str(summary_path),
            "--expected-workloads",
            "fixture_task",
            "--expected-queries-per-split",
            "1",
            "--require-final-only",
        ]
        analysis_completed = subprocess.run(
            analyzer_command,
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert analysis_completed.returncode == 0, analysis_completed.stdout
        assert "Final trace checks passed: queries=2" in analysis_completed.stdout
        summary_rows = load_rows(summary_path)
        assert len(summary_rows) == 2
        assert summary_rows[0]["workload"] == "fixture_task"
        assert summary_rows[0]["query_count"] == "1"
        assert summary_rows[1]["workload"] == "__all__"
        assert summary_rows[1]["query_count"] == "1"

        corrupt_path = directory / "corrupt_per_query.csv"
        corrupt_rows = final_per_query.copy()
        corrupt_row = next(
            row
            for row in corrupt_rows
            if row["baseline"] == "pim_selective_kv_local_combine"
        )
        corrupt_row["local_vadd_operation_groups"] = str(
            float(corrupt_row["local_vadd_operation_groups"]) + 1
        )
        with corrupt_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(corrupt_rows[0]))
            writer.writeheader()
            writer.writerows(corrupt_rows)
        corrupt_command = analyzer_command.copy()
        corrupt_command[corrupt_command.index(str(directory / "per_query.csv"))] = str(
            corrupt_path
        )
        corrupt_completed = subprocess.run(
            corrupt_command,
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert corrupt_completed.returncode != 0
        assert "VADD count identity failed" in corrupt_completed.stdout

        config.write_text(
            config.read_text().replace(
                "capacity_bytes_per_bank: 4096",
                "capacity_bytes_per_bank: 512",
            )
        )
        overflow_completed = run_simulator(binary, config)
        assert overflow_completed.returncode == 0, overflow_completed.stdout
        overflow_rows = load_rows(directory / "per_query.csv")
        overflow_t4 = next(
            row
            for row in overflow_rows
            if row["split"] == "validation"
            and row["graph_compute_placement"] == "hash"
            and row["kv_storage_placement"] == "balanced"
            and row["baseline"] == "pim_selective_kv_local_combine"
            and row["memory_token_tile"] == "4"
        )
        assert_close(overflow_t4["local_combine_buffer_max_bytes"], 1024)
        assert_close(overflow_t4["local_buffer_capacity_bytes_per_bank"], 512)
        assert_close(overflow_t4["local_buffer_overflow_bytes_per_bank"], 512)
        assert_close(overflow_t4["local_buffer_capacity_utilization"], 2)
        assert overflow_t4["local_buffer_capacity_exceeded"] == "1"
        assert_close(
            overflow_t4["critical_path_latency_ns"],
            float(local_t4["critical_path_latency_ns"]),
        )


Mutation = Callable[[Path], None]


def query_path(trace_root: Path) -> Path:
    return trace_root / TASK_DIRECTORY / "query_000000.json"


def mutate_json(path: Path, mutation: Callable[[dict], None]) -> None:
    payload = json.loads(path.read_text())
    mutation(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def expect_validation_failure(
    binary: Path, name: str, mutation: Mutation, expected_message: str
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"patch5-{name}-") as temporary:
        directory = Path(temporary)
        trace_root = directory / "traces"
        shutil.copytree(FIXTURE_ROOT, trace_root)
        mutation(trace_root)
        config = write_config(directory, trace_root)
        completed = run_simulator(binary, config)
        assert completed.returncode != 0, f"{name} unexpectedly passed"
        assert expected_message in completed.stdout, completed.stdout


def check_negative_cases(binary: Path) -> None:
    def index_mismatch(root: Path) -> None:
        path = root / TASK_DIRECTORY / "trace_index.jsonl"
        lines = path.read_text().splitlines()
        entry = json.loads(lines[0])
        entry["query_id"] = "query_wrong"
        lines[0] = json.dumps(entry, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")

    def summary_count(root: Path) -> None:
        mutate_json(
            root / TASK_DIRECTORY / "summary.json",
            lambda payload: payload["tasks"]["fixture_task"].__setitem__(
                "query_count", 3
            ),
        )

    def inventory_selection(root: Path) -> None:
        mutate_json(
            query_path(root),
            lambda payload: payload["selective_kv_access"].__setitem__(
                "eligible_item_indices", [0, 3]
            ),
        )

    def kv_mask(root: Path) -> None:
        def mutation(payload: dict) -> None:
            payload["selective_kv_access"]["key_item_mask"][0] = False

        mutate_json(query_path(root), mutation)

    def bytes_mismatch(root: Path) -> None:
        def mutation(payload: dict) -> None:
            payload["traffic_metadata"]["runtime_loaded_cache_bytes"] += 1

        mutate_json(query_path(root), mutation)

    def shape_token_mismatch(root: Path) -> None:
        def mutation(payload: dict) -> None:
            item = payload["runtime_operation_shapes"]["layers"][0]["items"][
                0
            ]
            item["qk_shape"][-1] += 1
            item["softmax_probability_shape"][-1] += 1

        mutate_json(query_path(root), mutation)

    cases = [
        ("index", index_mismatch, "query_id mismatch"),
        ("summary", summary_count, "summary query_count mismatch"),
        ("inventory", inventory_selection, "eligible item inventory mismatch"),
        ("mask", kv_mask, "key_item_mask does not match selected K items"),
        ("bytes", bytes_mismatch, "trace traffic metadata mismatch"),
        (
            "shape",
            shape_token_mismatch,
            "runtime QK token dimension does not match selected/online K",
        ),
    ]
    for name, mutation, expected_message in cases:
        expect_validation_failure(binary, name, mutation, expected_message)


def check_patch6_run(binary: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="patch6-valid-") as temporary:
        directory = Path(temporary)
        config = write_config(directory, FIXTURE_ROOT)
        config.write_text(
            config.read_text()
            .replace(
                "splits: [validation, test]",
                "splits: [validation, test]\n  simulation_splits: [validation]",
            )
            .replace(
                "scale_group_cycles: 1",
                "scale_group_cycles: 1\n"
                "  gnn_scale_group_cycles: 1\n"
                "  cached_kv_scale_group_cycles: 1\n"
                "  cached_kv_lut_scale_overlap: 0",
            )
            .replace("memory_token_tiles: [1, 4]", "memory_token_tiles: [4]")
            .replace(
                "graph_compute_placement_sweep: [hash, hybrid_locality_balanced]",
                "graph_compute_placement_sweep: [source_dst_locality]",
            )
            .replace(
                "kv_storage_placement_sweep: [hash, balanced]",
                "kv_storage_placement_sweep: [balanced]",
            )
        )

        base_completed = run_simulator(binary, config)
        assert base_completed.returncode == 0, base_completed.stdout
        assert "Trace simulation selection: queries=1, splits=validation" in base_completed.stdout
        base_rows = load_rows(directory / "per_query.csv")
        base_aggregate = load_rows(directory / "aggregate.csv")
        assert len(base_rows) == 2
        assert len(base_aggregate) == 2
        assert {row["split"] for row in base_rows} == {"validation"}
        base_local = next(
            row
            for row in base_rows
            if row["baseline"] == "pim_selective_kv_local_combine"
        )
        assert_close(base_local["cached_kv_lut_scale_overlap"], 0)
        assert_close(base_local["cached_kv_overlap_hidden_cycles"], 0)
        assert_close(
            base_local["cached_kv_unoverlapped_cycles"],
            float(base_local["cached_kv_cycles"]),
        )
        assert_close(
            base_local["cached_kv_unoverlapped_cycles"],
            float(base_local["cached_kv_critical_lut_cycles"])
            + float(base_local["cached_kv_critical_scale_cycles"]),
        )

        overlap_completed = run_simulator(
            binary,
            config,
            ("near_bank_pe.cached_kv_lut_scale_overlap=0.5",),
        )
        assert overlap_completed.returncode == 0, overlap_completed.stdout
        overlap_local = next(
            row
            for row in load_rows(directory / "per_query.csv")
            if row["baseline"] == "pim_selective_kv_local_combine"
        )
        assert_close(overlap_local["cached_kv_lut_scale_overlap"], 0.5)
        assert_close(
            overlap_local["cached_kv_unoverlapped_cycles"],
            float(base_local["cached_kv_unoverlapped_cycles"]),
        )
        expected_hidden = 0.5 * min(
            float(overlap_local["cached_kv_critical_lut_cycles"]),
            float(overlap_local["cached_kv_critical_scale_cycles"]),
        )
        assert_close(
            overlap_local["cached_kv_overlap_hidden_cycles"], expected_hidden
        )
        assert_close(
            overlap_local["cached_kv_cycles"],
            float(overlap_local["cached_kv_unoverlapped_cycles"])
            - expected_hidden,
        )
        assert float(overlap_local["cached_kv_cycles"]) < float(
            base_local["cached_kv_cycles"]
        )

        slow_completed = run_simulator(
            binary,
            config,
            ("near_bank_pe.cached_kv_scale_group_cycles=2",),
        )
        assert slow_completed.returncode == 0, slow_completed.stdout
        slow_rows = load_rows(directory / "per_query.csv")
        slow_local = next(
            row
            for row in slow_rows
            if row["baseline"] == "pim_selective_kv_local_combine"
        )
        assert_close(
            slow_local["cached_kv_scale_cycles"],
            2 * float(base_local["cached_kv_scale_cycles"]),
        )
        assert_close(
            slow_local["gnn_score_scale_cycles"],
            float(base_local["gnn_score_scale_cycles"]),
        )
        assert_close(
            slow_local["gnn_value_scale_cycles"],
            float(base_local["gnn_value_scale_cycles"]),
        )

        invalid_completed = run_simulator(
            binary, config, ("near_bank_pe.not_a_parameter=1",)
        )
        assert invalid_completed.returncode != 0
        assert "Unsupported analytical PIM override" in invalid_completed.stdout
        zero_completed = run_simulator(
            binary, config, ("near_bank_pe.cached_kv_scale_group_cycles=0",)
        )
        assert zero_completed.returncode != 0
        assert "positive finite number" in zero_completed.stdout
        invalid_overlap_completed = run_simulator(
            binary, config, ("near_bank_pe.cached_kv_lut_scale_overlap=1.1",)
        )
        assert invalid_overlap_completed.returncode != 0
        assert "within [0,1]" in invalid_overlap_completed.stdout

        sensitivity_directory = directory / "sensitivity"
        sensitivity_completed = subprocess.run(
            [
                sys.executable,
                str(PATCH6_RUNNER),
                "--binary",
                str(binary),
                "--config",
                str(config),
                "--output-directory",
                str(sensitivity_directory),
                "--expected-workloads",
                "fixture_task",
                "--expected-queries-per-workload",
                "1",
            ],
            cwd=REPOSITORY_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        assert sensitivity_completed.returncode == 0, sensitivity_completed.stdout
        assert "No test rows were simulated or reported" in sensitivity_completed.stdout
        overall = load_rows(
            sensitivity_directory / "patch6_sensitivity_overall.csv"
        )
        by_scenario = {row["scenario"]: row for row in overall}
        assert len(overall) == 17
        assert float(by_scenario["overlap_half"]["mean_latency_ns"]) < float(
            by_scenario["base"]["mean_latency_ns"]
        )
        assert float(by_scenario["overlap_full"]["mean_latency_ns"]) < float(
            by_scenario["overlap_half"]["mean_latency_ns"]
        )
        assert float(by_scenario["cached_scale_fast"]["mean_latency_ns"]) < float(
            by_scenario["base"]["mean_latency_ns"]
        )
        assert float(by_scenario["cached_scale_slow"]["mean_latency_ns"]) > float(
            by_scenario["base"]["mean_latency_ns"]
        )
        assert float(by_scenario["optimistic_joint"]["mean_latency_ns"]) < float(
            by_scenario["base"]["mean_latency_ns"]
        )
        assert float(by_scenario["conservative_joint"]["mean_latency_ns"]) > float(
            by_scenario["base"]["mean_latency_ns"]
        )
        assert len(
            load_rows(sensitivity_directory / "patch6_sensitivity_by_workload.csv")
        ) == 17
        assert len(
            load_rows(sensitivity_directory / "patch6_parameter_manifest.csv")
        ) == 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"Missing analytical_pim binary: {binary}")
    check_valid_run(binary)
    check_negative_cases(binary)
    check_patch6_run(binary)
    print("Patch 5/6 analytical tests passed: valid=7, negative=10")


if __name__ == "__main__":
    main()
