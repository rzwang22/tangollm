#!/usr/bin/env python3
"""End-to-end and negative tests for the Patch 5 GOFA trace path."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures/gofa_trace_v1"
CONFIG_TEMPLATE = REPOSITORY_ROOT / "tests/analytical_pim_trace_fixture_config.yaml"
TASK_DIRECTORY = "fixture_task_formal_v1"


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


def run_simulator(binary: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), str(config)],
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"Missing analytical_pim binary: {binary}")
    check_valid_run(binary)
    check_negative_cases(binary)
    print("Patch 5 trace tests passed: valid=1, negative=6")


if __name__ == "__main__":
    main()
