#!/usr/bin/env python3
"""Validate and summarize the frozen Patch 5 GOFA test configuration."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


FROZEN_GRAPH_PLACEMENT = "source_dst_locality"
FROZEN_KV_PLACEMENT = "balanced"
FROZEN_MEMORY_TOKEN_TILE = "4"
NO_LOCAL_BASELINE = "pim_selective_kv_no_local_combine"
LOCAL_BASELINE = "pim_selective_kv_local_combine"
BASELINES = (NO_LOCAL_BASELINE, LOCAL_BASELINE)
SPLITS = ("validation", "test")
DEFAULT_WORKLOADS = (
    "cora_node",
    "cora_link",
    "pubmed_node",
    "wikics",
    "arxiv",
)

PER_QUERY_COLUMNS = {
    "workload",
    "workload_mode",
    "split",
    "evaluation_role",
    "trace_order",
    "trace_query_id",
    "graph_compute_placement",
    "kv_storage_placement",
    "baseline",
    "memory_token_tile",
    "selected_kv_count",
    "runtime_loaded_cache_bytes",
    "local_combine_reduction_ratio",
    "local_combine_buffer_max_bytes",
    "local_buffer_capacity_utilization",
    "local_buffer_capacity_exceeded",
    "local_vadd_input_edge_groups",
    "local_vadd_initialization_groups",
    "local_vadd_operation_groups",
    "local_vadd_cycles",
    "reducer_cycles",
    "critical_path_latency_ns",
    "traffic_stall_fraction",
    "bank_to_pc_total_bytes",
    "bottleneck_stage",
}

AGGREGATE_COLUMNS = {
    "workload",
    "workload_mode",
    "split",
    "evaluation_role",
    "graph_compute_placement",
    "kv_storage_placement",
    "baseline",
    "memory_token_tile",
    "num_queries",
    "mean_critical_path_latency_ns",
    "p50_latency_ns",
    "p95_latency_ns",
}


class AnalysisError(RuntimeError):
    pass


def read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise AnalysisError(f"missing CSV: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(required_columns - columns)
        if missing:
            raise AnalysisError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise AnalysisError(f"{path}: CSV contains no data rows")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise AnalysisError(f"{path}: malformed or misaligned CSV row")
    return rows


def number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise AnalysisError(f"invalid numeric field {field}={row.get(field)!r}") from error
    if not math.isfinite(value):
        raise AnalysisError(f"non-finite numeric field {field}={row[field]!r}")
    return value


def mean(rows: Sequence[dict[str, str]], field: str) -> float:
    return statistics.fmean(number(row, field) for row in rows)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise AnalysisError("cannot compute a percentile of an empty group")
    rank = fraction * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def reduction(before: float, after: float, label: str) -> float:
    if before <= 0.0:
        raise AnalysisError(f"{label}: non-positive reference value {before}")
    return 1.0 - after / before


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-3)


def matches_frozen_configuration(row: dict[str, str]) -> bool:
    return (
        row["graph_compute_placement"] == FROZEN_GRAPH_PLACEMENT
        and row["kv_storage_placement"] == FROZEN_KV_PLACEMENT
        and row["memory_token_tile"] == FROZEN_MEMORY_TOKEN_TILE
        and row["baseline"] in BASELINES
    )


def select_group(
    rows: Sequence[dict[str, str]], workload: str, split: str, baseline: str
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if matches_frozen_configuration(row)
        and row["workload"] == workload
        and row["split"] == split
        and row["baseline"] == baseline
    ]


def validate_per_query(
    rows: Sequence[dict[str, str]],
    workloads: Sequence[str],
    expected_queries_per_split: int,
    require_final_only: bool,
) -> None:
    if require_final_only and any(not matches_frozen_configuration(row) for row in rows):
        raise AnalysisError("per-query CSV contains a non-frozen configuration")
    expected_roles = {"validation": "config_selection", "test": "final_evaluation"}
    expected_keys: set[tuple[str, str, str, str]] = set()
    for workload in workloads:
        for split in SPLITS:
            groups = {
                baseline: select_group(rows, workload, split, baseline)
                for baseline in BASELINES
            }
            for baseline, group in groups.items():
                if len(group) != expected_queries_per_split:
                    raise AnalysisError(
                        f"{workload}/{split}/{baseline}: expected "
                        f"{expected_queries_per_split} rows, found {len(group)}"
                    )
                if any(row["workload_mode"] != "trace" for row in group):
                    raise AnalysisError(f"{workload}/{split}: workload_mode is not trace")
                if any(row["evaluation_role"] != expected_roles[split] for row in group):
                    raise AnalysisError(f"{workload}/{split}: evaluation_role mismatch")
                trace_orders = [int(row["trace_order"]) for row in group]
                if trace_orders != sorted(trace_orders) or len(set(trace_orders)) != len(group):
                    raise AnalysisError(f"{workload}/{split}/{baseline}: trace order mismatch")
                keys = {
                    (row["workload"], row["split"], row["trace_order"], row["trace_query_id"])
                    for row in group
                }
                if baseline == NO_LOCAL_BASELINE:
                    expected_keys = keys
                elif keys != expected_keys:
                    raise AnalysisError(f"{workload}/{split}: baseline query sets differ")

            for row in groups[LOCAL_BASELINE]:
                input_groups = number(row, "local_vadd_input_edge_groups")
                initialization_groups = number(row, "local_vadd_initialization_groups")
                operation_groups = number(row, "local_vadd_operation_groups")
                if not math.isclose(
                    input_groups - initialization_groups,
                    operation_groups,
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise AnalysisError(
                        f"{workload}/{split}/{row['trace_query_id']}: VADD count identity failed"
                    )
                if input_groups > 0.0 and not math.isclose(
                    operation_groups / input_groups,
                    number(row, "local_combine_reduction_ratio"),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise AnalysisError(
                        f"{workload}/{split}/{row['trace_query_id']}: VADD/reduction ratio failed"
                    )
                if number(row, "local_buffer_capacity_exceeded") != 0.0:
                    raise AnalysisError(
                        f"{workload}/{split}/{row['trace_query_id']}: local buffer overflow"
                    )

            for row in groups[NO_LOCAL_BASELINE]:
                no_local_vadd = sum(
                    number(row, field)
                    for field in (
                        "local_vadd_input_edge_groups",
                        "local_vadd_initialization_groups",
                        "local_vadd_operation_groups",
                        "local_vadd_cycles",
                    )
                )
                if no_local_vadd != 0.0:
                    raise AnalysisError(
                        f"{workload}/{split}/{row['trace_query_id']}: no-local VADD is nonzero"
                    )

    if require_final_only:
        expected_rows = len(workloads) * len(SPLITS) * len(BASELINES) * expected_queries_per_split
        if len(rows) != expected_rows:
            raise AnalysisError(
                f"final per-query row count mismatch: expected {expected_rows}, found {len(rows)}"
            )
    local_rows = [
        row
        for row in rows
        if matches_frozen_configuration(row)
        and row["baseline"] == LOCAL_BASELINE
        and row["workload"] in workloads
    ]
    trace_orders = [int(row["trace_order"]) for row in local_rows]
    expected_query_count = len(workloads) * len(SPLITS) * expected_queries_per_split
    if trace_orders != list(range(expected_query_count)):
        raise AnalysisError("frozen rows do not preserve the complete global trace order")


def validate_aggregate(
    aggregate_rows: Sequence[dict[str, str]],
    per_query_rows: Sequence[dict[str, str]],
    workloads: Sequence[str],
    expected_queries_per_split: int,
    require_final_only: bool,
) -> None:
    if require_final_only and any(
        not matches_frozen_configuration(row) for row in aggregate_rows
    ):
        raise AnalysisError("aggregate CSV contains a non-frozen configuration")
    for workload in workloads:
        for split in SPLITS:
            for baseline in BASELINES:
                aggregate = select_group(aggregate_rows, workload, split, baseline)
                if len(aggregate) != 1:
                    raise AnalysisError(
                        f"{workload}/{split}/{baseline}: expected one aggregate row, "
                        f"found {len(aggregate)}"
                    )
                queries = select_group(per_query_rows, workload, split, baseline)
                row = aggregate[0]
                expected_role = (
                    "config_selection" if split == "validation" else "final_evaluation"
                )
                if row["workload_mode"] != "trace" or row["evaluation_role"] != expected_role:
                    raise AnalysisError(
                        f"{workload}/{split}/{baseline}: aggregate role mismatch"
                    )
                if int(number(row, "num_queries")) != expected_queries_per_split:
                    raise AnalysisError(f"{workload}/{split}/{baseline}: aggregate count mismatch")
                latencies = [number(query, "critical_path_latency_ns") for query in queries]
                checks = {
                    "mean_critical_path_latency_ns": statistics.fmean(latencies),
                    "p50_latency_ns": percentile(latencies, 0.50),
                    "p95_latency_ns": percentile(latencies, 0.95),
                }
                for field, expected in checks.items():
                    if not close(number(row, field), expected):
                        raise AnalysisError(
                            f"{workload}/{split}/{baseline}: {field} does not match per-query CSV"
                        )
    if require_final_only:
        expected_rows = len(workloads) * len(SPLITS) * len(BASELINES)
        if len(aggregate_rows) != expected_rows:
            raise AnalysisError(
                f"final aggregate row count mismatch: expected {expected_rows}, "
                f"found {len(aggregate_rows)}"
            )


def build_summary(
    workload: str,
    validation_local: Sequence[dict[str, str]],
    test_local: Sequence[dict[str, str]],
    test_no_local: Sequence[dict[str, str]],
) -> dict[str, str | int | float]:
    validation_latency = [number(row, "critical_path_latency_ns") for row in validation_local]
    test_latency = [number(row, "critical_path_latency_ns") for row in test_local]
    validation_mean = statistics.fmean(validation_latency)
    test_mean = statistics.fmean(test_latency)
    no_local_mean = mean(test_no_local, "critical_path_latency_ns")
    dominant, dominant_count = Counter(row["bottleneck_stage"] for row in test_local).most_common(1)[0]
    return {
        "workload": workload,
        "query_count": len(test_local),
        "validation_local_mean_ms": validation_mean / 1e6,
        "validation_local_p95_ms": percentile(validation_latency, 0.95) / 1e6,
        "test_local_mean_ms": test_mean / 1e6,
        "test_local_p50_ms": percentile(test_latency, 0.50) / 1e6,
        "test_local_p95_ms": percentile(test_latency, 0.95) / 1e6,
        "test_vs_validation_mean_change": test_mean / validation_mean - 1.0,
        "test_no_local_mean_ms": no_local_mean / 1e6,
        "test_local_latency_reduction": reduction(no_local_mean, test_mean, "latency"),
        "test_bank_to_pc_traffic_reduction": reduction(
            mean(test_no_local, "bank_to_pc_total_bytes"),
            mean(test_local, "bank_to_pc_total_bytes"),
            "bank-to-PC traffic",
        ),
        "test_reducer_cycle_reduction": reduction(
            mean(test_no_local, "reducer_cycles"),
            mean(test_local, "reducer_cycles"),
            "reducer cycles",
        ),
        "test_mean_runtime_loaded_mib": mean(test_local, "runtime_loaded_cache_bytes") / 1024**2,
        "test_mean_selected_kv_count": mean(test_local, "selected_kv_count"),
        "test_mean_local_combine_reduction_ratio": mean(
            test_local, "local_combine_reduction_ratio"
        ),
        "test_mean_traffic_stall_fraction": mean(test_local, "traffic_stall_fraction"),
        "test_max_local_buffer_kib": max(
            number(row, "local_combine_buffer_max_bytes") for row in test_local
        )
        / 1024,
        "test_max_buffer_capacity_utilization": max(
            number(row, "local_buffer_capacity_utilization") for row in test_local
        ),
        "test_dominant_bottleneck_stage": dominant,
        "test_bottleneck_query_fraction": dominant_count / len(test_local),
    }


def summarize_group(
    per_query_rows: Sequence[dict[str, str]], workload: str
) -> dict[str, str | int | float]:
    return build_summary(
        workload,
        select_group(per_query_rows, workload, "validation", LOCAL_BASELINE),
        select_group(per_query_rows, workload, "test", LOCAL_BASELINE),
        select_group(per_query_rows, workload, "test", NO_LOCAL_BASELINE),
    )


def summarize_all(
    per_query_rows: Sequence[dict[str, str]], workloads: Sequence[str]
) -> dict[str, str | int | float]:
    workload_set = set(workloads)

    def combined(split: str, baseline: str) -> list[dict[str, str]]:
        return [
            row
            for row in per_query_rows
            if matches_frozen_configuration(row)
            and row["workload"] in workload_set
            and row["split"] == split
            and row["baseline"] == baseline
        ]

    return build_summary(
        "__all__",
        combined("validation", LOCAL_BASELINE),
        combined("test", LOCAL_BASELINE),
        combined("test", NO_LOCAL_BASELINE),
    )


def write_summary(path: Path, summaries: Sequence[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def print_summary(summaries: Sequence[dict[str, str | int | float]]) -> None:
    print("Frozen configuration")
    print(
        f"graph={FROZEN_GRAPH_PLACEMENT}, kv={FROZEN_KV_PLACEMENT}, "
        f"T={FROZEN_MEMORY_TOKEN_TILE}, design={LOCAL_BASELINE}"
    )
    print("\nTest split PIM-internal characterization")
    print(
        "workload queries local_mean_ms local_p50_ms local_p95_ms "
        "local_latency_reduction bank_to_pc_reduction reducer_reduction "
        "buffer_max_KiB stall bottleneck"
    )
    for row in summaries:
        print(
            f"{row['workload']:<12} {row['query_count']:>7} "
            f"{row['test_local_mean_ms']:>13.3f} "
            f"{row['test_local_p50_ms']:>12.3f} "
            f"{row['test_local_p95_ms']:>12.3f} "
            f"{row['test_local_latency_reduction']:>23.4%} "
            f"{row['test_bank_to_pc_traffic_reduction']:>20.2%} "
            f"{row['test_reducer_cycle_reduction']:>17.2%} "
            f"{row['test_max_local_buffer_kib']:>14.3f} "
            f"{row['test_mean_traffic_stall_fraction']:>6.2%} "
            f"{row['test_dominant_bottleneck_stage']}"
        )
    print("\nValidation-to-test mean-latency shift")
    for row in summaries:
        print(
            f"{row['workload']:<12} validation={row['validation_local_mean_ms']:.3f} ms "
            f"test={row['test_local_mean_ms']:.3f} ms "
            f"change={row['test_vs_validation_mean_change']:+.2%}"
        )


def parse_workloads(value: str) -> tuple[str, ...]:
    workloads = tuple(item.strip() for item in value.split(",") if item.strip())
    if not workloads:
        raise argparse.ArgumentTypeError("at least one workload is required")
    if len(set(workloads)) != len(workloads):
        raise argparse.ArgumentTypeError("workload names must be unique")
    return workloads


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize the frozen GOFA trace test result."
    )
    parser.add_argument("--per-query", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-workloads",
        type=parse_workloads,
        default=DEFAULT_WORKLOADS,
        help="comma-separated workload names",
    )
    parser.add_argument("--expected-queries-per-split", type=int, default=100)
    parser.add_argument(
        "--require-final-only",
        action="store_true",
        help="reject CSV rows outside the frozen final configuration",
    )
    args = parser.parse_args()
    if args.expected_queries_per_split <= 0:
        raise SystemExit("--expected-queries-per-split must be positive")

    try:
        per_query_rows = read_csv(args.per_query, PER_QUERY_COLUMNS)
        aggregate_rows = read_csv(args.aggregate, AGGREGATE_COLUMNS)
        validate_per_query(
            per_query_rows,
            args.expected_workloads,
            args.expected_queries_per_split,
            args.require_final_only,
        )
        validate_aggregate(
            aggregate_rows,
            per_query_rows,
            args.expected_workloads,
            args.expected_queries_per_split,
            args.require_final_only,
        )
        summaries = [
            summarize_group(per_query_rows, workload)
            for workload in args.expected_workloads
        ]
        summaries.append(summarize_all(per_query_rows, args.expected_workloads))
        write_summary(args.output, summaries)
    except AnalysisError as error:
        raise SystemExit(f"ERROR: {error}") from error

    expected_queries = len(args.expected_workloads) * len(SPLITS) * args.expected_queries_per_split
    print(
        f"Final trace checks passed: queries={expected_queries}, "
        f"per_query_rows={len(per_query_rows)}, aggregate_rows={len(aggregate_rows)}"
    )
    print_summary(summaries)
    print(f"\nsummary_csv={args.output}")


if __name__ == "__main__":
    main()
