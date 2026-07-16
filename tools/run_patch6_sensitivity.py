#!/usr/bin/env python3
"""Run validation-only Patch 6 calibration-envelope sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


GRAPH_PLACEMENT = "source_dst_locality"
KV_PLACEMENT = "balanced"
MEMORY_TOKEN_TILE = "4"
NO_LOCAL = "pim_selective_kv_no_local_combine"
LOCAL = "pim_selective_kv_local_combine"
DEFAULT_WORKLOADS = (
    "cora_node",
    "cora_link",
    "pubmed_node",
    "wikics",
    "arxiv",
)

PARAMETERS = {
    "near_bank_pe.q8k8_group_cycles": ("q8k8_cycles", 1.0, "cycles/group"),
    "near_bank_pe.p8v8_group_cycles": ("p8v8_cycles", 1.0, "cycles/group"),
    "near_bank_pe.q8k2_lut_group_cycles": (
        "q8k2_lut_cycles",
        1.0,
        "cycles/group",
    ),
    "near_bank_pe.p8v2_lut_group_cycles": (
        "p8v2_lut_cycles",
        1.0,
        "cycles/group",
    ),
    "near_bank_pe.vadd_group_cycles": ("vadd_cycles", 1.0, "cycles/group"),
    "near_bank_pe.gnn_scale_group_cycles": (
        "gnn_scale_cycles",
        1.0,
        "cycles/group",
    ),
    "near_bank_pe.cached_kv_scale_group_cycles": (
        "cached_kv_scale_cycles",
        1.0,
        "cycles/group",
    ),
    "near_bank_pe.cached_kv_lut_scale_overlap": (
        "cached_kv_lut_scale_overlap",
        0.0,
        "fraction",
    ),
    "communication.q_broadcast_bandwidth_bytes_per_cycle_per_bank": (
        "q_broadcast_bw",
        64.0,
        "bytes/cycle/bank",
    ),
    "communication.bank_to_pc_bandwidth_bytes_per_cycle_per_bank": (
        "bank_to_pc_bw",
        32.0,
        "bytes/cycle/bank",
    ),
    "communication.pc_to_global_bandwidth_bytes_per_cycle_per_pc": (
        "pc_to_global_bw",
        64.0,
        "bytes/cycle/pc",
    ),
    "communication.global_to_npu_bandwidth_bytes_per_cycle": (
        "global_to_npu_bw",
        256.0,
        "bytes/cycle",
    ),
    "reducer.pc_throughput_groups_per_cycle_per_lane": (
        "pc_reducer_throughput",
        1.0,
        "groups/cycle/lane",
    ),
    "reducer.pc_input_bandwidth_bytes_per_cycle": (
        "pc_reducer_input_bw",
        512.0,
        "bytes/cycle",
    ),
    "reducer.global_throughput_groups_per_cycle_per_lane": (
        "global_reducer_throughput",
        1.0,
        "groups/cycle/lane",
    ),
    "reducer.global_input_bandwidth_bytes_per_cycle": (
        "global_reducer_input_bw",
        1024.0,
        "bytes/cycle",
    ),
}

CYCLE_PARAMETERS = {
    key for key, (_, _, unit) in PARAMETERS.items() if unit == "cycles/group"
}
THROUGHPUT_PARAMETERS = {
    key for key, (_, _, unit) in PARAMETERS.items() if unit == "groups/cycle/lane"
}

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
    "local_combine_reduction_ratio",
    "local_buffer_capacity_exceeded",
    "local_combine_buffer_max_bytes",
    "local_vadd_input_edge_groups",
    "local_vadd_initialization_groups",
    "local_vadd_operation_groups",
    "local_vadd_cycles",
    "compute_cycles",
    "total_cycles",
    "communication_cycles",
    "reducer_cycles",
    "critical_path_latency_ns",
    "traffic_stall_fraction",
    "bank_to_pc_total_bytes",
    "cached_kv_cycles",
    "cached_kv_scale_cycles",
    "cached_kv_lut_scale_overlap",
    "cached_kv_critical_lut_cycles",
    "cached_kv_critical_scale_cycles",
    "cached_kv_unoverlapped_cycles",
    "cached_kv_overlap_hidden_cycles",
    "cached_kv_raw_lut_fraction",
    "cached_kv_raw_scale_fraction",
    "cached_kv_overlap_hidden_fraction",
    "cached_kv_pipeline_compute_fraction",
    "cached_kv_pipeline_critical_path_fraction",
    "q8k2_lut_cycles",
    "p8v2_lut_cycles",
    "q8k8_vdot_cycles",
    "p8v8_vmul_cycles",
    "bottleneck_stage",
    "bottleneck_fraction",
}

AGGREGATE_COLUMNS = {
    "workload",
    "split",
    "graph_compute_placement",
    "kv_storage_placement",
    "baseline",
    "memory_token_tile",
    "num_queries",
}


class SensitivityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Scenario:
    name: str
    category: str
    description: str
    values: dict[str, float]

    @property
    def overrides(self) -> dict[str, float]:
        return dict(self.values)


def base_values() -> dict[str, float]:
    return {key: base for key, (_, base, _) in PARAMETERS.items()}


def scaled_values(groups: Sequence[set[str]], factors: Sequence[float]) -> dict[str, float]:
    values = base_values()
    for group, factor in zip(groups, factors):
        for key in group:
            values[key] *= factor
    return values


GNN_PIPELINE_PARAMETERS = {
    "near_bank_pe.q8k8_group_cycles",
    "near_bank_pe.p8v8_group_cycles",
    "near_bank_pe.gnn_scale_group_cycles",
}
INT2_PARAMETERS = {
    "near_bank_pe.q8k2_lut_group_cycles",
    "near_bank_pe.p8v2_lut_group_cycles",
}
VADD_PARAMETERS = {"near_bank_pe.vadd_group_cycles"}
CACHED_SCALE_PARAMETERS = {"near_bank_pe.cached_kv_scale_group_cycles"}
COMMUNICATION_PARAMETERS = {
    key for key in PARAMETERS if key.startswith("communication.")
}
REDUCER_PARAMETERS = THROUGHPUT_PARAMETERS | {
    "reducer.pc_input_bandwidth_bytes_per_cycle",
    "reducer.global_input_bandwidth_bytes_per_cycle",
}


def make_scenarios() -> tuple[Scenario, ...]:
    scenarios = [
        Scenario("base", "reference", "Patch 5 base assumptions", base_values())
    ]
    for name, overlap in (("overlap_half", 0.5), ("overlap_full", 1.0)):
        values = base_values()
        values["near_bank_pe.cached_kv_lut_scale_overlap"] = overlap
        scenarios.append(
            Scenario(
                name,
                "execution_overlap",
                f"cached-KV LUT/scale overlap={overlap:g}",
                values,
            )
        )
    factors = (("fast", 0.5), ("slow", 2.0))
    cycle_groups = (
        ("cached_scale", CACHED_SCALE_PARAMETERS),
        ("int2_lut", INT2_PARAMETERS),
        ("gnn_pipeline", GNN_PIPELINE_PARAMETERS),
        ("vadd", VADD_PARAMETERS),
    )
    for prefix, group in cycle_groups:
        for label, factor in factors:
            scenarios.append(
                Scenario(
                    f"{prefix}_{label}",
                    "single_factor",
                    f"{prefix} cycle cost x{factor:g}",
                    scaled_values((group,), (factor,)),
                )
            )
    for label, factor in (("slow", 0.5), ("fast", 2.0)):
        scenarios.append(
            Scenario(
                f"communication_{label}",
                "single_factor",
                f"all modeled communication bandwidth x{factor:g}",
                scaled_values((COMMUNICATION_PARAMETERS,), (factor,)),
            )
        )
        scenarios.append(
            Scenario(
                f"reducer_{label}",
                "single_factor",
                f"all modeled reducer throughput/bandwidth x{factor:g}",
                scaled_values((REDUCER_PARAMETERS,), (factor,)),
            )
        )
    all_cycle_groups = (
        CACHED_SCALE_PARAMETERS | INT2_PARAMETERS | GNN_PIPELINE_PARAMETERS | VADD_PARAMETERS
    )
    optimistic = scaled_values(
        (all_cycle_groups, COMMUNICATION_PARAMETERS, REDUCER_PARAMETERS),
        (0.5, 2.0, 2.0),
    )
    optimistic["near_bank_pe.cached_kv_lut_scale_overlap"] = 1.0
    conservative = scaled_values(
        (all_cycle_groups, COMMUNICATION_PARAMETERS, REDUCER_PARAMETERS),
        (2.0, 0.5, 0.5),
    )
    scenarios.extend(
        (
            Scenario(
                "optimistic_joint",
                "joint_envelope",
                "cycle costs x0.5, full LUT/scale overlap, and bandwidth/throughput x2",
                optimistic,
            ),
            Scenario(
                "conservative_joint",
                "joint_envelope",
                "cycle costs x2, no LUT/scale overlap, and bandwidth/throughput x0.5",
                conservative,
            ),
        )
    )
    return tuple(scenarios)


SCENARIOS = make_scenarios()


def parse_workloads(value: str) -> tuple[str, ...]:
    workloads = tuple(item.strip() for item in value.split(",") if item.strip())
    if not workloads or len(set(workloads)) != len(workloads):
        raise argparse.ArgumentTypeError("workloads must be a non-empty unique list")
    return workloads


def parse_scenarios(value: str) -> tuple[str, ...]:
    if value == "all":
        return tuple(scenario.name for scenario in SCENARIOS)
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    known = {scenario.name for scenario in SCENARIOS}
    unknown = sorted(set(names) - known)
    if not names or unknown:
        raise argparse.ArgumentTypeError(
            "unknown or empty scenario list: " + ",".join(unknown)
        )
    if "base" not in names:
        raise argparse.ArgumentTypeError("scenario list must include base")
    return names


def read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise SensitivityError(f"missing simulator output: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required_columns - set(reader.fieldnames or ()))
        if missing:
            raise SensitivityError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise SensitivityError(f"{path}: empty or malformed CSV")
    return rows


def number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise SensitivityError(f"invalid {field}={row.get(field)!r}") from error
    if not math.isfinite(value):
        raise SensitivityError(f"non-finite {field}={row[field]!r}")
    return value


def mean(rows: Sequence[dict[str, str]], field: str) -> float:
    return statistics.fmean(number(row, field) for row in rows)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def reduction(before: float, after: float) -> float:
    if before <= 0.0:
        raise SensitivityError(f"non-positive reduction reference: {before}")
    return 1.0 - after / before


def is_frozen_row(row: dict[str, str]) -> bool:
    return (
        row["split"] == "validation"
        and row["graph_compute_placement"] == GRAPH_PLACEMENT
        and row["kv_storage_placement"] == KV_PLACEMENT
        and row["memory_token_tile"] == MEMORY_TOKEN_TILE
        and row["baseline"] in (NO_LOCAL, LOCAL)
    )


def select(
    rows: Sequence[dict[str, str]], workload: str | None, baseline: str
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if is_frozen_row(row)
        and row["baseline"] == baseline
        and (workload is None or row["workload"] == workload)
    ]


def validate_outputs(
    per_query: Sequence[dict[str, str]],
    aggregate: Sequence[dict[str, str]],
    workloads: Sequence[str],
    queries_per_workload: int,
    scenario: str,
    expected_overlap: float,
) -> None:
    expected_per_query = len(workloads) * queries_per_workload * 2
    expected_aggregate = len(workloads) * 2
    if len(per_query) != expected_per_query or any(
        not is_frozen_row(row) for row in per_query
    ):
        raise SensitivityError(
            f"{scenario}: expected {expected_per_query} validation-only rows, "
            f"found {len(per_query)}"
        )
    if len(aggregate) != expected_aggregate or any(
        not is_frozen_row(row) for row in aggregate
    ):
        raise SensitivityError(
            f"{scenario}: expected {expected_aggregate} validation aggregate rows, "
            f"found {len(aggregate)}"
        )
    if any(int(number(row, "num_queries")) != queries_per_workload for row in aggregate):
        raise SensitivityError(f"{scenario}: aggregate query count mismatch")
    if any(
        row["workload_mode"] != "trace"
        or row["evaluation_role"] != "config_selection"
        for row in per_query
    ):
        raise SensitivityError(f"{scenario}: test or non-trace row leaked into output")

    for row in per_query:
        overlap = number(row, "cached_kv_lut_scale_overlap")
        critical_lut = number(row, "cached_kv_critical_lut_cycles")
        critical_scale = number(row, "cached_kv_critical_scale_cycles")
        unoverlapped = number(row, "cached_kv_unoverlapped_cycles")
        hidden = number(row, "cached_kv_overlap_hidden_cycles")
        effective = number(row, "cached_kv_cycles")
        raw_lut_fraction = number(row, "cached_kv_raw_lut_fraction")
        raw_scale_fraction = number(row, "cached_kv_raw_scale_fraction")
        hidden_fraction = number(row, "cached_kv_overlap_hidden_fraction")
        compute_fraction = number(row, "cached_kv_pipeline_compute_fraction")
        critical_path_fraction = number(
            row, "cached_kv_pipeline_critical_path_fraction"
        )
        compute = number(row, "compute_cycles")
        total = number(row, "total_cycles")
        if compute <= 0.0 or total <= 0.0:
            raise SensitivityError(f"{scenario}: non-positive PIM cycle total")
        if not math.isclose(overlap, expected_overlap, rel_tol=0.0, abs_tol=1e-9):
            raise SensitivityError(f"{scenario}: cached-KV overlap override mismatch")
        if not math.isclose(
            unoverlapped, critical_lut + critical_scale, rel_tol=0.0, abs_tol=1e-6
        ):
            raise SensitivityError(f"{scenario}: cached-KV raw-cycle identity failed")
        if not math.isclose(
            hidden,
            overlap * min(critical_lut, critical_scale),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise SensitivityError(f"{scenario}: cached-KV hidden-cycle identity failed")
        if not math.isclose(
            effective, unoverlapped - hidden, rel_tol=0.0, abs_tol=1e-6
        ):
            raise SensitivityError(f"{scenario}: cached-KV effective-cycle identity failed")
        if unoverlapped > 0.0:
            fraction_checks = (
                (raw_lut_fraction, critical_lut / unoverlapped, "raw LUT"),
                (raw_scale_fraction, critical_scale / unoverlapped, "raw scale"),
                (hidden_fraction, hidden / unoverlapped, "hidden"),
            )
            for actual, expected, label in fraction_checks:
                if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
                    raise SensitivityError(
                        f"{scenario}: cached-KV {label} fraction identity failed"
                    )
            if not math.isclose(
                raw_lut_fraction + raw_scale_fraction,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise SensitivityError(
                    f"{scenario}: cached-KV raw fractions do not sum to one"
                )
        if not math.isclose(
            compute_fraction,
            effective / compute,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise SensitivityError(f"{scenario}: cached-KV compute fraction identity failed")
        if not math.isclose(
            critical_path_fraction,
            effective / total,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise SensitivityError(
                f"{scenario}: cached-KV critical-path fraction identity failed"
            )

    for workload in workloads:
        baseline_keys: set[tuple[str, str]] | None = None
        for baseline in (NO_LOCAL, LOCAL):
            group = select(per_query, workload, baseline)
            if len(group) != queries_per_workload:
                raise SensitivityError(
                    f"{scenario}/{workload}/{baseline}: row count mismatch"
                )
            trace_orders = [int(row["trace_order"]) for row in group]
            if trace_orders != sorted(trace_orders) or len(set(trace_orders)) != len(group):
                raise SensitivityError(
                    f"{scenario}/{workload}/{baseline}: trace order mismatch"
                )
            keys = {(row["trace_order"], row["trace_query_id"]) for row in group}
            if baseline_keys is None:
                baseline_keys = keys
            elif keys != baseline_keys:
                raise SensitivityError(
                    f"{scenario}/{workload}: baseline query sets differ"
                )

        for row in select(per_query, workload, LOCAL):
            input_groups = number(row, "local_vadd_input_edge_groups")
            initialization_groups = number(row, "local_vadd_initialization_groups")
            operation_groups = number(row, "local_vadd_operation_groups")
            if not math.isclose(
                input_groups - initialization_groups,
                operation_groups,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise SensitivityError(f"{scenario}: VADD identity failed")
            if number(row, "local_buffer_capacity_exceeded") != 0.0:
                raise SensitivityError(f"{scenario}: local buffer overflow")
        for row in select(per_query, workload, NO_LOCAL):
            if any(
                number(row, field) != 0.0
                for field in (
                    "local_vadd_input_edge_groups",
                    "local_vadd_initialization_groups",
                    "local_vadd_operation_groups",
                    "local_vadd_cycles",
                )
            ):
                raise SensitivityError(f"{scenario}: no-local VADD is nonzero")


def metric_row(
    scenario: Scenario,
    workload: str,
    local: Sequence[dict[str, str]],
    no_local: Sequence[dict[str, str]],
) -> dict[str, str | int | float]:
    latencies = [number(row, "critical_path_latency_ns") for row in local]
    no_local_by_query = {
        (row["trace_order"], row["trace_query_id"]): row for row in no_local
    }
    faster_queries = sum(
        number(row, "critical_path_latency_ns")
        < number(
            no_local_by_query[(row["trace_order"], row["trace_query_id"])],
            "critical_path_latency_ns",
        )
        for row in local
    )
    dominant, dominant_count = Counter(
        row["bottleneck_stage"] for row in local
    ).most_common(1)[0]
    mean_cached_kv_unoverlapped = mean(local, "cached_kv_unoverlapped_cycles")
    mean_cached_kv_hidden = mean(local, "cached_kv_overlap_hidden_cycles")
    result: dict[str, str | int | float] = {
        "scenario": scenario.name,
        "category": scenario.category,
        "description": scenario.description,
        "workload": workload,
        "num_queries": len(local),
        "mean_latency_ns": statistics.fmean(latencies),
        "p50_latency_ns": percentile(latencies, 0.50),
        "p95_latency_ns": percentile(latencies, 0.95),
        "latency_change_vs_base": 0.0,
        "mean_compute_cycles": mean(local, "compute_cycles"),
        "mean_communication_cycles": mean(local, "communication_cycles"),
        "mean_reducer_cycles": mean(local, "reducer_cycles"),
        "mean_cached_kv_cycles": mean(local, "cached_kv_cycles"),
        "mean_cached_kv_scale_cycles": mean(local, "cached_kv_scale_cycles"),
        "mean_cached_kv_critical_lut_cycles": mean(
            local, "cached_kv_critical_lut_cycles"
        ),
        "mean_cached_kv_critical_scale_cycles": mean(
            local, "cached_kv_critical_scale_cycles"
        ),
        "mean_cached_kv_unoverlapped_cycles": mean_cached_kv_unoverlapped,
        "mean_cached_kv_overlap_hidden_cycles": mean_cached_kv_hidden,
        "mean_cached_kv_raw_lut_fraction": mean(
            local, "cached_kv_raw_lut_fraction"
        ),
        "mean_cached_kv_raw_scale_fraction": mean(
            local, "cached_kv_raw_scale_fraction"
        ),
        "mean_cached_kv_overlap_hidden_fraction": mean(
            local, "cached_kv_overlap_hidden_fraction"
        ),
        "mean_cached_kv_pipeline_compute_fraction": mean(
            local, "cached_kv_pipeline_compute_fraction"
        ),
        "mean_cached_kv_pipeline_critical_path_fraction": mean(
            local, "cached_kv_pipeline_critical_path_fraction"
        ),
        "cached_kv_overlap_cycle_reduction": (
            mean_cached_kv_hidden / mean_cached_kv_unoverlapped
            if mean_cached_kv_unoverlapped > 0.0
            else 0.0
        ),
        "mean_q8k2_lut_cycles": mean(local, "q8k2_lut_cycles"),
        "mean_p8v2_lut_cycles": mean(local, "p8v2_lut_cycles"),
        "mean_q8k8_vdot_cycles": mean(local, "q8k8_vdot_cycles"),
        "mean_p8v8_vmul_cycles": mean(local, "p8v8_vmul_cycles"),
        "mean_local_vadd_cycles": mean(local, "local_vadd_cycles"),
        "mean_traffic_stall_fraction": mean(local, "traffic_stall_fraction"),
        "local_latency_reduction_vs_no_local": reduction(
            mean(no_local, "critical_path_latency_ns"),
            statistics.fmean(latencies),
        ),
        "local_faster_query_fraction": faster_queries / len(local),
        "bank_to_pc_traffic_reduction": reduction(
            mean(no_local, "bank_to_pc_total_bytes"),
            mean(local, "bank_to_pc_total_bytes"),
        ),
        "reducer_cycle_reduction": reduction(
            mean(no_local, "reducer_cycles"), mean(local, "reducer_cycles")
        ),
        "mean_local_combine_reduction_ratio": mean(
            local, "local_combine_reduction_ratio"
        ),
        "max_local_buffer_kib": max(
            number(row, "local_combine_buffer_max_bytes") for row in local
        )
        / 1024,
        "dominant_bottleneck_stage": dominant,
        "bottleneck_query_fraction": dominant_count / len(local),
        "mean_bottleneck_stage_fraction": mean(local, "bottleneck_fraction"),
    }
    for key, (short_name, _, _) in PARAMETERS.items():
        result[short_name] = scenario.values[key]
    return result


def write_csv(path: Path, rows: Sequence[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_parameter_manifest(path: Path) -> None:
    rows = []
    for key, (_, base, unit) in PARAMETERS.items():
        cycles = key in CYCLE_PARAMETERS
        if unit == "fraction":
            optimistic = 1.0
            conservative = 0.0
        else:
            optimistic = base * (0.5 if cycles else 2.0)
            conservative = base * (2.0 if cycles else 0.5)
        rows.append(
            {
                "parameter": key,
                "unit": unit,
                "optimistic_value": optimistic,
                "base_value": base,
                "conservative_value": conservative,
                "calibration_status": "uncalibrated_assumption",
                "source_note": "Patch 6 sensitivity envelope; replace with measured or published value",
            }
        )
    write_csv(path, rows)


def run_scenario(
    binary: Path,
    config: Path,
    output_directory: Path,
    scenario: Scenario,
    workloads: Sequence[str],
    queries_per_workload: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)
    per_query_path = (raw_directory / f"{scenario.name}_per_query.csv").resolve()
    aggregate_path = (raw_directory / f"{scenario.name}_aggregate.csv").resolve()
    command = [str(binary), str(config)]
    overrides: dict[str, str | float] = dict(scenario.overrides)
    overrides["output.per_query_csv"] = str(per_query_path)
    overrides["output.aggregate_csv"] = str(aggregate_path)
    for key, value in overrides.items():
        command.extend(("--set", f"{key}={value}"))
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (raw_directory / f"{scenario.name}.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise SensitivityError(
            f"{scenario.name}: simulator failed with code {completed.returncode}; "
            f"see {raw_directory / f'{scenario.name}.log'}"
        )
    if "Trace simulation selection: queries=" not in completed.stdout or \
            "splits=validation" not in completed.stdout:
        raise SensitivityError(f"{scenario.name}: validation-only execution not confirmed")
    per_query = read_csv(per_query_path, PER_QUERY_COLUMNS)
    aggregate = read_csv(aggregate_path, AGGREGATE_COLUMNS)
    validate_outputs(
        per_query,
        aggregate,
        workloads,
        queries_per_workload,
        scenario.name,
        scenario.values["near_bank_pe.cached_kv_lut_scale_overlap"],
    )
    return per_query, aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Patch 6 validation-only calibration sensitivity."
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--expected-workloads",
        type=parse_workloads,
        default=DEFAULT_WORKLOADS,
    )
    parser.add_argument("--expected-queries-per-workload", type=int, default=100)
    parser.add_argument("--scenarios", type=parse_scenarios, default=parse_scenarios("all"))
    args = parser.parse_args()
    binary = args.binary.resolve()
    config = args.config.resolve()
    output_directory = args.output_directory.resolve()
    if not binary.is_file() or not config.is_file():
        raise SystemExit("--binary and --config must name existing files")
    if args.expected_queries_per_workload <= 0:
        raise SystemExit("--expected-queries-per-workload must be positive")

    selected_names = set(args.scenarios)
    selected_scenarios = [
        scenario for scenario in SCENARIOS if scenario.name in selected_names
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    write_parameter_manifest(output_directory / "patch6_parameter_manifest.csv")

    overall_rows: list[dict[str, str | int | float]] = []
    workload_rows: list[dict[str, str | int | float]] = []
    try:
        for index, scenario in enumerate(selected_scenarios, start=1):
            print(f"[{index}/{len(selected_scenarios)}] {scenario.name}", flush=True)
            per_query, _ = run_scenario(
                binary,
                config,
                output_directory,
                scenario,
                args.expected_workloads,
                args.expected_queries_per_workload,
            )
            for workload in args.expected_workloads:
                workload_rows.append(
                    metric_row(
                        scenario,
                        workload,
                        select(per_query, workload, LOCAL),
                        select(per_query, workload, NO_LOCAL),
                    )
                )
            overall_rows.append(
                metric_row(
                    scenario,
                    "__all__",
                    select(per_query, None, LOCAL),
                    select(per_query, None, NO_LOCAL),
                )
            )
    except SensitivityError as error:
        raise SystemExit(f"ERROR: {error}") from error

    base_overall = next(row for row in overall_rows if row["scenario"] == "base")
    base_workload = {
        row["workload"]: row
        for row in workload_rows
        if row["scenario"] == "base"
    }
    for row in overall_rows:
        row["latency_change_vs_base"] = (
            float(row["mean_latency_ns"]) / float(base_overall["mean_latency_ns"]) - 1.0
        )
    for row in workload_rows:
        row["latency_change_vs_base"] = (
            float(row["mean_latency_ns"])
            / float(base_workload[str(row["workload"])]["mean_latency_ns"])
            - 1.0
        )

    overall_path = output_directory / "patch6_sensitivity_overall.csv"
    workload_path = output_directory / "patch6_sensitivity_by_workload.csv"
    write_csv(overall_path, overall_rows)
    write_csv(workload_path, workload_rows)

    print("\nPatch 6 validation-only sensitivity summary")
    print(
        "scenario mean_ms p95_ms change_vs_base local_latency_reduction "
        "local_faster_queries stall overlap lut_raw scale_raw hidden_kv "
        "kv_path bottleneck stage_share query_fraction"
    )
    for row in overall_rows:
        print(
            f"{row['scenario']:<24} "
            f"{float(row['mean_latency_ns']) / 1e6:>8.3f} "
            f"{float(row['p95_latency_ns']) / 1e6:>8.3f} "
            f"{float(row['latency_change_vs_base']):>+14.2%} "
            f"{float(row['local_latency_reduction_vs_no_local']):>23.4%} "
            f"{float(row['local_faster_query_fraction']):>20.2%} "
            f"{float(row['mean_traffic_stall_fraction']):>6.2%} "
            f"{float(row['cached_kv_lut_scale_overlap']):>7.2f} "
            f"{float(row['mean_cached_kv_raw_lut_fraction']):>7.2%} "
            f"{float(row['mean_cached_kv_raw_scale_fraction']):>9.2%} "
            f"{float(row['cached_kv_overlap_cycle_reduction']):>9.2%} "
            f"{float(row['mean_cached_kv_pipeline_critical_path_fraction']):>7.2%} "
            f"{row['dominant_bottleneck_stage']} "
            f"{float(row['mean_bottleneck_stage_fraction']):.2%} "
            f"{float(row['bottleneck_query_fraction']):.2%}"
        )
    print(f"\nparameter_manifest={output_directory / 'patch6_parameter_manifest.csv'}")
    print(f"by_workload_csv={workload_path}")
    print(f"overall_csv={overall_path}")
    print("No test rows were simulated or reported; no PIM/H100 speedup was computed.")


if __name__ == "__main__":
    main()
