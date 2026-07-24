#!/usr/bin/env python3
"""Create or fit a validation-only H100 profile for Patch 7A."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA_VERSION = "h100_profile_v1"
BASELINE = "h100_realistic_cache_path"
GRAPH_PLACEMENT = "source_dst_locality"
KV_PLACEMENT = "balanced"
MEMORY_TOKEN_TILE = "4"
DEFAULT_WORKLOADS = (
    "cora_node",
    "cora_link",
    "pubmed_node",
    "wikics",
    "arxiv",
)

IDENTITY_COLUMNS = (
    "workload",
    "split",
    "trace_order",
    "runtime_query_index",
    "trace_query_id",
)
TRACE_FEATURE_COLUMNS = (
    "selected_kv_bytes",
    "runtime_loaded_cache_bytes",
    "runtime_loaded_scale_bytes",
    "runtime_gather_index_metadata_bytes",
    "runtime_loaded_total_bytes",
    "trace_cached_qk_groups",
    "trace_cached_pv_groups",
    "h100_gnn_score_groups",
    "h100_gnn_message_groups",
    "h100_cached_kv_groups",
)
PROVENANCE_COLUMNS = (
    "batch_size",
    "device_name",
    "device_uuid",
    "cuda_version",
    "pytorch_version",
    "profile_run_id",
    "measurement_method",
    "warmup_iterations",
    "measured_iterations",
)
TIMING_COLUMNS = (
    "measured_total_latency_ns",
    "gnn_score_compute_ns",
    "gnn_message_compute_ns",
    "cached_kv_compute_ns",
    "cache_read_ns",
    "int2_unpack_ns",
    "scale_dequant_ns",
    "layout_conversion_ns",
    "irregular_gather_ns",
    "small_batch_penalty_ns",
    "fixed_overhead_ns",
)
STAGE_COLUMNS = TIMING_COLUMNS[1:]
PROFILE_COLUMNS = (
    "schema_version",
    *IDENTITY_COLUMNS,
    *TRACE_FEATURE_COLUMNS,
    *PROVENANCE_COLUMNS,
    *TIMING_COLUMNS,
)
SIMULATOR_COLUMNS = {
    *IDENTITY_COLUMNS,
    *TRACE_FEATURE_COLUMNS,
    "workload_mode",
    "evaluation_role",
    "graph_compute_placement",
    "kv_storage_placement",
    "baseline",
    "memory_token_tile",
    "total_cycles",
    "latency_ns",
    "critical_path_latency_ns",
    "gnn_score_cycles",
    "gnn_message_cycles",
    "cached_kv_cycles",
    "h100_cache_read_cycles",
    "h100_int2_unpack_cycles",
    "h100_scale_dequant_cycles",
    "h100_layout_conversion_cycles",
    "h100_irregular_gather_penalty_cycles",
    "h100_small_batch_penalty_cycles",
    "h100_fixed_overhead_cycles",
}


class CalibrationError(RuntimeError):
    pass


def parse_workloads(value: str) -> tuple[str, ...]:
    workloads = tuple(item.strip() for item in value.split(",") if item.strip())
    if not workloads or len(set(workloads)) != len(workloads):
        raise argparse.ArgumentTypeError("workload list must be non-empty and unique")
    return workloads


def read_csv(path: Path, required: set[str]) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise CalibrationError(f"missing CSV: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or ())
        missing = sorted(required - set(header))
        if missing:
            raise CalibrationError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise CalibrationError(f"{path}: empty or malformed CSV")
    return header, rows


def number(row: dict[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as error:
        raise CalibrationError(
            f"{context}: invalid {field}={row.get(field)!r}"
        ) from error
    if not math.isfinite(value):
        raise CalibrationError(f"{context}: non-finite {field}={row[field]!r}")
    return value


def integer(row: dict[str, str], field: str, context: str) -> int:
    try:
        return int(row[field])
    except (KeyError, ValueError) as error:
        raise CalibrationError(
            f"{context}: invalid integer {field}={row.get(field)!r}"
        ) from error


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in IDENTITY_COLUMNS)


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = fraction * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def fit_slope_origin(samples: Sequence[tuple[float, float]], name: str) -> float:
    denominator = sum(x * x for x, _ in samples)
    if denominator <= 0.0:
        raise CalibrationError(f"cannot fit {name}: zero feature energy")
    slope = sum(x * y for x, y in samples) / denominator
    if not math.isfinite(slope) or slope <= 0.0:
        raise CalibrationError(f"cannot fit {name}: non-positive slope {slope}")
    return slope


def run_feature_pass(binary: Path, config: Path, output: Path) -> list[dict[str, str]]:
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    per_query = (raw / "h100_features_per_query.csv").resolve()
    aggregate = (raw / "h100_features_aggregate.csv").resolve()
    command = [
        str(binary),
        str(config),
        "--set",
        f"output.per_query_csv={per_query}",
        "--set",
        f"output.aggregate_csv={aggregate}",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (raw / "feature_pass.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise CalibrationError(
            f"feature pass failed with code {completed.returncode}; "
            f"see {raw / 'feature_pass.log'}"
        )
    if "splits=validation" not in completed.stdout:
        raise CalibrationError("feature pass did not confirm validation-only execution")
    _, rows = read_csv(per_query, SIMULATOR_COLUMNS)
    return rows


def validate_feature_rows(
    rows: Sequence[dict[str, str]],
    workloads: Sequence[str],
    expected_queries_per_workload: int,
) -> float:
    expected_count = len(workloads) * expected_queries_per_workload
    if len(rows) != expected_count:
        raise CalibrationError(
            f"feature row count mismatch: expected {expected_count}, got {len(rows)}"
        )
    seen: set[tuple[str, ...]] = set()
    by_workload: dict[str, list[int]] = defaultdict(list)
    clock_values: list[float] = []
    for index, row in enumerate(rows):
        context = f"feature row {index + 2}"
        if row["workload"] not in workloads:
            raise CalibrationError(f"{context}: unexpected workload {row['workload']}")
        if row["workload_mode"] != "trace" or row["split"] != "validation":
            raise CalibrationError(f"{context}: Patch 7A accepts validation trace rows only")
        if row["evaluation_role"] != "config_selection":
            raise CalibrationError(f"{context}: expected evaluation_role=config_selection")
        if (
            row["graph_compute_placement"] != GRAPH_PLACEMENT
            or row["kv_storage_placement"] != KV_PLACEMENT
            or row["baseline"] != BASELINE
            or row["memory_token_tile"] != MEMORY_TOKEN_TILE
        ):
            raise CalibrationError(f"{context}: Patch 7A configuration is not frozen")
        identity = key(row)
        if identity in seen:
            raise CalibrationError(f"{context}: duplicate query identity {identity}")
        seen.add(identity)
        by_workload[row["workload"]].append(integer(row, "trace_order", context))
        for field in TRACE_FEATURE_COLUMNS:
            if number(row, field, context) < 0.0:
                raise CalibrationError(f"{context}: negative trace feature {field}")
        cycles = number(row, "total_cycles", context)
        latency = number(row, "latency_ns", context)
        if cycles <= 0.0 or latency <= 0.0:
            raise CalibrationError(f"{context}: non-positive simulator cycle/time")
        clock_values.append(latency / cycles)
    for workload in workloads:
        orders = by_workload.get(workload, [])
        if len(orders) != expected_queries_per_workload:
            raise CalibrationError(
                f"{workload}: expected {expected_queries_per_workload} queries, "
                f"got {len(orders)}"
            )
        if orders != sorted(orders) or len(set(orders)) != len(orders):
            raise CalibrationError(f"{workload}: trace_index order is not preserved")
    clock_ns = statistics.fmean(clock_values)
    if max(abs(value - clock_ns) for value in clock_values) > 1e-9:
        raise CalibrationError("feature rows use inconsistent clock_ns")
    return clock_ns


def write_template(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_COLUMNS)
        writer.writeheader()
        for feature in rows:
            row = {column: "" for column in PROFILE_COLUMNS}
            row["schema_version"] = SCHEMA_VERSION
            for column in (*IDENTITY_COLUMNS, *TRACE_FEATURE_COLUMNS):
                row[column] = feature[column]
            row["batch_size"] = "1"
            row["measurement_method"] = "cuda_event_stage_additive_v1"
            writer.writerow(row)


def validate_profile(
    path: Path,
    feature_rows: Sequence[dict[str, str]],
    stage_sum_relative_tolerance: float,
    stage_sum_absolute_tolerance_ns: float,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    header, rows = read_csv(path, set(PROFILE_COLUMNS))
    if tuple(header) != PROFILE_COLUMNS:
        raise CalibrationError(
            f"{path}: profile header must exactly match {SCHEMA_VERSION} schema"
        )
    feature_by_key = {key(row): row for row in feature_rows}
    profile_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    provenance_values: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        context = f"profile row {index + 2}"
        if row["schema_version"] != SCHEMA_VERSION:
            raise CalibrationError(f"{context}: unsupported schema_version")
        if row["split"] != "validation":
            raise CalibrationError(f"{context}: test profile rows are forbidden in Patch 7A")
        identity = key(row)
        if identity in profile_by_key:
            raise CalibrationError(f"{context}: duplicate query identity {identity}")
        profile_by_key[identity] = row
        if integer(row, "batch_size", context) != 1:
            raise CalibrationError(f"{context}: batch_size must be 1")
        if integer(row, "warmup_iterations", context) < 1:
            raise CalibrationError(f"{context}: warmup_iterations must be positive")
        if integer(row, "measured_iterations", context) < 1:
            raise CalibrationError(f"{context}: measured_iterations must be positive")
        for field in (
            "device_name",
            "device_uuid",
            "cuda_version",
            "pytorch_version",
            "profile_run_id",
        ):
            if not row[field].strip():
                raise CalibrationError(f"{context}: {field} must be non-empty")
            provenance_values[field].add(row[field])
        if row["measurement_method"] != "cuda_event_stage_additive_v1":
            raise CalibrationError(
                f"{context}: measurement_method must be "
                "cuda_event_stage_additive_v1"
            )
        feature = feature_by_key.get(identity)
        if feature is None:
            raise CalibrationError(f"{context}: profile query not found in validation trace")
        for field in TRACE_FEATURE_COLUMNS:
            measured = number(row, field, context)
            expected = number(feature, field, context)
            if not math.isclose(measured, expected, rel_tol=1e-9, abs_tol=1e-6):
                raise CalibrationError(
                    f"{context}: trace feature mismatch for {field}: "
                    f"profile={measured}, simulator={expected}"
                )
        total = number(row, "measured_total_latency_ns", context)
        stages = [number(row, field, context) for field in STAGE_COLUMNS]
        if total <= 0.0 or any(value < 0.0 for value in stages):
            raise CalibrationError(f"{context}: timing values must be non-negative and total positive")
        stage_sum = sum(stages)
        tolerance = max(
            stage_sum_absolute_tolerance_ns,
            stage_sum_relative_tolerance * total,
        )
        if abs(stage_sum - total) > tolerance:
            raise CalibrationError(
                f"{context}: stage sum mismatch: stages={stage_sum}, "
                f"total={total}, tolerance={tolerance}"
            )
    expected_keys = set(feature_by_key)
    profile_keys = set(profile_by_key)
    if profile_keys != expected_keys:
        missing = len(expected_keys - profile_keys)
        extra = len(profile_keys - expected_keys)
        raise CalibrationError(
            f"profile/query coverage mismatch: missing={missing}, extra={extra}"
        )
    for field in (
        "device_name",
        "device_uuid",
        "cuda_version",
        "pytorch_version",
        "profile_run_id",
    ):
        if len(provenance_values[field]) != 1:
            raise CalibrationError(f"profile mixes multiple {field} values")
    return [(feature, profile_by_key[key(feature)]) for feature in feature_rows]


def fit_parameters(
    joined: Sequence[tuple[dict[str, str], dict[str, str]]], clock_ns: float
) -> dict[str, float]:
    group_samples: list[tuple[float, float]] = []
    for feature, profile in joined:
        group_samples.extend(
            (
                (number(feature, "h100_gnn_score_groups", "fit"), number(profile, "gnn_score_compute_ns", "fit")),
                (number(feature, "h100_gnn_message_groups", "fit"), number(profile, "gnn_message_compute_ns", "fit")),
                (number(feature, "h100_cached_kv_groups", "fit"), number(profile, "cached_kv_compute_ns", "fit")),
            )
        )
    group_ns = fit_slope_origin(group_samples, "h100_group_cycles")
    cache_ns_per_byte = fit_slope_origin(
        [
            (
                number(feature, "runtime_loaded_total_bytes", "fit"),
                number(profile, "cache_read_ns", "fit"),
            )
            for feature, profile in joined
        ],
        "h100_cache_bytes_per_cycle",
    )
    cached_groups = [
        number(feature, "h100_cached_kv_groups", "fit")
        for feature, _ in joined
    ]

    def group_stage(field: str) -> float:
        return fit_slope_origin(
            [
                (groups, number(profile, field, "fit"))
                for groups, (_, profile) in zip(cached_groups, joined)
            ],
            field,
        ) / clock_ns

    irregular_slope = fit_slope_origin(
        [
            (number(profile, "cache_read_ns", "fit"), number(profile, "irregular_gather_ns", "fit"))
            for _, profile in joined
        ],
        "h100_irregular_gather_penalty",
    )
    small_batch_slope = fit_slope_origin(
        [
            (
                number(profile, "cache_read_ns", "fit")
                + number(profile, "irregular_gather_ns", "fit"),
                number(profile, "small_batch_penalty_ns", "fit"),
            )
            for _, profile in joined
        ],
        "h100_small_batch_efficiency",
    )
    return {
        "h100_group_cycles": group_ns / clock_ns,
        "h100_cache_bytes_per_cycle": clock_ns / cache_ns_per_byte,
        "h100_int2_unpack_group_cycles": group_stage("int2_unpack_ns"),
        "h100_scale_dequant_group_cycles": group_stage("scale_dequant_ns"),
        "h100_layout_conversion_group_cycles": group_stage("layout_conversion_ns"),
        "h100_irregular_gather_penalty": 1.0 + irregular_slope,
        "h100_small_batch_efficiency": 1.0 / (1.0 + small_batch_slope),
        "h100_fixed_overhead_cycles": statistics.fmean(
            number(profile, "fixed_overhead_ns", "fit")
            for _, profile in joined
        ) / clock_ns,
    }


def predict(
    feature: dict[str, str], parameters: dict[str, float], clock_ns: float
) -> dict[str, float]:
    score = number(feature, "h100_gnn_score_groups", "prediction") * parameters["h100_group_cycles"] * clock_ns
    message = number(feature, "h100_gnn_message_groups", "prediction") * parameters["h100_group_cycles"] * clock_ns
    cached = number(feature, "h100_cached_kv_groups", "prediction") * parameters["h100_group_cycles"] * clock_ns
    cache = number(feature, "runtime_loaded_total_bytes", "prediction") / parameters["h100_cache_bytes_per_cycle"] * clock_ns
    irregular = cache * (parameters["h100_irregular_gather_penalty"] - 1.0)
    small = (cache + irregular) * (1.0 / parameters["h100_small_batch_efficiency"] - 1.0)
    groups = number(feature, "h100_cached_kv_groups", "prediction")
    unpack = groups * parameters["h100_int2_unpack_group_cycles"] * clock_ns
    scale = groups * parameters["h100_scale_dequant_group_cycles"] * clock_ns
    layout = groups * parameters["h100_layout_conversion_group_cycles"] * clock_ns
    fixed = parameters["h100_fixed_overhead_cycles"] * clock_ns
    stages = {
        "predicted_gnn_score_compute_ns": score,
        "predicted_gnn_message_compute_ns": message,
        "predicted_cached_kv_compute_ns": cached,
        "predicted_cache_read_ns": cache,
        "predicted_int2_unpack_ns": unpack,
        "predicted_scale_dequant_ns": scale,
        "predicted_layout_conversion_ns": layout,
        "predicted_irregular_gather_ns": irregular,
        "predicted_small_batch_penalty_ns": small,
        "predicted_fixed_overhead_ns": fixed,
    }
    stages["predicted_total_latency_ns"] = sum(stages.values())
    return stages


def run_calibrated_pass(
    binary: Path,
    config: Path,
    output: Path,
    feature_rows: Sequence[dict[str, str]],
    parameters: dict[str, float],
    workloads: Sequence[str],
    expected_queries_per_workload: int,
    clock_ns: float,
) -> list[dict[str, str]]:
    raw = output / "raw"
    per_query = (raw / "h100_calibrated_per_query.csv").resolve()
    aggregate = (raw / "h100_calibrated_aggregate.csv").resolve()
    command = [
        str(binary),
        str(config),
        "--set",
        f"output.per_query_csv={per_query}",
        "--set",
        f"output.aggregate_csv={aggregate}",
    ]
    for name, value in parameters.items():
        command.extend(("--set", f"near_bank_pe.{name}={value:.17g}"))
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (raw / "calibrated_pass.log").write_text(completed.stdout)
    if completed.returncode != 0:
        raise CalibrationError(
            f"calibrated pass failed with code {completed.returncode}; "
            f"see {raw / 'calibrated_pass.log'}"
        )
    _, calibrated = read_csv(per_query, SIMULATOR_COLUMNS)
    calibrated_clock = validate_feature_rows(
        calibrated, workloads, expected_queries_per_workload
    )
    if not math.isclose(calibrated_clock, clock_ns, rel_tol=0.0, abs_tol=1e-9):
        raise CalibrationError("calibrated pass changed clock_ns")
    calibrated_by_key = {key(row): row for row in calibrated}
    if set(calibrated_by_key) != {key(row) for row in feature_rows}:
        raise CalibrationError("calibrated pass changed validation query coverage")
    for feature in feature_rows:
        row = calibrated_by_key[key(feature)]
        for field in TRACE_FEATURE_COLUMNS:
            if not math.isclose(
                number(row, field, "calibrated pass"),
                number(feature, field, "feature pass"),
                rel_tol=1e-9,
                abs_tol=1e-6,
            ):
                raise CalibrationError(
                    f"calibrated pass changed trace feature {field} for {key(feature)}"
                )
        expected = predict(feature, parameters, clock_ns)[
            "predicted_total_latency_ns"
        ]
        actual = number(row, "critical_path_latency_ns", "calibrated pass")
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-4):
            raise CalibrationError(
                f"Python/C++ calibrated prediction mismatch for {key(feature)}: "
                f"python={expected}, simulator={actual}"
            )
    return calibrated


def metrics(rows: Sequence[dict[str, str]]) -> dict[str, float]:
    measured = [float(row["measured_total_latency_ns"]) for row in rows]
    predicted = [float(row["predicted_total_latency_ns"]) for row in rows]
    absolute = [abs(p - m) for p, m in zip(predicted, measured)]
    percentage = [error / measured_value for error, measured_value in zip(absolute, measured)]
    return {
        "num_queries": float(len(rows)),
        "measured_mean_ns": statistics.fmean(measured),
        "measured_p50_ns": percentile(measured, 0.50),
        "measured_p95_ns": percentile(measured, 0.95),
        "predicted_mean_ns": statistics.fmean(predicted),
        "predicted_p50_ns": percentile(predicted, 0.50),
        "predicted_p95_ns": percentile(predicted, 0.95),
        "mae_ns": statistics.fmean(absolute),
        "rmse_ns": math.sqrt(statistics.fmean(error * error for error in absolute)),
        "mape": statistics.fmean(percentage),
        "p95_absolute_percentage_error": percentile(percentage, 0.95),
    }


def write_dict_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output: Path,
    joined: Sequence[tuple[dict[str, str], dict[str, str]]],
    parameters: dict[str, float],
    clock_ns: float,
    calibrated_rows: Sequence[dict[str, str]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    calibrated_by_key = {key(row): row for row in calibrated_rows}
    per_query: list[dict[str, object]] = []
    for feature, profile in joined:
        prediction = predict(feature, parameters, clock_ns)
        prediction["predicted_total_latency_ns"] = number(
            calibrated_by_key[key(feature)],
            "critical_path_latency_ns",
            "calibrated output",
        )
        measured = number(profile, "measured_total_latency_ns", "output")
        predicted = prediction["predicted_total_latency_ns"]
        row: dict[str, object] = {
            **{column: feature[column] for column in IDENTITY_COLUMNS},
            "device_name": profile["device_name"],
            "profile_run_id": profile["profile_run_id"],
            "measured_total_latency_ns": measured,
            **prediction,
            "residual_ns": predicted - measured,
            "absolute_error_ns": abs(predicted - measured),
            "absolute_percentage_error": abs(predicted - measured) / measured,
        }
        per_query.append(row)
    write_dict_rows(output / "patch7a_h100_validation_per_query.csv", per_query)

    by_workload: dict[str, dict[str, float]] = {}
    workload_rows: list[dict[str, object]] = []
    for workload in sorted({str(row["workload"]) for row in per_query}):
        selected = [row for row in per_query if row["workload"] == workload]
        result = metrics(selected)  # type: ignore[arg-type]
        by_workload[workload] = result
        workload_rows.append({"workload": workload, **result})
    write_dict_rows(output / "patch7a_h100_validation_by_workload.csv", workload_rows)
    overall = metrics(per_query)  # type: ignore[arg-type]
    write_dict_rows(
        output / "patch7a_h100_validation_overall.csv",
        [{"scope": "all_validation", **overall}],
    )

    parameter_rows = []
    units = {
        "h100_group_cycles": "cycles/group",
        "h100_cache_bytes_per_cycle": "bytes/cycle",
        "h100_int2_unpack_group_cycles": "cycles/group",
        "h100_scale_dequant_group_cycles": "cycles/group",
        "h100_layout_conversion_group_cycles": "cycles/group",
        "h100_irregular_gather_penalty": "ratio",
        "h100_small_batch_efficiency": "fraction",
        "h100_fixed_overhead_cycles": "cycles/query",
    }
    for name, value in parameters.items():
        parameter_rows.append(
            {
                "parameter": f"near_bank_pe.{name}",
                "value": value,
                "unit": units[name],
                "fit_split": "validation",
                "num_queries": len(joined),
                "fit_method": "stage_supervised_origin_least_squares",
            }
        )
    write_dict_rows(output / "patch7a_h100_parameter_manifest.csv", parameter_rows)
    yaml_path = output / "patch7a_h100_calibrated_parameters.yaml"
    with yaml_path.open("w") as handle:
        handle.write("# Patch 7A validation-only calibration; not a test result.\n")
        handle.write("near_bank_pe:\n")
        for name, value in parameters.items():
            handle.write(f"  {name}: {value:.12g}\n")
    return overall, by_workload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or fit the strict Patch 7A H100 validation profile."
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--expected-workloads", type=parse_workloads, default=DEFAULT_WORKLOADS
    )
    parser.add_argument("--expected-queries-per-workload", type=int, default=100)
    parser.add_argument("--stage-sum-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--stage-sum-absolute-tolerance-ns", type=float, default=1.0)
    parser.add_argument("--max-overall-mape", type=float, default=0.15)
    parser.add_argument("--max-workload-mape", type=float, default=0.20)
    args = parser.parse_args()
    try:
        binary = args.binary.resolve()
        config = args.config.resolve()
        output = args.output_directory.resolve()
        if not binary.is_file() or not config.is_file():
            raise CalibrationError("binary or config does not exist")
        if args.expected_queries_per_workload <= 0:
            raise CalibrationError("expected query count must be positive")
        for name in (
            "stage_sum_relative_tolerance",
            "stage_sum_absolute_tolerance_ns",
            "max_overall_mape",
            "max_workload_mape",
        ):
            if getattr(args, name) < 0.0:
                raise CalibrationError(f"{name} must be non-negative")
        output.mkdir(parents=True, exist_ok=True)
        features = run_feature_pass(binary, config, output)
        clock_ns = validate_feature_rows(
            features, args.expected_workloads, args.expected_queries_per_workload
        )
        template = output / "h100_profile_template.csv"
        write_template(template, features)
        if args.profile is None:
            print("Patch 7A H100 validation feature pass completed")
            print(f"validation_queries={len(features)}, clock_ns={clock_ns:.6f}")
            print(f"profile_template={template}")
            print("Fill measured/provenance columns, then rerun with --profile FILE.")
            print("No test rows or PIM/H100 speedup were evaluated.")
            return
        joined = validate_profile(
            args.profile.resolve(),
            features,
            args.stage_sum_relative_tolerance,
            args.stage_sum_absolute_tolerance_ns,
        )
        parameters = fit_parameters(joined, clock_ns)
        calibrated_rows = run_calibrated_pass(
            binary,
            config,
            output,
            features,
            parameters,
            args.expected_workloads,
            args.expected_queries_per_workload,
            clock_ns,
        )
        overall, by_workload = write_outputs(
            output, joined, parameters, clock_ns, calibrated_rows
        )
        failing = [
            workload
            for workload, result in by_workload.items()
            if result["mape"] > args.max_workload_mape
        ]
        if overall["mape"] > args.max_overall_mape or failing:
            raise CalibrationError(
                "calibration error threshold exceeded: "
                f"overall_mape={overall['mape']:.4%}, "
                f"failing_workloads={','.join(failing) or 'none'}"
            )
        print("Patch 7A H100 validation calibration passed")
        print(
            f"validation_queries={len(joined)}, mean_ms="
            f"{overall['measured_mean_ns'] / 1e6:.6f}, "
            f"mape={overall['mape']:.4%}, "
            f"p95_ape={overall['p95_absolute_percentage_error']:.4%}"
        )
        for workload in args.expected_workloads:
            result = by_workload[workload]
            print(
                f"{workload} queries={int(result['num_queries'])} "
                f"mape={result['mape']:.4%} "
                f"p95_ape={result['p95_absolute_percentage_error']:.4%}"
            )
        print(f"calibrated_parameters={output / 'patch7a_h100_calibrated_parameters.yaml'}")
        print("Validation was used for fitting; test was not read and speedup was not computed.")
    except CalibrationError as error:
        print(f"ERROR: Patch 7A calibration: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
