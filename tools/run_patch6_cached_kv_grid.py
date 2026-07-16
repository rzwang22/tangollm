#!/usr/bin/env python3
"""Run the validation-only Patch 6 cached-KV joint calibration grid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

from run_patch6_sensitivity import (
    DEFAULT_WORKLOADS,
    LOCAL,
    NO_LOCAL,
    Scenario,
    SensitivityError,
    base_values,
    metric_row,
    parse_workloads,
    run_scenario,
    select,
    write_csv,
)


DEFAULT_LUT_CYCLES = (0.5, 1.0, 2.0)
DEFAULT_SCALE_CYCLES = (0.25, 0.5, 1.0, 2.0)
DEFAULT_OVERLAPS = (0.0, 0.5, 1.0)
REFERENCE_POINT = (1.0, 1.0, 0.0)


def parse_positive_values(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("grid values must be numeric") from error
    if not values or any(
        not math.isfinite(item) or item <= 0.0 for item in values
    ):
        raise argparse.ArgumentTypeError("grid values must be positive and finite")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("grid values must be unique")
    return values


def parse_overlap_values(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("overlap values must be numeric") from error
    if not values or any(
        not math.isfinite(item) or item < 0.0 or item > 1.0 for item in values
    ):
        raise argparse.ArgumentTypeError("overlap values must be within [0,1]")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("overlap values must be unique")
    return values


def value_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def scenario_name(lut: float, scale: float, overlap: float) -> str:
    return (
        f"lut{value_slug(lut)}_scale{value_slug(scale)}_"
        f"overlap{value_slug(overlap)}"
    )


def make_grid(
    lut_cycles: Sequence[float],
    scale_cycles: Sequence[float],
    overlaps: Sequence[float],
) -> tuple[Scenario, ...]:
    scenarios = []
    for lut in lut_cycles:
        for scale in scale_cycles:
            for overlap in overlaps:
                values = base_values()
                values["near_bank_pe.q8k2_lut_group_cycles"] = lut
                values["near_bank_pe.p8v2_lut_group_cycles"] = lut
                values["near_bank_pe.cached_kv_scale_group_cycles"] = scale
                values["near_bank_pe.cached_kv_lut_scale_overlap"] = overlap
                scenarios.append(
                    Scenario(
                        scenario_name(lut, scale, overlap),
                        "cached_kv_joint_calibration",
                        (
                            f"INT2 LUT={lut:g} cycles/group, "
                            f"scale={scale:g} cycles/group, overlap={overlap:g}"
                        ),
                        values,
                    )
                )
    return tuple(scenarios)


def grid_values(scenario: Scenario) -> tuple[float, float, float]:
    return (
        scenario.values["near_bank_pe.q8k2_lut_group_cycles"],
        scenario.values["near_bank_pe.cached_kv_scale_group_cycles"],
        scenario.values["near_bank_pe.cached_kv_lut_scale_overlap"],
    )


def add_grid_fields(
    row: dict[str, str | int | float], scenario: Scenario
) -> dict[str, str | int | float]:
    lut, scale, overlap = grid_values(scenario)
    row["grid_int2_lut_group_cycles"] = lut
    row["grid_cached_kv_scale_group_cycles"] = scale
    row["grid_cached_kv_lut_scale_overlap"] = overlap
    row["is_reference_point"] = int((lut, scale, overlap) == REFERENCE_POINT)
    return row


def write_grid_manifest(path: Path, scenarios: Sequence[Scenario]) -> None:
    rows = []
    for scenario in scenarios:
        lut, scale, overlap = grid_values(scenario)
        rows.append(
            {
                "scenario": scenario.name,
                "int2_lut_group_cycles": lut,
                "cached_kv_scale_group_cycles": scale,
                "cached_kv_lut_scale_overlap": overlap,
                "is_reference_point": int((lut, scale, overlap) == REFERENCE_POINT),
                "calibration_status": "uncalibrated_assumption",
            }
        )
    write_csv(path, rows)


def set_reference_changes(
    overall_rows: list[dict[str, str | int | float]],
    workload_rows: list[dict[str, str | int | float]],
) -> None:
    reference_name = scenario_name(*REFERENCE_POINT)
    reference_overall = next(
        row for row in overall_rows if row["scenario"] == reference_name
    )
    references_by_workload = {
        str(row["workload"]): row
        for row in workload_rows
        if row["scenario"] == reference_name
    }
    for row in overall_rows:
        change = (
            float(row["mean_latency_ns"])
            / float(reference_overall["mean_latency_ns"])
            - 1.0
        )
        row["latency_change_vs_reference"] = change
        row["latency_change_vs_base"] = change
    for row in workload_rows:
        reference = references_by_workload[str(row["workload"])]
        change = (
            float(row["mean_latency_ns"]) / float(reference["mean_latency_ns"])
            - 1.0
        )
        row["latency_change_vs_reference"] = change
        row["latency_change_vs_base"] = change


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the validation-only cached-KV joint calibration grid."
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--expected-workloads", type=parse_workloads, default=DEFAULT_WORKLOADS
    )
    parser.add_argument("--expected-queries-per-workload", type=int, default=100)
    parser.add_argument(
        "--lut-cycles",
        type=parse_positive_values,
        default=DEFAULT_LUT_CYCLES,
        help="comma-separated tied Q8K2/P8V2 LUT cycles/group",
    )
    parser.add_argument(
        "--scale-cycles",
        type=parse_positive_values,
        default=DEFAULT_SCALE_CYCLES,
        help="comma-separated cached-KV scale cycles/group",
    )
    parser.add_argument(
        "--overlaps",
        type=parse_overlap_values,
        default=DEFAULT_OVERLAPS,
        help="comma-separated LUT/scale overlap fractions",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    config = args.config.resolve()
    output_directory = args.output_directory.resolve()
    if not binary.is_file() or not config.is_file():
        raise SystemExit("--binary and --config must name existing files")
    if args.expected_queries_per_workload <= 0:
        raise SystemExit("--expected-queries-per-workload must be positive")
    if not all(
        reference in values
        for reference, values in zip(
            REFERENCE_POINT, (args.lut_cycles, args.scale_cycles, args.overlaps)
        )
    ):
        raise SystemExit("grid must contain the reference point LUT=1, scale=1, overlap=0")

    scenarios = make_grid(args.lut_cycles, args.scale_cycles, args.overlaps)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "patch6_cached_kv_grid_manifest.csv"
    write_grid_manifest(manifest_path, scenarios)

    overall_rows: list[dict[str, str | int | float]] = []
    workload_rows: list[dict[str, str | int | float]] = []
    try:
        for index, scenario in enumerate(scenarios, start=1):
            print(f"[{index}/{len(scenarios)}] {scenario.name}", flush=True)
            per_query, _ = run_scenario(
                binary,
                config,
                output_directory,
                scenario,
                args.expected_workloads,
                args.expected_queries_per_workload,
            )
            for workload in args.expected_workloads:
                row = metric_row(
                    scenario,
                    workload,
                    select(per_query, workload, LOCAL),
                    select(per_query, workload, NO_LOCAL),
                )
                workload_rows.append(add_grid_fields(row, scenario))
            row = metric_row(
                scenario,
                "__all__",
                select(per_query, None, LOCAL),
                select(per_query, None, NO_LOCAL),
            )
            overall_rows.append(add_grid_fields(row, scenario))
    except SensitivityError as error:
        raise SystemExit(f"ERROR: {error}") from error

    set_reference_changes(overall_rows, workload_rows)
    overall_path = output_directory / "patch6_cached_kv_grid_overall.csv"
    workload_path = output_directory / "patch6_cached_kv_grid_by_workload.csv"
    write_csv(overall_path, overall_rows)
    write_csv(workload_path, workload_rows)

    print("\nPatch 6 validation-only cached-KV joint calibration grid")
    print(
        "scenario lut scale overlap mean_ms p95_ms change_vs_reference "
        "kv_path lut_raw scale_raw hidden_kv local_gain bottleneck "
        "stage_share query_fraction"
    )
    for row in overall_rows:
        print(
            f"{row['scenario']:<31} "
            f"{float(row['grid_int2_lut_group_cycles']):>4g} "
            f"{float(row['grid_cached_kv_scale_group_cycles']):>5g} "
            f"{float(row['grid_cached_kv_lut_scale_overlap']):>7.2f} "
            f"{float(row['mean_latency_ns']) / 1e6:>8.3f} "
            f"{float(row['p95_latency_ns']) / 1e6:>8.3f} "
            f"{float(row['latency_change_vs_reference']):>+19.2%} "
            f"{float(row['mean_cached_kv_pipeline_critical_path_fraction']):>7.2%} "
            f"{float(row['mean_cached_kv_raw_lut_fraction']):>7.2%} "
            f"{float(row['mean_cached_kv_raw_scale_fraction']):>9.2%} "
            f"{float(row['mean_cached_kv_overlap_hidden_fraction']):>9.2%} "
            f"{float(row['local_latency_reduction_vs_no_local']):>9.4%} "
            f"{row['dominant_bottleneck_stage']} "
            f"{float(row['mean_bottleneck_stage_fraction']):.2%} "
            f"{float(row['bottleneck_query_fraction']):.2%}"
        )
    print(f"\nmanifest_csv={manifest_path}")
    print(f"by_workload_csv={workload_path}")
    print(f"overall_csv={overall_path}")
    print("No test rows were simulated or reported; no PIM/H100 speedup was computed.")


if __name__ == "__main__":
    main()
