#!/usr/bin/env python3
"""Run validation-only local-combine VADD/bank-link break-even analysis."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from run_patch6_cached_kv_grid import parse_positive_values, value_slug
from run_patch6_sensitivity import (
    DEFAULT_WORKLOADS,
    LOCAL,
    NO_LOCAL,
    Scenario,
    SensitivityError,
    base_values,
    mean,
    metric_row,
    parse_workloads,
    percentile,
    run_scenario,
    select,
    write_csv,
)


DEFAULT_VADD_CYCLES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0)
DEFAULT_BANK_TO_PC_BANDWIDTHS = (8.0, 16.0, 32.0, 64.0, 128.0)
ANCHOR_REFERENCE_POINT = (1.0, 32.0)


@dataclass(frozen=True)
class CachedKVAnchor:
    name: str
    lut_cycles: float
    scale_cycles: float
    overlap: float
    description: str


@dataclass(frozen=True)
class BreakEvenPoint:
    anchor: CachedKVAnchor
    vadd_cycles: float
    bank_to_pc_bandwidth: float
    scenario: Scenario


ANCHORS = (
    CachedKVAnchor(
        "conservative_serial",
        2.0,
        2.0,
        0.0,
        "slow serial cached-KV envelope",
    ),
    CachedKVAnchor(
        "reference_serial",
        1.0,
        1.0,
        0.0,
        "Patch 6 serial reference",
    ),
    CachedKVAnchor(
        "optimistic_pipelined",
        0.5,
        0.25,
        1.0,
        "fast fully overlapped cached-KV envelope",
    ),
)


def scenario_name(anchor: CachedKVAnchor, vadd: float, bandwidth: float) -> str:
    return (
        f"{anchor.name}_vadd{value_slug(vadd)}_"
        f"b2pc{value_slug(bandwidth)}"
    )


def make_grid(
    vadd_cycles: Sequence[float], bank_to_pc_bandwidths: Sequence[float]
) -> tuple[BreakEvenPoint, ...]:
    points = []
    for anchor in ANCHORS:
        for bandwidth in bank_to_pc_bandwidths:
            for vadd in vadd_cycles:
                values = base_values()
                values["near_bank_pe.q8k2_lut_group_cycles"] = anchor.lut_cycles
                values["near_bank_pe.p8v2_lut_group_cycles"] = anchor.lut_cycles
                values["near_bank_pe.cached_kv_scale_group_cycles"] = (
                    anchor.scale_cycles
                )
                values["near_bank_pe.cached_kv_lut_scale_overlap"] = anchor.overlap
                values["near_bank_pe.vadd_group_cycles"] = vadd
                values[
                    "communication.bank_to_pc_bandwidth_bytes_per_cycle_per_bank"
                ] = bandwidth
                scenario = Scenario(
                    scenario_name(anchor, vadd, bandwidth),
                    "local_combine_break_even",
                    (
                        f"{anchor.name}, VADD={vadd:g} cycles/group, "
                        f"bank-to-PC={bandwidth:g} bytes/cycle/bank"
                    ),
                    values,
                )
                points.append(BreakEvenPoint(anchor, vadd, bandwidth, scenario))
    return tuple(points)


def write_manifest(path: Path, points: Sequence[BreakEvenPoint]) -> None:
    rows = []
    for point in points:
        rows.append(
            {
                "scenario": point.scenario.name,
                "cached_kv_anchor": point.anchor.name,
                "anchor_lut_group_cycles": point.anchor.lut_cycles,
                "anchor_scale_group_cycles": point.anchor.scale_cycles,
                "anchor_overlap": point.anchor.overlap,
                "vadd_group_cycles": point.vadd_cycles,
                "bank_to_pc_bandwidth_bytes_per_cycle_per_bank": (
                    point.bank_to_pc_bandwidth
                ),
                "is_anchor_reference_point": int(
                    (point.vadd_cycles, point.bank_to_pc_bandwidth)
                    == ANCHOR_REFERENCE_POINT
                ),
                "calibration_status": "uncalibrated_assumption",
            }
        )
    write_csv(path, rows)


def add_break_even_fields(
    row: dict[str, str | int | float],
    point: BreakEvenPoint,
    local_rows: Sequence[dict[str, str]],
    no_local_rows: Sequence[dict[str, str]],
) -> dict[str, str | int | float]:
    local_mean = mean(local_rows, "critical_path_latency_ns")
    no_local_mean = mean(no_local_rows, "critical_path_latency_ns")
    no_local_latencies = [
        float(item["critical_path_latency_ns"]) for item in no_local_rows
    ]
    row["cached_kv_anchor"] = point.anchor.name
    row["anchor_lut_group_cycles"] = point.anchor.lut_cycles
    row["anchor_scale_group_cycles"] = point.anchor.scale_cycles
    row["anchor_overlap"] = point.anchor.overlap
    row["grid_vadd_group_cycles"] = point.vadd_cycles
    row["grid_bank_to_pc_bandwidth_bytes_per_cycle_per_bank"] = (
        point.bank_to_pc_bandwidth
    )
    row["is_anchor_reference_point"] = int(
        (point.vadd_cycles, point.bank_to_pc_bandwidth) == ANCHOR_REFERENCE_POINT
    )
    row["mean_local_latency_ns"] = local_mean
    row["mean_no_local_latency_ns"] = no_local_mean
    row["p95_local_latency_ns"] = row["p95_latency_ns"]
    row["p95_no_local_latency_ns"] = percentile(no_local_latencies, 0.95)
    row["local_absolute_savings_ns"] = no_local_mean - local_mean
    row["local_wins_mean"] = int(no_local_mean > local_mean)
    return row


def set_anchor_reference_changes(
    rows: list[dict[str, str | int | float]],
) -> None:
    references = {
        (str(row["cached_kv_anchor"]), str(row["workload"])): row
        for row in rows
        if int(row["is_anchor_reference_point"]) == 1
    }
    for row in rows:
        key = (str(row["cached_kv_anchor"]), str(row["workload"]))
        reference = references.get(key)
        if reference is None:
            raise SensitivityError(
                f"missing anchor reference for {key[0]}/{key[1]}"
            )
        change = (
            float(row["mean_latency_ns"]) / float(reference["mean_latency_ns"])
            - 1.0
        )
        row["latency_change_vs_anchor_reference"] = change
        row["latency_change_vs_base"] = change


def estimate_break_even(
    rows: Sequence[dict[str, str | int | float]],
    anchor: str,
    workload: str,
    bandwidth: float,
) -> dict[str, str | int | float]:
    ordered = sorted(rows, key=lambda row: float(row["grid_vadd_group_cycles"]))
    deltas = [float(row["local_absolute_savings_ns"]) for row in ordered]
    tolerance = 1e-6
    if any(
        later > earlier + tolerance
        for earlier, later in zip(deltas, deltas[1:])
    ):
        raise SensitivityError(
            f"{anchor}/{workload}/b2pc={bandwidth:g}: "
            "local savings are not monotonic in VADD cost"
        )

    exact = [
        float(row["grid_vadd_group_cycles"])
        for row, delta in zip(ordered, deltas)
        if abs(delta) <= tolerance
    ]
    winning = [
        float(row["grid_vadd_group_cycles"])
        for row, delta in zip(ordered, deltas)
        if delta > tolerance
    ]
    losing = [
        float(row["grid_vadd_group_cycles"])
        for row, delta in zip(ordered, deltas)
        if delta < -tolerance
    ]

    status = "interpolated"
    threshold: float | str = ""
    if len(exact) == len(ordered):
        status = "flat_at_zero"
    elif exact:
        status = "exact_grid_point"
        threshold = exact[0]
    elif not losing:
        status = "above_sweep"
    elif not winning:
        status = "below_sweep"
    else:
        upper_index = next(index for index, delta in enumerate(deltas) if delta < 0.0)
        lower_index = upper_index - 1
        lower_vadd = float(ordered[lower_index]["grid_vadd_group_cycles"])
        upper_vadd = float(ordered[upper_index]["grid_vadd_group_cycles"])
        lower_delta = deltas[lower_index]
        upper_delta = deltas[upper_index]
        threshold = lower_vadd + lower_delta / (lower_delta - upper_delta) * (
            upper_vadd - lower_vadd
        )

    return {
        "cached_kv_anchor": anchor,
        "workload": workload,
        "bank_to_pc_bandwidth_bytes_per_cycle_per_bank": bandwidth,
        "criterion": "mean_latency_ns",
        "status": status,
        "break_even_vadd_group_cycles": threshold,
        "largest_winning_tested_vadd": max(winning) if winning else "",
        "smallest_losing_tested_vadd": min(losing) if losing else "",
        "minimum_tested_vadd": float(ordered[0]["grid_vadd_group_cycles"]),
        "maximum_tested_vadd": float(ordered[-1]["grid_vadd_group_cycles"]),
        "savings_at_minimum_vadd_ns": deltas[0],
        "savings_at_maximum_vadd_ns": deltas[-1],
    }


def build_break_even_rows(
    rows: Sequence[dict[str, str | int | float]],
    workloads: Sequence[str],
    bandwidths: Sequence[float],
) -> list[dict[str, str | int | float]]:
    output = []
    for anchor in ANCHORS:
        for workload in (*workloads, "__all__"):
            for bandwidth in bandwidths:
                group = [
                    row
                    for row in rows
                    if row["cached_kv_anchor"] == anchor.name
                    and row["workload"] == workload
                    and float(
                        row[
                            "grid_bank_to_pc_bandwidth_bytes_per_cycle_per_bank"
                        ]
                    )
                    == bandwidth
                ]
                if not group:
                    raise SensitivityError(
                        f"missing break-even rows for {anchor.name}/{workload}/"
                        f"b2pc={bandwidth:g}"
                    )
                output.append(
                    estimate_break_even(group, anchor.name, workload, bandwidth)
                )
    return output


def validate_anchor_invariance(
    rows: Sequence[dict[str, str | int | float]],
) -> float:
    grouped: dict[tuple[str, float, float], list[float]] = {}
    for row in rows:
        key = (
            str(row["workload"]),
            float(row["grid_vadd_group_cycles"]),
            float(row["grid_bank_to_pc_bandwidth_bytes_per_cycle_per_bank"]),
        )
        grouped.setdefault(key, []).append(float(row["local_absolute_savings_ns"]))
    maximum_error = max(max(values) - min(values) for values in grouped.values())
    if maximum_error > 1e-3:
        raise SensitivityError(
            "cached-KV anchor changed local/no-local absolute savings: "
            f"max_error_ns={maximum_error}"
        )
    return maximum_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validation-only local-combine break-even analysis."
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--expected-workloads", type=parse_workloads, default=DEFAULT_WORKLOADS
    )
    parser.add_argument("--expected-queries-per-workload", type=int, default=100)
    parser.add_argument(
        "--vadd-cycles",
        type=parse_positive_values,
        default=DEFAULT_VADD_CYCLES,
        help="comma-separated VADD cycles/group",
    )
    parser.add_argument(
        "--bank-to-pc-bandwidths",
        type=parse_positive_values,
        default=DEFAULT_BANK_TO_PC_BANDWIDTHS,
        help="comma-separated bytes/cycle/bank",
    )
    args = parser.parse_args()

    binary = args.binary.resolve()
    config = args.config.resolve()
    output_directory = args.output_directory.resolve()
    if not binary.is_file() or not config.is_file():
        raise SystemExit("--binary and --config must name existing files")
    if args.expected_queries_per_workload <= 0:
        raise SystemExit("--expected-queries-per-workload must be positive")
    if (
        ANCHOR_REFERENCE_POINT[0] not in args.vadd_cycles
        or ANCHOR_REFERENCE_POINT[1] not in args.bank_to_pc_bandwidths
    ):
        raise SystemExit(
            "grid must contain the anchor reference VADD=1 and bank-to-PC=32"
        )

    points = make_grid(args.vadd_cycles, args.bank_to_pc_bandwidths)
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "patch6_local_combine_grid_manifest.csv"
    write_manifest(manifest_path, points)

    overall_rows: list[dict[str, str | int | float]] = []
    workload_rows: list[dict[str, str | int | float]] = []
    try:
        for index, point in enumerate(points, start=1):
            print(f"[{index}/{len(points)}] {point.scenario.name}", flush=True)
            per_query, _ = run_scenario(
                binary,
                config,
                output_directory,
                point.scenario,
                args.expected_workloads,
                args.expected_queries_per_workload,
            )
            for workload in args.expected_workloads:
                local_rows = select(per_query, workload, LOCAL)
                no_local_rows = select(per_query, workload, NO_LOCAL)
                row = metric_row(point.scenario, workload, local_rows, no_local_rows)
                workload_rows.append(
                    add_break_even_fields(row, point, local_rows, no_local_rows)
                )
            local_rows = select(per_query, None, LOCAL)
            no_local_rows = select(per_query, None, NO_LOCAL)
            row = metric_row(point.scenario, "__all__", local_rows, no_local_rows)
            overall_rows.append(
                add_break_even_fields(row, point, local_rows, no_local_rows)
            )

        all_rows = [*workload_rows, *overall_rows]
        set_anchor_reference_changes(all_rows)
        anchor_invariance_error = validate_anchor_invariance(all_rows)
        break_even_rows = build_break_even_rows(
            all_rows, args.expected_workloads, args.bank_to_pc_bandwidths
        )
    except SensitivityError as error:
        raise SystemExit(f"ERROR: {error}") from error

    overall_path = output_directory / "patch6_local_combine_grid_overall.csv"
    workload_path = output_directory / "patch6_local_combine_grid_by_workload.csv"
    break_even_path = output_directory / "patch6_local_combine_break_even.csv"
    write_csv(overall_path, overall_rows)
    write_csv(workload_path, workload_rows)
    write_csv(break_even_path, break_even_rows)

    print("\nPatch 6 validation-only local-combine mean-latency break-even summary")
    print(
        "anchor b2pc_Bpc status break_even_vadd largest_win smallest_loss "
        "saving_at_min_us saving_at_max_us"
    )
    for row in break_even_rows:
        if row["workload"] != "__all__":
            continue
        threshold = row["break_even_vadd_group_cycles"]
        threshold_text = "-" if threshold == "" else f"{float(threshold):.4f}"
        print(
            f"{row['cached_kv_anchor']:<22} "
            f"{float(row['bank_to_pc_bandwidth_bytes_per_cycle_per_bank']):>9g} "
            f"{row['status']:<16} "
            f"{threshold_text:>15} "
            f"{str(row['largest_winning_tested_vadd']):>11} "
            f"{str(row['smallest_losing_tested_vadd']):>13} "
            f"{float(row['savings_at_minimum_vadd_ns']) / 1e3:>16.3f} "
            f"{float(row['savings_at_maximum_vadd_ns']) / 1e3:>16.3f}"
        )
    print(f"\nanchor_invariance_max_error_ns={anchor_invariance_error:.6f}")
    print(f"manifest_csv={manifest_path}")
    print(f"by_workload_csv={workload_path}")
    print(f"overall_csv={overall_path}")
    print(f"break_even_csv={break_even_path}")
    print("No test rows were simulated or reported; no PIM/H100 speedup was computed.")


if __name__ == "__main__":
    main()
