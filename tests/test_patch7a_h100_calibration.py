#!/usr/bin/env python3
"""End-to-end and strict negative tests for Patch 7A H100 calibration."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests/fixtures/gofa_trace_v1"
CONFIG_TEMPLATE = REPOSITORY_ROOT / "tests/analytical_pim_trace_fixture_config.yaml"
CALIBRATOR = REPOSITORY_ROOT / "tools/run_patch7a_h100_calibration.py"

TARGET = {
    "h100_group_cycles": 2.0,
    "h100_cache_bytes_per_cycle": 1024.0,
    "h100_int2_unpack_group_cycles": 0.5,
    "h100_scale_dequant_group_cycles": 0.75,
    "h100_layout_conversion_group_cycles": 0.25,
    "h100_irregular_gather_penalty": 1.5,
    "h100_small_batch_efficiency": 0.8,
    "h100_fixed_overhead_cycles": 100.0,
}


def write_config(directory: Path) -> Path:
    text = CONFIG_TEMPLATE.read_text()
    text = text.replace(
        "root: tests/fixtures/gofa_trace_v1", f"root: {FIXTURE_ROOT}"
    )
    text = text.replace(
        "splits: [validation, test]",
        "splits: [validation, test]\n  simulation_splits: [validation]",
    )
    text = text.replace(
        "graph_compute_placement_sweep: [hash, hybrid_locality_balanced]",
        "graph_compute_placement_sweep: [source_dst_locality]",
    )
    text = text.replace(
        "kv_storage_placement_sweep: [hash, balanced]",
        "kv_storage_placement_sweep: [balanced]",
    )
    text = text.replace(
        "baselines:\n"
        "  - pim_selective_kv_no_local_combine\n"
        "  - pim_selective_kv_local_combine",
        "baselines:\n  - h100_realistic_cache_path",
    )
    text = text.replace("memory_token_tiles: [1, 4]", "memory_token_tiles: [4]")
    path = directory / "patch7a_fixture_config.yaml"
    path.write_text(text)
    return path


def run_calibrator(
    binary: Path, config: Path, output: Path, profile: Path | None = None
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(CALIBRATOR),
        "--binary",
        str(binary),
        "--config",
        str(config),
        "--output-directory",
        str(output),
        "--expected-workloads",
        "fixture_task",
        "--expected-queries-per-workload",
        "1",
    ]
    if profile is not None:
        command.extend(("--profile", str(profile)))
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def populate_profile(template: Path, profile: Path) -> None:
    rows = read_rows(template)
    for row in rows:
        row.update(
            {
                "device_name": "NVIDIA H100 80GB HBM3",
                "device_uuid": "GPU-fixture-h100",
                "cuda_version": "12.4",
                "pytorch_version": "2.5.1",
                "profile_run_id": "fixture-run-001",
                "warmup_iterations": "10",
                "measured_iterations": "100",
            }
        )
        score = float(row["h100_gnn_score_groups"]) * TARGET["h100_group_cycles"]
        message = (
            float(row["h100_gnn_message_groups"]) * TARGET["h100_group_cycles"]
        )
        cached = (
            float(row["h100_cached_kv_groups"]) * TARGET["h100_group_cycles"]
        )
        cache_read = (
            float(row["runtime_loaded_total_bytes"])
            / TARGET["h100_cache_bytes_per_cycle"]
        )
        groups = float(row["h100_cached_kv_groups"])
        unpack = groups * TARGET["h100_int2_unpack_group_cycles"]
        scale = groups * TARGET["h100_scale_dequant_group_cycles"]
        layout = groups * TARGET["h100_layout_conversion_group_cycles"]
        irregular = cache_read * (TARGET["h100_irregular_gather_penalty"] - 1.0)
        small = (cache_read + irregular) * (
            1.0 / TARGET["h100_small_batch_efficiency"] - 1.0
        )
        fixed = TARGET["h100_fixed_overhead_cycles"]
        stages = {
            "gnn_score_compute_ns": score,
            "gnn_message_compute_ns": message,
            "cached_kv_compute_ns": cached,
            "cache_read_ns": cache_read,
            "int2_unpack_ns": unpack,
            "scale_dequant_ns": scale,
            "layout_conversion_ns": layout,
            "irregular_gather_ns": irregular,
            "small_batch_penalty_ns": small,
            "fixed_overhead_ns": fixed,
        }
        for name, value in stages.items():
            row[name] = f"{value:.12f}"
        row["measured_total_latency_ns"] = f"{sum(stages.values()):.12f}"
    write_rows(profile, rows)


def assert_close(actual: float, expected: float) -> None:
    assert abs(actual - expected) <= max(1e-8, abs(expected) * 1e-8), (
        actual,
        expected,
    )


def expect_failure(
    binary: Path,
    config: Path,
    valid_profile: Path,
    directory: Path,
    name: str,
    mutation: Callable[[list[dict[str, str]]], None],
    message: str,
) -> None:
    rows = read_rows(valid_profile)
    mutation(rows)
    profile = directory / f"invalid_{name}.csv"
    write_rows(profile, rows)
    completed = run_calibrator(
        binary, config, directory / f"invalid_{name}_output", profile
    )
    assert completed.returncode != 0, completed.stdout
    assert message in completed.stdout, completed.stdout


def check_patch7a(binary: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="patch7a-") as temporary:
        directory = Path(temporary)
        config = write_config(directory)
        template_output = directory / "template_output"
        template_run = run_calibrator(binary, config, template_output)
        assert template_run.returncode == 0, template_run.stdout
        assert "validation_queries=1" in template_run.stdout
        assert "No test rows or PIM/H100 speedup" in template_run.stdout
        template = template_output / "h100_profile_template.csv"
        assert template.is_file()

        profile = directory / "valid_profile.csv"
        populate_profile(template, profile)
        calibrated_output = directory / "calibrated"
        completed = run_calibrator(binary, config, calibrated_output, profile)
        assert completed.returncode == 0, completed.stdout
        assert "Patch 7A H100 validation calibration passed" in completed.stdout
        assert "test was not read and speedup was not computed" in completed.stdout

        manifest = {
            row["parameter"].removeprefix("near_bank_pe."): float(row["value"])
            for row in read_rows(
                calibrated_output / "patch7a_h100_parameter_manifest.csv"
            )
        }
        assert set(manifest) == set(TARGET)
        for name, expected in TARGET.items():
            assert_close(manifest[name], expected)
        overall = read_rows(
            calibrated_output / "patch7a_h100_validation_overall.csv"
        )[0]
        assert_close(float(overall["mape"]), 0.0)
        assert_close(float(overall["rmse_ns"]), 0.0)
        residual = read_rows(
            calibrated_output / "patch7a_h100_validation_per_query.csv"
        )[0]
        assert residual["split"] == "validation"
        assert_close(float(residual["residual_ns"]), 0.0)

        cases: tuple[
            tuple[str, Callable[[list[dict[str, str]]], None], str], ...
        ] = (
            (
                "test_split",
                lambda rows: rows[0].__setitem__("split", "test"),
                "test profile rows are forbidden",
            ),
            (
                "duplicate",
                lambda rows: rows.append(dict(rows[0])),
                "duplicate query identity",
            ),
            (
                "identity",
                lambda rows: rows[0].__setitem__("trace_query_id", "query_wrong"),
                "profile query not found",
            ),
            (
                "feature",
                lambda rows: rows[0].__setitem__("selected_kv_bytes", "1"),
                "trace feature mismatch",
            ),
            (
                "negative",
                lambda rows: rows[0].__setitem__("cache_read_ns", "-1"),
                "timing values must be non-negative",
            ),
            (
                "stage_sum",
                lambda rows: rows[0].__setitem__("measured_total_latency_ns", "1"),
                "stage sum mismatch",
            ),
            (
                "batch",
                lambda rows: rows[0].__setitem__("batch_size", "2"),
                "batch_size must be 1",
            ),
        )
        for name, mutation, message in cases:
            expect_failure(
                binary, config, profile, directory, name, mutation, message
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"Missing analytical_pim binary: {binary}")
    check_patch7a(binary)
    print("Patch 7A H100 calibration tests passed: valid=2, negative=7")


if __name__ == "__main__":
    main()
