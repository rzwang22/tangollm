# Analytical PIM Cost Model

The analytical simulator can optionally estimate the incremental area, energy,
average power, and a structural peak-power upper bound of the HBM-PIM logic.
The model is enabled with `cost_model.enabled: true`.

The strict HBM2E 8-Hi configuration uses the following architectural choices:

- 28 nm logic technology
- 400 MHz PE/reducer/fabric clock (`clock_ns: 2.5`)
- one near-bank PE and one private 1 KiB local buffer per addressable bank
- 512 banks per stack and five stacks, for 2,560 PEs and 2.5 MiB of local
  buffer capacity
- one pseudo-channel reducer per pCH and one global reducer per stack

## Outputs

When the model is enabled, the simulator writes three additional files:

- `hardware_cost_csv`: one row describing provisioned resources, incremental
  area, and leakage-power bounds
- `cost_per_query_csv`: event counts and energy/power estimates for every PIM
  query result
- `cost_aggregate_csv`: mean, p50, and p95 cost metrics grouped by workload,
  split, placement, baseline, and tile size

The latency and traffic CSVs retain their existing data-only byte columns and
add separate scale, gather-index, and total-byte columns.

## Accounting

Incremental area is the sum of near-bank PE logic, private local buffers,
pseudo-channel reducers, global reducers, and per-stack router logic. The model
reports total area, area per stack, near-bank area per DRAM die, and buffer-die
logic area per stack.

Dynamic energy is computed from simulator event counts:

```text
E_dynamic = E_Q8K8 + E_P8V8 + E_Q8K2 + E_P8V2 + E_scale + E_VADD
          + E_local_buffer + E_HBM_read + E_communication + E_reducer
```

The HBM term uses trace-native `runtime_loaded_total_bytes`. Legacy traces
provide quantized data only. Traces using the separate metadata schema add FP32
scale bytes and UINT32 gather-index bytes to the same runtime read total. The
term does not include xPU model-weight traffic or unspecified graph-state DRAM
traffic. The communication term includes Q broadcast, bank-to-pCH,
pCH-to-global, and global-to-NPU traffic.

Leakage energy uses the query latency and a configurable inactive-unit leakage
factor. The CSV also reports the all-units-on leakage bound. Average power is
`total_energy / latency`. Peak dynamic power is a structural upper bound based
on the maximum concurrently active banks, pCHs, stacks, reducer lanes, and link
bandwidth for any modeled stage; it is not derived from average utilization.

## Calibration Status

`provisional_28nm_v1` coefficients are first-order assumptions that make the
accounting executable and support sensitivity analysis. They are not results
from RTL synthesis, CACTI/SRAM compiler, DRAMPower, or signoff power analysis.
Paper-final absolute area and power claims require replacing the configurable
coefficients with tool or measurement reports. The event accounting and CSV
schema do not need to change when those calibrated values become available.
