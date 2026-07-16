#!/usr/bin/env python3
"""Generate the minimal formal GOFA trace fixture used by Patch 5 tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MEMORY_TOKENS = 128
HIDDEN_SIZE = 4096
ATTENTION_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SUFFIX_LAYERS = list(range(26, 32))
TEXT_LENGTHS = [3, 5, 7, 11]
SELECTED_ITEMS = [0, 3]
CACHE_KEYS = [
    "fixture-key-35",
    "fixture-key-1",
    "fixture-key-2",
    "fixture-key-40",
]


def quantized_bytes(shape: list[int], bits: int) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return (elements * bits + 7) // 8


def build_inventory() -> list[dict]:
    inventory = []
    for item_index, item_type in enumerate(("node", "node", "NOG", "edge")):
        text_length = TEXT_LENGTHS[item_index]
        inventory.append(
            {
                "item_index": item_index,
                "item_type": item_type,
                "cache_key": CACHE_KEYS[item_index],
                "cache_eligible": item_type != "NOG",
                "is_nog": item_type == "NOG",
                "sequence_length": MEMORY_TOKENS + text_length,
                "text_length": text_length,
                "memory_bits": 4,
                "key_bits": 2,
                "value_bits": 2,
                "memory_shape": [MEMORY_TOKENS, HIDDEN_SIZE],
                "text_kv_shapes": [
                    {
                        "layer_id": layer_id,
                        "key_shape": [KV_HEADS, text_length, HEAD_DIM],
                        "value_shape": [KV_HEADS, text_length, HEAD_DIM],
                    }
                    for layer_id in SUFFIX_LAYERS
                ],
            }
        )
    return inventory


def build_runtime_layers() -> list[dict]:
    item_order = [0, 2, 3]
    runtime_layers = []
    for layer_id in SUFFIX_LAYERS:
        items = []
        for runtime_index, item_index in enumerate(item_order):
            qk_tokens = MEMORY_TOKENS + TEXT_LENGTHS[item_index]
            items.append(
                {
                    "runtime_item_index": runtime_index,
                    "item_index": item_index,
                    "q_projection_input_shape": [1, MEMORY_TOKENS, HIDDEN_SIZE],
                    "q_projection_output_shape": [1, MEMORY_TOKENS, HIDDEN_SIZE],
                    "qk_shape": [1, ATTENTION_HEADS, MEMORY_TOKENS, qk_tokens],
                    "softmax_probability_shape": [
                        1,
                        ATTENTION_HEADS,
                        MEMORY_TOKENS,
                        qk_tokens,
                    ],
                    "pv_shape": [1, ATTENTION_HEADS, MEMORY_TOKENS, HEAD_DIM],
                    "attention_output_shape": [1, MEMORY_TOKENS, HIDDEN_SIZE],
                    "mlp_input_shape": [1, MEMORY_TOKENS, HIDDEN_SIZE],
                    "mlp_output_shape": [1, MEMORY_TOKENS, HIDDEN_SIZE],
                }
            )
        runtime_layers.append(
            {
                "layer_id": layer_id,
                "items": items,
                "gnn": {
                    "node_input_shape": [2, MEMORY_TOKENS, HIDDEN_SIZE],
                    "node_output_shape": [2, MEMORY_TOKENS, HIDDEN_SIZE],
                    "edge_input_shape": [2, MEMORY_TOKENS, HIDDEN_SIZE],
                    "edge_output_shape": None,
                    "edge_output_status": "not_produced_by_gofa_gnn_layer",
                },
            }
        )
    return runtime_layers


def traffic(inventory: list[dict]) -> dict:
    cacheable = [item for item in inventory if item["cache_eligible"]]
    memory_bytes = sum(
        quantized_bytes(item["memory_shape"], item["memory_bits"])
        for item in cacheable
    )
    full_key_bytes = sum(
        quantized_bytes(layer["key_shape"], item["key_bits"])
        for item in cacheable
        for layer in item["text_kv_shapes"]
    )
    full_value_bytes = sum(
        quantized_bytes(layer["value_shape"], item["value_bits"])
        for item in cacheable
        for layer in item["text_kv_shapes"]
    )
    selected_key_bytes = sum(
        quantized_bytes(inventory[item_index]["text_kv_shapes"][layer]["key_shape"], 2)
        for layer in range(len(SUFFIX_LAYERS))
        for item_index in SELECTED_ITEMS
    )
    selected_value_bytes = sum(
        quantized_bytes(
            inventory[item_index]["text_kv_shapes"][layer]["value_shape"], 2
        )
        for layer in range(len(SUFFIX_LAYERS))
        for item_index in SELECTED_ITEMS
    )
    edge = inventory[3]
    edge_cache_bytes = quantized_bytes(edge["memory_shape"], 4) + sum(
        quantized_bytes(layer[component], 2)
        for layer in edge["text_kv_shapes"]
        for component in ("key_shape", "value_shape")
    )
    return {
        "byte_accounting": "quantized_data_only_excluding_scale_and_container_metadata",
        "memory_cache_bytes": memory_bytes,
        "selected_key_bytes": selected_key_bytes,
        "selected_value_bytes": selected_value_bytes,
        "full_key_bytes": full_key_bytes,
        "full_value_bytes": full_value_bytes,
        "edge_cache_bytes": edge_cache_bytes,
        "nog_online_item_count": 1,
        "persistent_cache_bytes": memory_bytes + full_key_bytes + full_value_bytes,
        "runtime_loaded_cache_bytes": memory_bytes
        + selected_key_bytes
        + selected_value_bytes,
    }


def build_trace(query_id: str, split: str) -> dict:
    inventory = build_inventory()
    trace_traffic = traffic(inventory)
    by_layer = {str(layer_id): SELECTED_ITEMS for layer_id in SUFFIX_LAYERS}
    return {
        "trace_format": "gofa_query_trace",
        "trace_version": 1,
        "query_id": query_id,
        "created_at": "2026-07-16T00:00:00",
        "repository_commit_sha": "fixture",
        "task_name": "fixture_task",
        "dataset_name": "fixture_dataset",
        "split": split,
        "runtime_query_index": 0,
        "batch_size": 1,
        "cache_mode": "memory_kv",
        "cache_tag": "fixture",
        "model_configuration": {
            "hidden_size": HIDDEN_SIZE,
            "num_attention_heads": ATTENTION_HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "mem_size": MEMORY_TOKENS,
            "gnn_start_layer": SUFFIX_LAYERS[0],
            "suffix_layer_ids": SUFFIX_LAYERS,
        },
        "query_graph_structure": {
            "num_graph_nodes": 2,
            "num_node_text_items": 3,
            "num_edge_text_items": 1,
            "num_structural_edges": 2,
            "target_index": [0],
            "question_index": [1],
            "nog_local_index": 1,
            "nog_local_indices": [1],
            "node_map": [0, 2],
            "node_map_semantics": "graph_local_node_to_encoder_text_item",
            "edge_index": [[0, 1], [1, 0]],
            "edge_map": [0, 0],
            "edge_map_semantics": "structural_edge_to_edge_text_item",
            "batch": [0, 0],
            "ptr": [0, 2],
        },
        "cache_item_inventory": inventory,
        "selective_kv_access": {
            "policy_name": "target_1hop",
            "eligible_item_indices": [0, 1, 3],
            "selected_key_item_indices": SELECTED_ITEMS,
            "selected_value_item_indices": SELECTED_ITEMS,
            "complete_kv_item_indices": SELECTED_ITEMS,
            "k_only_item_indices": [],
            "v_only_item_indices": [],
            "skipped_item_indices": [1, 2],
            "key_item_mask": [True, False, False, True],
            "value_item_mask": [True, False, False, True],
            "effective_key_items_by_layer": by_layer,
            "effective_value_items_by_layer": by_layer,
            "selection_is_shared_across_suffix_layers": True,
        },
        "runtime_operation_shapes": {
            "item_order": [0, 2, 3],
            "layers": build_runtime_layers(),
        },
        "traffic_metadata": trace_traffic,
        "summary": {
            "total_item_count": 4,
            "cacheable_item_count": 3,
            "NOG_count": 1,
            "selection_ratio_basis": "cacheable_item_layer_accesses",
            "selected_K_ratio": 2 / 3,
            "selected_V_ratio": 2 / 3,
            "selected_KV_ratio": 2 / 3,
            "persistent_cache_bytes": trace_traffic["persistent_cache_bytes"],
            "runtime_loaded_cache_bytes": trace_traffic[
                "runtime_loaded_cache_bytes"
            ],
        },
    }


def distribution(value: int) -> dict:
    return {
        "min": value,
        "mean": value,
        "p50": float(value),
        "p95": float(value),
        "max": value,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("tests/fixtures/gofa_trace_v1"),
    )
    args = parser.parse_args()
    task_directory = args.root / "fixture_task_formal_v1"
    task_directory.mkdir(parents=True, exist_ok=True)

    traces = [
        build_trace("query_000000", "validation"),
        build_trace("query_000001", "test"),
    ]
    for trace in traces:
        write_json(task_directory / f"{trace['query_id']}.json", trace)

    index_lines = []
    for trace in traces:
        graph = trace["query_graph_structure"]
        summary = trace["summary"]
        access = trace["selective_kv_access"]
        index_lines.append(
            json.dumps(
                {
                    "query_id": trace["query_id"],
                    "task": trace["task_name"],
                    "split": trace["split"],
                    "trace_path": f"{trace['query_id']}.json",
                    "num_nodes": graph["num_graph_nodes"],
                    "num_edges": graph["num_structural_edges"],
                    "cacheable_items": summary["cacheable_item_count"],
                    "selected_K_items": len(
                        access["selected_key_item_indices"]
                    ),
                    "selected_V_items": len(
                        access["selected_value_item_indices"]
                    ),
                    "runtime_loaded_bytes": summary[
                        "runtime_loaded_cache_bytes"
                    ],
                },
                sort_keys=True,
            )
        )
    (task_directory / "trace_index.jsonl").write_text(
        "\n".join(index_lines) + "\n"
    )

    trace_traffic = traces[0]["traffic_metadata"]
    summary = {
        "trace_format": "gofa_query_trace_summary_v1",
        "tasks": {
            "fixture_task": {
                "query_count": 2,
                "node_count": distribution(2),
                "edge_count": distribution(2),
                "node_text_item_count": {"total": 6, "mean": 3},
                "edge_text_item_count": {"total": 2, "mean": 1},
                "selected_K_ratio": 2 / 3,
                "selected_V_ratio": 2 / 3,
                "selected_KV_ratio": 2 / 3,
                "persistent_cache_bytes": {
                    "total": 2 * trace_traffic["persistent_cache_bytes"],
                    "mean": trace_traffic["persistent_cache_bytes"],
                },
                "loaded_cache_bytes": {
                    "total": 2 * trace_traffic["runtime_loaded_cache_bytes"],
                    "mean": trace_traffic["runtime_loaded_cache_bytes"],
                },
                "NOG_online_count": {"total": 2, "mean": 1},
            }
        },
    }
    write_json(task_directory / "summary.json", summary)


if __name__ == "__main__":
    main()
