#!/usr/bin/env python3
"""kv_budget.py

Compute vLLM KV‑cache pre‑allocation so that the cache itself fits into a fixed
VRAM budget (e.g. 3 GB). The script reads a list of local model paths from a
YAML file, calculates how many cache blocks are needed for each model, and
saves the results (or any errors) back to another YAML file.  If a model cannot
be processed, it is skipped and the script continues with the next one.

Example usage
-------------
    python kv_budget.py models.yml --gb 3 --dtype bf16 -o kv_cache_results.yml

Dependencies
------------
    pip install transformers pyyaml
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import typing as T

import yaml
from transformers import AutoConfig

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DTYPE_BYTES: dict[str, int] = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_cfg(src: str | pathlib.Path) -> dict:
    """Load the *config.json* of a model without downloading the weights.

    `src` can be a Hugging Face repo‑ID, a local model directory or a direct
    path to a *config.json* file.
    """
    src = pathlib.Path(src)

    # Direct path to config.json → load as plain JSON
    if src.is_file() and src.name.endswith(".json"):
        return json.loads(src.read_text())

    # Otherwise let *transformers* locate & read the config
    return AutoConfig.from_pretrained(str(src)).to_dict()


def suggest_fixed_kv_cache(
    model_path: str | pathlib.Path,
    *,
    cache_gb: float = 3.0,
    dtype: str | None = None,
    block_size: int = 16,
) -> dict[str, T.Any]:
    """Return KV‑cache settings whose *total* VRAM ≦ `cache_gb`.

    The function calculates how many cache *blocks* are required so that the
    pre‑allocated buffer does **not** exceed the user‑defined memory budget. If
    the budget is too small to fit even a single block, a *ValueError* is
    raised.
    """
    cfg = _load_cfg(model_path)

    hidden_size: int = cfg["hidden_size"]
    num_layers: int = cfg["num_hidden_layers"]
    num_heads: int = cfg["num_attention_heads"]
    num_kv_heads: int = cfg.get("num_key_value_heads", num_heads)
    head_dim: int = hidden_size // num_heads

    # Determine dtype size (bytes per element)
    dtype = (dtype or cfg.get("torch_dtype", "float16")).lower()
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    bytes_per_element = DTYPE_BYTES[dtype]

    #  — size of one *token* inside the KV cache —
    bytes_per_token = (
        num_layers * num_kv_heads * head_dim * 2 * bytes_per_element  # K + V
    )
    bytes_per_block = block_size * bytes_per_token

    budget_bytes = cache_gb * 1024**3
    num_blocks = math.floor(budget_bytes / bytes_per_block)

    if num_blocks == 0:
        raise ValueError(
            f"Cache budget {cache_gb} GB is too small to fit even a single "
            f"block for model '{model_path}'."
        )

    actual_bytes = num_blocks * bytes_per_block
    actual_gb = actual_bytes / 1024**3
    max_tokens = num_blocks * block_size

    return {
        "num_gpu_blocks_override": int(num_blocks),
        "allocated_gb": round(actual_gb, 3),
        "max_model_len": int(max_tokens),  # informative, can be added to vLLM
        "vllm_flags": (
            f"--dtype {dtype} --block-size {block_size} "
            f"--num-gpu-blocks-override {num_blocks}"
        ),
    }


def load_model_list(yaml_file: str | pathlib.Path) -> list[str]:
    """Return the list of model paths from *models.yml*."""
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return [item["name"] for item in data.get("models", [])]


def save_results(results: list[dict[str, T.Any]], out_file: str | pathlib.Path) -> None:
    """Write results (including errors) to a YAML file."""
    with open(out_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(results, f, sort_keys=False)

# ──────────────────────────────────────────────────────────────────────────────
# Script entry‑point
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calculate a fixed‑size KV cache for each model in models.yml",
    )
    parser.add_argument("yaml", help="Path to models.yml")
    parser.add_argument("--gb", type=float, default=3.0,
                        help="Target VRAM budget for the cache (GB). Default: 3")
    parser.add_argument("--dtype", default=None,
                        help="Override dtype (e.g. bf16, fp16). By default taken from the model config.")
    parser.add_argument("--block-size", type=int, default=16,
                        help="vLLM block size. Default: 16")
    parser.add_argument("-o", "--output", default="kv_cache_results.yml",
                        help="Output YAML file. Default: kv_cache_results.yml")

    args = parser.parse_args(argv)

    # Read model list (ignore empty list gracefully)
    try:
        model_paths = load_model_list(args.yaml)
    except FileNotFoundError:
        sys.exit(f"models.yml not found: {args.yaml}")

    if not model_paths:
        sys.exit("No models defined in models.yml → nothing to do.")

    results: list[dict[str, T.Any]] = []

    for path in model_paths:
        entry: dict[str, T.Any] = {"model": path}
        try:
            info = suggest_fixed_kv_cache(
                path,
                cache_gb=args.gb,
                dtype=args.dtype,
                block_size=args.block_size,
            )
            entry.update(info)
            print(
                f"✔  {path}\n"
                f"   allocated_gb = {info['allocated_gb']}\n"
                f"   num_blocks   = {info['num_gpu_blocks_override']}"
            )
        except Exception as exc:
            entry["error"] = str(exc)
            print(f"✗  {path}  →  {exc}")
        finally:
            results.append(entry)

    save_results(results, args.output)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
