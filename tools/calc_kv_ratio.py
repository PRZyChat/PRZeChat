#!/usr/bin/env python3
"""
calc_kv_ratio_yaml.py – Compute the proper --gpu-memory-utilization value for one
or more models described in a YAML file, so that vLLM reserves a fixed amount of
GPU memory (in GB) for the KV‑cache.

How it works
------------
1.  The script takes *minimal* GPU memory for the KV‑cache (1 % of the free
    memory) and loads the model weights with vLLM.  This leaves the largest
    possible remainder of free VRAM that we can measure.
2.  It calculates what fraction of that remainder corresponds to the desired
    KV‑cache size (e.g. 5 GB) and prints a vLLM launch command with the correct
    `--gpu-memory-utilization` value.
3.  The process is repeated for every model entry in the YAML file.  The YAML
    can specify a custom KV‑cache size per model; otherwise the default from
    the command line is used.

YAML format
-----------
models:
  # Minimal example – target_kv_gb will be taken from --default_kv_gb
  - name: mistralai/Mistral-7B-Instruct-v0.3

  # Custom KV‑cache for this entry
  - name: TheBloke/Llama-3-8B-Instruct-GPTQ
    target_kv_gb: 4
"""
import argparse
import gc
import sys
from pathlib import Path

import torch
import yaml
from vllm import LLM

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _get_free_vram(device: torch.device) -> int:
    """Return free VRAM (in bytes) on *device* after all current allocations."""
    torch.cuda.synchronize(device)
    props = torch.cuda.get_device_properties(device)
    reserved = torch.cuda.memory_reserved(device)
    return props.total_memory - reserved


def _pretty_bytes(num_bytes: int) -> str:
    """Pretty‑print bytes as GiB with two decimals."""
    return f"{num_bytes / (1024 ** 3):.2f} GB"


# ---------------------------------------------------------------------------
# Main script logic
# ---------------------------------------------------------------------------

def process_model(model_name: str, target_kv_gb: float, gpu: int) -> None:
    """Load *model_name*, compute gpu‑memory‑utilization ratio for *target_kv_gb*."""
    device = torch.device(f"cuda:{gpu}")

    print("\n\u2699\ufe0f  Loading model", model_name, "...")

    # 1 % utilisation ≅ minimal KV‑cache; makes the measurement accurate
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.01,
        trust_remote_code=True,
        enforce_eager=True,  # ensure deterministic memory usage
    )

    free_after_weights = _get_free_vram(device)
    target_bytes = target_kv_gb * (1024 ** 3)

    if target_bytes >= free_after_weights:
        print(
            f"\u274c  ERROR: Only {_pretty_bytes(free_after_weights)} free after weights, "
            f"cannot reserve {_pretty_bytes(target_bytes)} for KV‑cache.",
            file=sys.stderr,
        )
        # Clean‑up before continuing to next model
        del llm
        torch.cuda.empty_cache()
        gc.collect()
        return

    util = round(target_bytes / free_after_weights, 4)

    print("\n\u1f4dd  Free VRAM after weights:", _pretty_bytes(free_after_weights))
    print("\ud83c\udf01  Requested KV‑cache:", _pretty_bytes(target_bytes))
    print("\n▶️  Launch vLLM with:\n")
    cmd = (
        "python -m vllm.entrypoints.api_server "
        f"--model {model_name} --gpu-memory-utilization {util}"
    )
    print(cmd)

    # Clean‑up GPU memory before the next iteration
    del llm
    torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute vLLM GPU memory ratios from YAML file.")
    parser.add_argument("--yaml", type=Path, default=Path("models.yml"), help="YAML file with model entries (default: models.yml)")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device ID (default: 0)")
    parser.add_argument("--default_kv_gb", type=float, default=5.0, help="Default KV‑cache size in GiB (used if entry omits target_kv_gb)")
    args = parser.parse_args()

    if not args.yaml.exists():
        sys.exit(f"YAML file {args.yaml} not found.")

    data = yaml.safe_load(args.yaml.read_text())
    models = data.get("models", [])
    if not models:
        sys.exit("No models defined under 'models' key in YAML.")

    for entry in models:
        name = entry.get("name")
        if not name:
            print("Skipping entry without 'name' field.", file=sys.stderr)
            continue
        kv_gb = float(entry.get("target_kv_gb", args.default_kv_gb))
        process_model(name, kv_gb, args.gpu)


if __name__ == "__main__":
    main()
