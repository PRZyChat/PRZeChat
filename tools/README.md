# KV Cache Budget Calculator

A tiny helper script that decides **how many KV‑cache blocks vLLM should pre‑allocate** so the cache itself fits into a fixed GPU‑memory budget (e.g. 3 GB).  
It reads a list of local models from `models.yml`, does **not** download the weights, and writes the result (or any error) to `kv_cache_results.yml`.

---

## Quick start

```bash
python kv_budget.py models.yml --gb 3 --dtype bf16 -o kv_cache_results.yml
```

* `--gb` – how many **GB of VRAM** to reserve for the cache (default 3).  
* `--dtype` – override dtype used for the cache (`bf16`, `fp16`, …).  
* `--block-size` – vLLM block size (default 16).

---

## What you get

`kv_cache_results.yml` collects one entry per model:

```yaml
- model: /path/to/model‑a
  allocated_gb: 2.99          # actual cache size
  num_gpu_blocks_override: 768
  max_model_len: 12288        # informational, can be passed to vLLM
  vllm_flags: "--dtype bf16 --block-size 16 --num-gpu-blocks-override 768"
- model: /path/to/model‑b
  error: "Cache budget 3 GB is too small to fit even a single block…"
```

Copy the `vllm_flags` string into your `vllm serve` command and the cache will stay within the chosen limit.

---

## How to use this information

Below is a minimal example showing how to launch vLLM for *model‑a* using the flags from the YAML:

```bash
vllm serve \
  --model /path/to/model‑a \
  --dtype bf16 \
  --block-size 16 \
  --num-gpu-blocks-override 768 \

```

*Replace the path and flags with the values listed for your model in* `kv_cache_results.yml`.  
If you have multiple GPUs, you can still add pipeline/tensor parallel flags– the cache reservation logic stays the same.

