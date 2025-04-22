 # calc_kv_ratio_yaml – Quick Guide

**Purpose:** Calculate the correct `--gpu-memory-utilization` for vLLM so each model reserves a fixed amount of GPU RAM for its KV‑cache.

---

### 1 · Install
```bash
pip install vllm torch pyyaml
```

### 2 · `models.yml`
```yaml
models:
  - name: /path/to/model
    target_kv_gb: 5   # optional, defaults to --default_kv_gb
```

### 3 · Run
```bash
python calc_kv_ratio_yaml.py --yaml models.yml --default_kv_gb 5 --gpu 0
```

