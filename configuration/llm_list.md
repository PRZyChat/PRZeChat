# Modele LLM

**Wzór:**  
`VRAM [GB] = (Parametry * Bajty na parametr * 1.2) / 1e9`  
_(1.2 = 20% zapasu)_

| Model                               | Parametry | BF16 (2B)    | int8 (1B)   | int6 (0.75B) | int4 (0.5B) | AWQ (4-bit) | Architektura       | Licencja         |
| ----------------------------------- | --------- | ------------ | ----------- | ------------ | ----------- | ----------- | ------------------ | ---------------- |
| **deepseek-v2-lite-chat**           | 16B       | 38.4 GB      | 19.2 GB     | 14.4 GB      | 9.6 GB      | ~9.6 GB     | MoE (Hybrydowa)    | DeepSeek License |
| **mistral-small-24b-instruct-2501** | 24B       | 57.6 GB      | 28.8 GB     | 21.6 GB      | 14.4 GB     | ~14.4 GB     | Transformer        | Apache 2.0       |
| **qwen2.5-32b-instruct**            | 32B       | 76.8 GB      | 38.4 GB     | 28.8 GB      | 19.2 GB     | ~19.2 GB     | Qwen Attention     | Tongyi Qianwen   |
| **llama3-70b-instruct**             | 70B       | ~~168.0 GB~~ | ~~84.0 GB~~ | 63.0 GB      | 42.0 GB     | ~42.0 GB     | Transformer (RoPE) | Meta License     |
| **qwen2-72b-instruct**              | 72B       | ~~172.8 GB~~ | ~~86.4 GB~~ | 64.8 GB      | 43.2 GB     | ~43.2 GB     | Qwen-Transformer   | Tongyi Qianwen   |

---

## Legenda:

- **~~Przekreślone~~** – przekroczenie limitu **80 GB VRAM**

---

### Kwantyzacja metodą AWQ

Nie kwantyzuje wszystkich wag w modelu, lecz zachowuje niewielki ich procent kluczowy dla wydajności LLM. Znacząco redukuje to utratę jakości przy kwantyzacji, **umożliwiając uruchamianie modeli w precyzji 4-bitowej bez pogorszenia ich działania**.

<br>

**NOTE**

Modele znajdują się w **~/data/models/**

---

## Komendy do uruchamienia modeli bez kwantyzacji:

### 1. deepSeek-v2-lite-chat

```bash
vllm serve ~/data/models/deepSeek-v2-lite-chat \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

### 2. mistral-small-24b-instruct-2501

```bash
vllm serve ~/data/models/mistral-small-24b-instruct-2501 \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000
```

### 3. qwen2.5-32b-instruct

```bash
vllm serve ~/data/models/qwen2.5-32b-instruct \
  --tensor-parallel-size 2 \
  --host 0.0.0.0 \
  --port 8000 \
```

## Komendy do uruchamienia modeli z kwantyzacją AWQ:

### 1. llama3-70b-instruct
```bash
vllm serve ~/data/models/llama3-70b-instruct-awq \
    --quantization awq \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

### 2. qwen2-72b-instruct
```bash
vllm serve ~/data/models/qwen2-72b-instruct-awq \
    --quantization awq \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

