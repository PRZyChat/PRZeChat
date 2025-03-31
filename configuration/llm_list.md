# Obliczenie zapotrzebowania VRAM dla modeli LLM

**Wzór:**  
`VRAM [GB] = (Parametry * Bajty na parametr * 1.2) / 1e9`  
_(1.2 = 20% zapasu)_

| Model                               | Parametry | BF16 (2B)                                   | int8 (1B)                                  | int6 (0.75B)                               | int4 (0.5B)                                | Architektura       | Licencja         |
| ----------------------------------- | --------- | ------------------------------------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------ | ---------------- |
| **deepseek-v2-lite-chat**           | 16B       | <span style="color:#90EE90">38.4 GB</span>  | <span style="color:#90EE90">19.2 GB</span> | <span style="color:#90EE90">14.4 GB</span> | <span style="color:#90EE90">9.6 GB</span>  | MoE (Hybrydowa)    | DeepSeek License |
| **mistral-small-24b-instruct-2501** | 24B       | <span style="color:#90EE90">57.6 GB</span>  | <span style="color:#90EE90">28.8 GB</span> | <span style="color:#90EE90">21.6 GB</span> | <span style="color:#90EE90">14.4 GB</span> | Transformer        | Apache 2.0       |
| **qwen2.5-32b-instruct**            | 32B       | <span style="color:#90EE90">76.8 GB</span>  | <span style="color:#90EE90">38.4 GB</span> | <span style="color:#90EE90">28.8 GB</span> | <span style="color:#90EE90">19.2 GB</span> | Qwen Attention     | Tongyi Qianwen   |
| **llama3-70b-instruct**             | 70B       | <span style="color:#FF6B6B">168.0 GB</span> | <span style="color:#FF6B6B">84.0 GB</span> | <span style="color:#90EE90">63.0 GB</span> | <span style="color:#90EE90">42.0 GB</span> | Transformer (RoPE) | Meta License     |
| **qwen2-72b-instruct**              | 72B       | <span style="color:#FF6B6B">172.8 GB</span> | <span style="color:#FF6B6B">86.4 GB</span> | <span style="color:#90EE90">64.8 GB</span> | <span style="color:#90EE90">43.2 GB</span> | Qwen-Transformer   | Tongyi Qianwen   |

---

### Kwantyzacja metodą AWQ

Nie kwantyzuje wszystkich wag w modelu, lecz zachowuje niewielki ich procent kluczowy dla wydajności LLM. Znacząco redukuje to utratę jakości przy kwantyzacji, **umożliwiając uruchamianie modeli w precyzji 4-bitowej bez pogorszenia ich działania**.

## Legenda:

- <span style="color:#FF6B6B">Czerwony</span> – przekroczenie limitu **80 GB VRAM**
- <span style="color:#90EE90">Zielony</span> – mieści się w limicie

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
