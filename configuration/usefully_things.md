# Run model on 2 GPUs
```bash
vllm serve "mistralai/Mistral-Small-24B-Instruct-2501" --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
```

# Speed up HF download
Set to `True` for faster uploads and downloads from the Hub using hf_transfer.

By default, huggingface_hub uses the Python-based requests.get and requests.post functions. Although these are reliable and versatile, they may not be the most efficient choice for machines with high bandwidth. hf_transfer is a Rust-based package developed to maximize the bandwidth used by dividing large files into smaller parts and transferring them simultaneously using multiple threads. This approach can potentially double the transfer speed. To use hf_transfer:

1. Specify the hf_transfer extra when installing huggingface_hub (e.g. `pip install huggingface_hub[hf_transfer]`).
2. Set `HF_HUB_ENABLE_HF_TRANSFER=1` as an environment variable.


vllm serve \
  --model /home/llm/data/models/deepSeek-v2-lite-chat \
  --device cuda \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 2048 \
 --host 0.0.0.0 \ 
--port 8000 \
--gpu-memory-utilization 0.98 \
--quantization fp16 \


vllm serve /home/llm/data/models/qwen2-72b-instruct \    
  --device cuda \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 2048 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \        
  --gpu-memory-utilization 0.98 \
  --quantization moe_wna16 



# usefull flags vllm serve
1.  ```
    --max-num-seqs = 10 
    ```
  This falg set max batch size to 10.