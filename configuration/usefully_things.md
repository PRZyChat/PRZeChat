# Run model on 2 GPUs
```bash
vllm serve "mistralai/Mistral-Small-24B-Instruct-2501" --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
```