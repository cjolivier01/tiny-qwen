# Qwen3-VL DDP Training

Distributed training infrastructure for Qwen3-VL models using PyTorch DDP, with support for Transformer Engine (FP8/FP4), Flash Attention 4, CUDA graphs, profiling, and Slurm multi-node launch.

## Files

| File | Description |
|------|-------------|
| `train_ddp.py` | Main DDP training script with all features |
| `benchmark.py` | PyTorch profiler benchmarks with timing and trace export |
| `launch_slurm.sh` | Slurm sbatch launcher (srun-based, no torchrun) |
| `model/model.py` | Model with optional FA4 (`flash_attn.cute`) attention path |

## Quick Start

### Local (torchrun)

```bash
# Random data smoke test (3 GPUs, skip GPU 0):
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --random-data --steps 20

# Real dataset (auto-downloads LLaVA-Instruct-150k):
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000

# With Flash Attention 4:
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --flash-attn-cute

# With Transformer Engine FP8:
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --te-fp8

# With FP4 (NVFP4):
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --te-fp4

# With CUDA graphs (via torch.compile):
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --cuda-graphs

# With profiling + TensorBoard:
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 50 --profile --tensorboard

# Kitchen sink:
CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --flash-attn-cute --te-fp8 --tensorboard
```

### Slurm (srun)

```bash
# 3 GPUs on 1 node:
./launch_slurm.sh 3 --steps 1000

# 8 GPUs across 2 nodes:
./launch_slurm.sh 8 --nodes 2 --random-data --steps 20

# Benchmark mode:
./launch_slurm.sh 4 --mode benchmark

# With training options:
./launch_slurm.sh 8 --nodes 2 --steps 1000 --te-fp8 --tensorboard

# Dry run (preview sbatch script):
DRY_RUN=1 ./launch_slurm.sh 8 --nodes 2
```

The Slurm launcher uses `srun` with one task per GPU (no torchrun). Each process reads `SLURM_PROCID` (global rank), `SLURM_LOCALID` (local rank/GPU index), and `SLURM_NTASKS` (world size) to initialize DDP.

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-name` | `Qwen/Qwen3-VL-8B-Instruct` | HuggingFace model name |
| `--random-data` | off | Use random tokens for smoke testing |
| `--steps` | 1000 | Training steps |
| `--batch-size` | 2 | Per-GPU batch size |
| `--seq-len` | 512 | Sequence length |
| `--lr` | 1e-5 | Peak learning rate |
| `--warmup-steps` | 50 | LR warmup steps (cosine schedule) |
| `--grad-accum` | 1 | Gradient accumulation steps |
| `--max-grad-norm` | 1.0 | Gradient clipping (0 to disable) |
| `--log-interval` | 10 | Log every N steps |
| `--save-interval` | 500 | Checkpoint every N steps (0 to disable) |
| `--te-fp8` | off | Transformer Engine FP8 |
| `--te-fp4` | off | Transformer Engine NVFP4 |
| `--te-fp8-recipe` | `current_scaling` | FP8 recipe: `delayed`, `current_scaling`, `mxfp8_block` |
| `--flash-attn-cute` | off | Flash Attention 4 (SM100/Blackwell) |
| `--cuda-graphs` | off | CUDA graphs via `torch.compile(mode="reduce-overhead")` |
| `--profile` | off | PyTorch profiler with TensorBoard trace export |
| `--tensorboard` | off | Log loss, grad norm, LR, tokens/sec, memory to TensorBoard |

## Performance Results

Tested on NVIDIA GB300 Max-Q (284 GB), Qwen3-VL-8B-Instruct, batch_size=2, seq_len=512.

### 3 GPUs (single node, torchrun)

| Mode | Step Time | Tokens/sec | Peak Memory |
|------|-----------|------------|-------------|
| BF16 (baseline) | ~283 ms | ~10,800 | 115.3 GB |
| Flash Attention 4 | ~225 ms | ~13,600 | 115.3 GB |
| TE FP8 (current_scaling) | ~360 ms | ~8,500 | 115.3 GB |
| TE NVFP4 | ~440 ms | ~7,000 | 115.3 GB |

### 8 GPUs (2 nodes, Slurm srun)

| Mode | Step Time | Tokens/sec | Peak Memory |
|------|-----------|------------|-------------|
| BF16 (baseline) | ~287 ms | ~28,500 | 115.3 GB |

### Real Dataset (LLaVA-Instruct-150k, 3 GPUs)

| Steps | Initial Loss | Final Loss | Duration |
|-------|-------------|------------|----------|
| 1000 | 1.83 | 0.89 | ~5 min |

## Features

### Flash Attention 4 (`flash_attn.cute`)

Uses CuTe-based flash attention kernels optimized for SM100/Blackwell GPUs. Enabled via `--flash-attn-cute`. The implementation:
- Transposes Q/K/V to `(B, T, H, D)` layout expected by FA4
- Handles GQA natively (no need for `repeat_interleave`)
- ~20% faster than SDPA baseline on GB300

### Transformer Engine FP8/FP4

Replaces `nn.Linear` and `RMSNorm` with Transformer Engine equivalents. Three FP8 recipes available:
- `current_scaling` (default) — `Float8CurrentScaling`
- `delayed` — `DelayedScaling` with history-based amax
- `mxfp8_block` — `MXFP8BlockScaling`
- FP4 uses `NVFP4BlockScaling`

Note: TE layer replacement adds overhead for this model size. Benefits are more pronounced at larger scales.

### CUDA Graphs

Implemented via `torch.compile(mode="reduce-overhead")`, which internally manages CUDA graph capture. Raw CUDA graph capture is incompatible with DDP allreduce hooks and autograd backward streams.

### Profiling

`--profile` generates PyTorch profiler traces viewable in TensorBoard:
```bash
tensorboard --logdir profiler_traces/
```

### TensorBoard Metrics

`--tensorboard` logs per-step metrics:
- `train/loss`, `train/lr`, `train/grad_norm`, `train/step_time_ms`
- `perf/tokens_per_sec_last20` (rolling average over last 20 batches)
- `perf/memory_gb`

```bash
tensorboard --logdir tb_logs/
```

### Slurm Multi-Node

The `launch_slurm.sh` script handles multi-node DDP via `srun`:
- One task per GPU, no torchrun wrapper
- Parses `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_NTASKS` for rank assignment
- `MASTER_ADDR` from `scontrol show hostnames`
- Supports configurable `--nodes` for multi-node scaling
- `DRY_RUN=1` to preview the sbatch script

### Dataset

Auto-downloads [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) (text-only conversations, ~158k samples) on first run. Uses the model's local `tokenizer.json` for tokenization.

## Output Directories

| Directory | Contents |
|-----------|----------|
| `checkpoints/` | Model checkpoints (every `--save-interval` steps) |
| `profiler_traces/` | PyTorch profiler TensorBoard traces |
| `tb_logs/` | TensorBoard training metrics |
| `slurm_logs/` | Slurm stdout/stderr logs |
| `data/` | Downloaded datasets |
