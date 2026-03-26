"""
PyTorch profiling benchmarks for Qwen3-VL-8B training.

Runs a configurable set of training steps with torch.profiler and generates
TensorBoard traces, summary tables, and optional Chrome JSON traces.

Usage:
  # Basic benchmark on GPUs 1,2,3:
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 benchmark.py

  # Benchmark with TE FP8:
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 benchmark.py --te-fp8

  # Benchmark with FP4:
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 benchmark.py --te-fp4

  # Single GPU quick test:
  CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 benchmark.py --steps 10
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.model import Qwen3VL, ModelConfig, RMSNorm


def parse_args():
    p = argparse.ArgumentParser(description="Profiling benchmarks for Qwen3-VL")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--output-dir", default="profiler_traces")

    p.add_argument("--te-fp8", action="store_true")
    p.add_argument("--te-fp4", action="store_true")
    p.add_argument("--te-fp8-recipe", default="current_scaling")
    p.add_argument("--cuda-graphs", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Support both torchrun and native srun launch
    from train_ddp import setup_distributed_from_slurm
    setup_distributed_from_slurm()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Benchmark config: batch_size={args.batch_size}, seq_len={args.seq_len}, "
              f"steps={args.steps}, world_size={world_size}")
        print(f"TE FP8: {args.te_fp8}, TE FP4: {args.te_fp4}, CUDA Graphs: {args.cuda_graphs}")

    # Import from train_ddp
    from train_ddp import load_model_for_training, replace_with_te_layers, get_te_recipe

    model, config, _ = load_model_for_training(args, local_rank)

    te_context_fn = nullcontext
    if args.te_fp8 or args.te_fp4:
        import transformer_engine.pytorch as te
        replace_with_te_layers(model, use_fp8=args.te_fp8, use_fp4=args.te_fp4)
        recipe = get_te_recipe(args.te_fp8_recipe, use_fp4=args.te_fp4)
        te_context_fn = lambda: te.fp8_autocast(enabled=True, fp8_recipe=recipe)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01, fused=True)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Generate random data
    input_ids = torch.randint(0, config.n_vocab, (args.batch_size, args.seq_len), device=device)
    labels = input_ids.clone()
    labels[:, 0] = -100

    # Warmup
    if rank == 0:
        print(f"\nWarming up ({args.warmup_steps} steps)...")
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        with te_context_fn():
            logits = model(input_ids=input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # ─── Timing benchmark ───
    if rank == 0:
        print(f"\nRunning timing benchmark ({args.steps} steps)...")

    torch.cuda.reset_peak_memory_stats(device)
    start_events = []
    end_events = []

    for i in range(args.steps):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        optimizer.zero_grad(set_to_none=True)
        with te_context_fn():
            logits = model(input_ids=input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()
        end_event.record()

        start_events.append(start_event)
        end_events.append(end_event)

    torch.cuda.synchronize()

    step_times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9

    if rank == 0:
        avg_ms = sum(step_times_ms) / len(step_times_ms)
        min_ms = min(step_times_ms)
        max_ms = max(step_times_ms)
        p50 = sorted(step_times_ms)[len(step_times_ms) // 2]
        p90 = sorted(step_times_ms)[int(len(step_times_ms) * 0.9)]
        p99 = sorted(step_times_ms)[int(len(step_times_ms) * 0.99)]
        tokens_per_step = args.batch_size * args.seq_len * world_size
        tokens_per_sec = tokens_per_step / (avg_ms / 1000)

        total_params = sum(p.numel() for p in model.parameters())
        # Rough FLOPS estimate: 6 * params * tokens (forward + backward)
        flops_per_step = 6 * total_params * args.batch_size * args.seq_len
        tflops = flops_per_step / (avg_ms / 1000) / 1e12

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)
        mode = "BF16"
        if args.te_fp8:
            mode = f"TE FP8 ({args.te_fp8_recipe})"
        elif args.te_fp4:
            mode = "TE NVFP4"
        print(f"  Mode:           {mode}")
        print(f"  World size:     {world_size}")
        print(f"  Batch size:     {args.batch_size} x {world_size} = {args.batch_size * world_size}")
        print(f"  Seq length:     {args.seq_len}")
        print(f"  Parameters:     {total_params/1e9:.2f}B")
        print(f"  Peak memory:    {peak_mem_gb:.1f} GB")
        print(f"  Avg step time:  {avg_ms:.1f} ms")
        print(f"  Min step time:  {min_ms:.1f} ms")
        print(f"  Max step time:  {max_ms:.1f} ms")
        print(f"  P50:            {p50:.1f} ms")
        print(f"  P90:            {p90:.1f} ms")
        print(f"  P99:            {p99:.1f} ms")
        print(f"  Tokens/sec:     {tokens_per_sec:.0f}")
        print(f"  Est. TFLOPS:    {tflops:.1f}")
        print("=" * 80)

    # ─── Profiler trace (all ranks run the loop; only rank 0 collects traces) ───
    if rank == 0:
        print(f"\nRunning profiler ({args.steps} steps)...")
        os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        prof_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=args.steps - 3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        )
    else:
        prof_ctx = nullcontext()

    with prof_ctx as prof:
        for i in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            with te_context_fn():
                logits = model(input_ids=input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            if prof is not None:
                prof.step()

    if rank == 0 and prof is not None:
        print("\nTop 20 CUDA kernel time consumers:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # Save summary
        summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
        summary = {
            "mode": mode,
            "world_size": world_size,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "parameters_B": round(total_params / 1e9, 2),
            "peak_memory_GB": round(peak_mem_gb, 1),
            "avg_step_ms": round(avg_ms, 1),
            "min_step_ms": round(min_ms, 1),
            "max_step_ms": round(max_ms, 1),
            "p50_ms": round(p50, 1),
            "p90_ms": round(p90, 1),
            "p99_ms": round(p99, 1),
            "tokens_per_sec": round(tokens_per_sec),
            "est_tflops": round(tflops, 1),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
        print(f"TensorBoard traces saved to {args.output_dir}/")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
