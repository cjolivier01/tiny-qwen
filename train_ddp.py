"""
DDP training script for Qwen3-VL-8B-Instruct.

Usage:
  # Random data smoke test (3 GPUs):
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --random-data --steps 20

  # Real dataset training (1000 steps):
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000

  # With Transformer Engine FP8:
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --te-fp8

  # With FP4 (NVFP4):
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --te-fp4

  # With profiling:
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 50 --profile

  # With CUDA graphs:
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --cuda-graphs

  # With TensorBoard logging:
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --tensorboard

  # Kitchen sink:
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 train_ddp.py --steps 1000 --te-fp8 --profile --tensorboard
"""

import os
import sys
import json
import argparse
import math
import time
from pathlib import Path
from contextlib import nullcontext
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from model.model import Qwen3VL, ModelConfig, Block, RMSNorm, DenseMLP, SelfAttention, enable_flash_attn_cute
from model.vision import VisionConfig

# ──────────────────────────── Argument parsing ────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DDP training for Qwen3-VL-8B")
    p.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--random-data", action="store_true", help="Use random data for smoke testing")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--grad-accum", type=int, default=3, help="Gradient accumulation steps (forward passes per optimizer step)")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--eval-interval", type=int, default=100, help="Eval on held-out batch every N steps (0 to disable)")
    p.add_argument("--is-ratio-interval", type=int, default=100, help="IS ratio eval every N steps (0 to disable)")
    p.add_argument("--is-ratio-tokens", type=int, default=128, help="Tokens to generate for IS ratio eval")
    p.add_argument("--is-ratio-temperature", type=float, default=1.0, help="Sampling temperature for IS ratio eval")
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--dataset-dir", default="data/llava_instruct_150k")

    # TE / precision
    p.add_argument("--te-fp8", action="store_true", help="Use Transformer Engine FP8")
    p.add_argument("--te-fp4", action="store_true", help="Use Transformer Engine NVFP4")
    p.add_argument("--te-fp8-recipe", default="current_scaling",
                   choices=["delayed", "current_scaling", "mxfp8_block"],
                   help="FP8 scaling recipe")

    # Flash Attention
    p.add_argument("--flash-attn", action="store_true", help="Use Flash Attention (already used via SDPA)")
    p.add_argument("--flash-attn-cute", action="store_true", help="Use Flash Attention 4 (flash_attn.cute) for SM100/Blackwell")

    # CUDA Graphs
    p.add_argument("--cuda-graphs", action="store_true", help="Enable CUDA graph capture for forward/backward")

    # Profiling
    p.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
    p.add_argument("--profile-dir", default="profiler_traces")
    p.add_argument("--profile-start-step", type=int, default=5)
    p.add_argument("--profile-end-step", type=int, default=15)

    # TensorBoard
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--tb-dir", default="tb_logs", help="TensorBoard log directory")

    return p.parse_args()


# ──────────────────────────── Datasets ────────────────────────────

class RandomTextDataset(Dataset):
    """Random token dataset for DDP smoke testing."""
    def __init__(self, vocab_size, seq_len, size=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = input_ids.clone()
        labels[0] = -100  # mask first token
        return {"input_ids": input_ids, "labels": labels}


class LLaVAInstructDataset(Dataset):
    """LLaVA-Instruct-150k text-only dataset for language model fine-tuning."""
    def __init__(self, data_path, tokenizer, seq_len=512, max_samples=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        with open(data_path, "r") as f:
            raw_data = json.load(f)

        self.samples = []
        for item in raw_data:
            if "image" in item:
                continue
            convs = item.get("conversations", [])
            if len(convs) >= 2:
                self.samples.append(convs)
            if max_samples and len(self.samples) >= max_samples:
                break

        # If not enough text-only, include image ones but ignore the image
        if len(self.samples) < 1000:
            for item in raw_data:
                if "image" not in item:
                    continue
                convs = item.get("conversations", [])
                if len(convs) >= 2:
                    self.samples.append(convs)
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        convs = self.samples[idx]
        text = ""
        for turn in convs:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human":
                text += f"<|im_start|>user\n{value}<|im_end|>\n"
            elif role == "gpt":
                text += f"<|im_start|>assistant\n{value}<|im_end|>\n"

        token_ids = self.tokenizer.encode(text).ids
        if len(token_ids) > self.seq_len:
            token_ids = token_ids[:self.seq_len]
        else:
            token_ids = token_ids + [0] * (self.seq_len - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == 0] = -100
        return {"input_ids": input_ids, "labels": labels}


# ──────────────────────────── TE Integration ────────────────────────────

def replace_with_te_layers(model, use_fp8=False, use_fp4=False):
    """Replace linear layers and RMSNorm with Transformer Engine equivalents."""
    import transformer_engine.pytorch as te

    replaced = 0
    for name, module in list(model.named_modules()):
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1] if name else ""
        if not child_name:
            continue

        parent = model
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)

        if isinstance(module, RMSNorm) and not isinstance(module, te.RMSNorm):
            te_norm = te.RMSNorm(
                module.weight.shape[0],
                eps=module.variance_epsilon,
            )
            te_norm.weight = module.weight
            setattr(parent, child_name, te_norm)
            replaced += 1

        elif isinstance(module, nn.Linear) and not isinstance(module, te.Linear):
            te_linear = te.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            te_linear.weight = module.weight
            if module.bias is not None:
                te_linear.bias = module.bias
            setattr(parent, child_name, te_linear)
            replaced += 1

    return replaced


def get_te_recipe(recipe_name, use_fp4=False):
    """Get Transformer Engine FP8/FP4 recipe."""
    if use_fp4:
        from transformer_engine.common.recipe import NVFP4BlockScaling
        return NVFP4BlockScaling()

    from transformer_engine.common.recipe import (
        DelayedScaling,
        Float8CurrentScaling,
        MXFP8BlockScaling,
    )

    if recipe_name == "delayed":
        return DelayedScaling()
    elif recipe_name == "current_scaling":
        return Float8CurrentScaling()
    elif recipe_name == "mxfp8_block":
        return MXFP8BlockScaling()
    else:
        return Float8CurrentScaling()


# ──────────────────────────── LR Scheduler ────────────────────────────

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=1e-7):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ──────────────────────────── CUDA Graph Wrapper ────────────────────────────

def apply_cuda_graph_compile(model, rank):
    """Apply torch.compile with reduce-overhead mode for CUDA graph benefits.

    This is the modern PyTorch approach - torch.compile with reduce-overhead
    mode internally uses CUDA graphs where possible, handling the complexity
    of graph capture/replay automatically.
    """
    if rank == 0:
        print("Applying torch.compile(mode='reduce-overhead') for CUDA graph optimization...")
    model = torch.compile(model, mode="reduce-overhead")
    return model


# ──────────────────────────── Profiler ────────────────────────────

def create_profiler(args, rank):
    if not args.profile or rank != 0:
        return nullcontext()

    os.makedirs(args.profile_dir, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=1,
        warmup=2,
        active=args.profile_end_step - args.profile_start_step,
        repeat=1,
        skip_first=args.profile_start_step,
    )

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )


# ──────────────────────────── TensorBoard Logger ────────────────────────────

class TBLogger:
    """Thin wrapper around TensorBoard SummaryWriter, no-op if disabled."""
    def __init__(self, enabled, log_dir, rank):
        self.writer = None
        if enabled and rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

    def scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        if self.writer:
            self.writer.close()


# ──────────────────────────── Model loading ────────────────────────────

def load_model_for_training(args, local_rank):
    """Load model weights and prepare for training on a single device."""
    from huggingface_hub import snapshot_download

    device = torch.device(f"cuda:{local_rank}")
    model_name = args.model_name

    rank = dist.get_rank()
    if rank == 0:
        try:
            weights_path = snapshot_download(repo_id=model_name, cache_dir=".cache", local_files_only=True)
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)
            weights_path = snapshot_download(repo_id=model_name, cache_dir=".cache")
        path_bytes = weights_path.encode()
        path_len = torch.tensor([len(path_bytes)], dtype=torch.long, device=device)
    else:
        path_len = torch.tensor([0], dtype=torch.long, device=device)

    dist.broadcast(path_len, src=0)

    if rank == 0:
        path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device=device)
    else:
        path_tensor = torch.zeros(path_len.item(), dtype=torch.uint8, device=device)

    dist.broadcast(path_tensor, src=0)
    weights_path = bytes(path_tensor.cpu().tolist()).decode()

    model_path = Path(weights_path)
    with open(model_path / "config.json", "r") as f:
        hf_config = json.load(f)

    llm_config = hf_config["text_config"]
    config = ModelConfig(
        n_embed=llm_config["hidden_size"],
        n_heads=llm_config["num_attention_heads"],
        n_kv_heads=llm_config["num_key_value_heads"],
        n_layer=llm_config["num_hidden_layers"],
        n_mlp=llm_config["intermediate_size"],
        n_vocab=llm_config["vocab_size"],
        tie_word_embeddings=hf_config["tie_word_embeddings"],
        rope_theta=llm_config["rope_theta"],
        rms_norm_eps=llm_config["rms_norm_eps"],
        d_head=llm_config.get("head_dim"),
        n_experts=llm_config.get("num_experts"),
        n_experts_per_token=llm_config.get("num_experts_per_tok"),
        n_moe_mlp=llm_config.get("moe_intermediate_size"),
    )

    model = Qwen3VL(config, vision_config=None)

    from safetensors.torch import load_file
    import glob as glob_mod
    safetensor_files = sorted(glob_mod.glob(str(model_path / "*.safetensors")))

    state_dict = {}
    for sf in safetensor_files:
        state_dict.update(load_file(sf, device="cpu"))

    model_state = model.state_dict()
    loaded_keys = []
    for key in model_state:
        if key in state_dict:
            model_state[key] = state_dict[key]
            loaded_keys.append(key)

    model.load_state_dict(model_state, strict=False)
    model = model.to(dtype=torch.bfloat16, device=device)

    if rank == 0:
        print(f"Loaded {len(loaded_keys)}/{len(model_state)} weight tensors from checkpoint")

    return model, config, weights_path


# ──────────────────────────── Main training loop ────────────────────────────

def setup_distributed_from_slurm():
    """Derive DDP env vars from Slurm when launched via srun (no torchrun).

    Slurm sets:
        SLURM_PROCID     – global rank
        SLURM_LOCALID    – local rank (GPU index on this node)
        SLURM_NTASKS     – world size
        SLURM_NODELIST   – compact node list, e.g. "node[01-04]"
        SLURM_STEP_NODELIST – same for srun steps (preferred when present)
    """
    if "SLURM_PROCID" not in os.environ:
        return  # not a Slurm srun launch

    # Only inject if torchrun hasn't already set them
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return

    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    # Master address: first node in the allocation
    nodelist = os.environ.get("SLURM_STEP_NODELIST") or os.environ.get("SLURM_NODELIST", "")
    if nodelist:
        import subprocess
        master = subprocess.check_output(
            ["scontrol", "show", "hostnames", nodelist],
            text=True,
        ).strip().split("\n")[0]
        os.environ.setdefault("MASTER_ADDR", master)
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")

    os.environ.setdefault("MASTER_PORT", "29500")


def is_ratio_eval(model, batch, device, max_new_tokens, temperature, loss_fn):
    """Importance Sampling ratio eval: forward_sample vs teacher-forced forward.

    1. Take a batch of input_ids as prompts (first half of tokens).
    2. forward_sample: autoregressively sample tokens, collecting per-token log-probs.
    3. forward (teacher-force): run the full sampled sequence through the model,
       extract per-token log-probs for the generated tokens.
    4. Compute ratio = exp(teacher_logprobs - rollout_logprobs) for generated tokens.

    Returns dict with ratio stats (mean, std, min, max) and teacher-forced eval loss.
    """
    input_ids = batch["input_ids"].to(device)
    B, T = input_ids.shape

    # Use first half of sequence as prompt
    prompt_len = T // 2
    prompt_ids = input_ids[:, :prompt_len]

    # Get the raw model (unwrap DDP)
    raw_model = model.module if hasattr(model, "module") else model

    # Step 1: forward_sample — autoregressive generation with log-prob collection
    sample_result = raw_model.forward_sample(
        input_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sampled_tokens = sample_result["sampled_tokens"]      # (B, prompt_len + gen_len)
    rollout_logprobs = sample_result["sampled_logprobs"]   # (B, prompt_len + gen_len)
    gen_start = sample_result["prompt_length"]

    # Step 2: teacher-force — single forward pass on the sampled sequence
    was_training = raw_model.training
    raw_model.eval()
    with torch.no_grad():
        logits = raw_model(input_ids=sampled_tokens)  # (B, seq_len, vocab)
        # logits[:, t, :] predicts token at position t+1
        # So log-prob for token at position t is from logits[:, t-1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        # Gather log-probs for the actual tokens at positions gen_start onwards
        # Token at position t was predicted by logits at position t-1
        teacher_logprobs = log_probs[:, :-1, :].gather(
            -1, sampled_tokens[:, 1:].unsqueeze(-1)
        ).squeeze(-1)  # (B, seq_len - 1)

    if was_training:
        raw_model.train()

    # Step 3: compute ratio for generated tokens only
    # Generated tokens start at position gen_start, so their log-probs in
    # teacher_logprobs are at indices gen_start-1 onwards (since teacher is shifted by 1)
    gen_len = sampled_tokens.shape[1] - gen_start
    if gen_len <= 0:
        return None

    # Rollout log-probs for generated tokens (positions gen_start to end)
    rollout_gen = rollout_logprobs[:, gen_start:].float()
    # Teacher log-probs for generated tokens (shifted by 1)
    teacher_gen = teacher_logprobs[:, gen_start - 1: gen_start - 1 + gen_len].float()

    # Align lengths
    min_len = min(rollout_gen.shape[1], teacher_gen.shape[1])
    rollout_gen = rollout_gen[:, :min_len]
    teacher_gen = teacher_gen[:, :min_len]

    ratio = torch.exp(teacher_gen - rollout_gen)

    return {
        "mean": ratio.mean().item(),
        "std": ratio.std().item(),
        "min": ratio.min().item(),
        "max": ratio.max().item(),
        "rollout_logprob_mean": rollout_gen.mean().item(),
        "teacher_logprob_mean": teacher_gen.mean().item(),
    }


def main():
    args = parse_args()

    # When launched via srun, populate the env vars that init_process_group needs
    setup_distributed_from_slurm()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    n_local_gpus = torch.cuda.device_count()
    if rank == 0:
        print(f"World size: {world_size}, local GPUs: {n_local_gpus}, "
              f"device: {torch.cuda.get_device_name(local_rank)}")

    # Load model
    if rank == 0:
        print(f"Loading model {args.model_name}...")

    model, config, weights_path = load_model_for_training(args, local_rank)

    # Flash Attention 4 (cute)
    if args.flash_attn_cute:
        enable_flash_attn_cute(True)
        if rank == 0:
            print("Using Flash Attention 4 (flash_attn.cute)")

    # Replace with TE layers if requested
    te_context_fn = nullcontext  # factory: callable that returns a context manager
    if args.te_fp8 or args.te_fp4:
        import transformer_engine.pytorch as te
        n_replaced = replace_with_te_layers(model, use_fp8=args.te_fp8, use_fp4=args.te_fp4)
        if rank == 0:
            print(f"Replaced {n_replaced} layers with Transformer Engine equivalents")
        recipe = get_te_recipe(args.te_fp8_recipe, use_fp4=args.te_fp4)
        te_context_fn = lambda: te.fp8_autocast(enabled=True, fp8_recipe=recipe)

    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    model.train()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Dataset
    if args.random_data:
        dataset = RandomTextDataset(config.n_vocab, args.seq_len,
                                     size=max(10000, args.steps * args.batch_size * world_size))
        if rank == 0:
            print("Using random data for smoke testing")
    else:
        data_path = os.path.join(args.dataset_dir, "llava_instruct_150k.json")
        if not os.path.exists(data_path):
            if rank == 0:
                print(f"Dataset not found at {data_path}. Downloading...")
                download_dataset(args.dataset_dir)
            dist.barrier()

        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(weights_path, "tokenizer.json"))
        dataset = LLaVAInstructDataset(data_path, tokenizer, seq_len=args.seq_len)
        if rank == 0:
            print(f"Loaded {len(dataset)} samples from LLaVA-Instruct-150k")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        fused=True,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # CUDA Graph via torch.compile
    if args.cuda_graphs:
        model = apply_cuda_graph_compile(model, rank)

    # Profiler
    profiler = create_profiler(args, rank)

    # TensorBoard
    tb = TBLogger(args.tensorboard, args.tb_dir, rank)

    # Training
    if rank == 0:
        eff_batch = args.batch_size * args.grad_accum * world_size
        print(f"\nStarting training: {args.steps} steps, micro_batch={args.batch_size}, "
              f"grad_accum={args.grad_accum}, effective_batch={eff_batch}, "
              f"seq_len={args.seq_len}, lr={args.lr}")
        print(f"TE FP8: {args.te_fp8}, TE FP4: {args.te_fp4}, "
              f"CUDA Graphs: {args.cuda_graphs}, Profile: {args.profile}, "
              f"TensorBoard: {args.tensorboard}")
        print("-" * 80)

    data_iter = iter(dataloader)
    epoch = 0
    total_loss = 0.0
    step_times = []
    log_losses = []
    recent_times = deque(maxlen=20)  # for FPS over last 20 batches

    with (profiler if args.profile and rank == 0 else nullcontext()) as prof:
        for step in range(args.steps):
            step_start = time.time()

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # LR schedule
            lr = get_lr(step, args.warmup_steps, args.steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            accum_loss = 0.0
            for micro_step in range(args.grad_accum):
                # Get micro-batch (reuse first batch from outer loop for micro_step 0)
                if micro_step > 0:
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        epoch += 1
                        sampler.set_epoch(epoch)
                        data_iter = iter(dataloader)
                        batch = next(data_iter)
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)

                # Skip DDP allreduce on non-final micro-steps
                ctx = model.no_sync if micro_step < args.grad_accum - 1 else nullcontext
                with ctx(), te_context_fn():
                    logits = model(input_ids=input_ids)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss = loss / args.grad_accum
                loss.backward()
                accum_loss += loss.item()

            grad_norm = None
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                ).item()

            optimizer.step()

            step_time = time.time() - step_start
            step_times.append(step_time)
            recent_times.append(step_time)
            loss_val = accum_loss
            total_loss += loss_val
            log_losses.append(loss_val)

            # TensorBoard logging
            tb.scalar("train/loss", loss_val, step)
            tb.scalar("train/lr", lr, step)
            tb.scalar("train/step_time_ms", step_time * 1000, step)
            if grad_norm is not None:
                tb.scalar("train/grad_norm", grad_norm, step)
            # FPS over last 20 batches
            if len(recent_times) > 0:
                avg_recent = sum(recent_times) / len(recent_times)
                fps_recent = (args.batch_size * args.grad_accum * args.seq_len * world_size) / avg_recent
                tb.scalar("perf/tokens_per_sec_last20", fps_recent, step)
            tb.scalar("perf/memory_gb", torch.cuda.max_memory_allocated(device) / 1e9, step)

            if rank == 0 and (step + 1) % args.log_interval == 0:
                n = min(args.log_interval, len(log_losses))
                avg_loss = sum(log_losses[-n:]) / n
                avg_step_time = sum(step_times[-n:]) / n
                tokens_per_sec = (args.batch_size * args.grad_accum * args.seq_len * world_size) / avg_step_time
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                gn_str = f"gnorm {grad_norm:.2f} | " if grad_norm is not None else ""

                print(f"step {step+1:5d}/{args.steps} | "
                      f"loss {avg_loss:.4f} | "
                      f"{gn_str}"
                      f"lr {lr:.2e} | "
                      f"step_time {avg_step_time*1000:.0f}ms | "
                      f"tokens/s {tokens_per_sec:.0f} | "
                      f"mem {mem_gb:.1f}GB")

            # Eval on held-out batch (right after optimizer step)
            if args.eval_interval > 0 and (step + 1) % args.eval_interval == 0:
                try:
                    eval_batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    data_iter = iter(dataloader)
                    eval_batch = next(data_iter)
                eval_ids = eval_batch["input_ids"].to(device)
                eval_labels = eval_batch["labels"].to(device)
                with torch.no_grad(), te_context_fn():
                    eval_logits = model(input_ids=eval_ids)
                eval_shift = eval_logits[:, :-1, :].contiguous()
                eval_lab = eval_labels[:, 1:].contiguous()
                eval_loss = loss_fn(
                    eval_shift.view(-1, eval_shift.size(-1)),
                    eval_lab.view(-1),
                ).item()
                tb.scalar("eval/loss", eval_loss, step)
                if rank == 0:
                    print(f"  [eval] step {step+1} | eval_loss {eval_loss:.4f}")

            # IS ratio eval (right after optimizer step)
            if args.is_ratio_interval > 0 and (step + 1) % args.is_ratio_interval == 0:
                try:
                    is_batch = next(data_iter)
                except StopIteration:
                    epoch += 1
                    sampler.set_epoch(epoch)
                    data_iter = iter(dataloader)
                    is_batch = next(data_iter)
                is_stats = is_ratio_eval(
                    model, is_batch, device,
                    max_new_tokens=args.is_ratio_tokens,
                    temperature=args.is_ratio_temperature,
                    loss_fn=loss_fn,
                )
                if is_stats is not None:
                    tb.scalar("is_ratio/mean", is_stats["mean"], step)
                    tb.scalar("is_ratio/std", is_stats["std"], step)
                    tb.scalar("is_ratio/min", is_stats["min"], step)
                    tb.scalar("is_ratio/max", is_stats["max"], step)
                    tb.scalar("is_ratio/rollout_logprob", is_stats["rollout_logprob_mean"], step)
                    tb.scalar("is_ratio/teacher_logprob", is_stats["teacher_logprob_mean"], step)
                    if rank == 0:
                        print(f"  [is_ratio] step {step+1} | mean {is_stats['mean']:.6f} "
                              f"std {is_stats['std']:.2e} "
                              f"rollout_lp {is_stats['rollout_logprob_mean']:.4f} "
                              f"teacher_lp {is_stats['teacher_logprob_mean']:.4f}")

            if prof is not None and args.profile:
                prof.step()

            # Save checkpoint
            if rank == 0 and args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_path = os.path.join(args.save_dir, f"step_{step+1}.pt")
                torch.save({
                    "step": step + 1,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val,
                }, ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    # Final summary
    if rank == 0:
        avg_loss = total_loss / args.steps
        avg_step_time = sum(step_times) / len(step_times)
        total_time = sum(step_times)
        print("\n" + "=" * 80)
        print(f"Training complete!")
        print(f"  Steps: {args.steps}")
        print(f"  Final avg loss: {avg_loss:.4f}")
        print(f"  Last loss: {loss_val:.4f}")
        print(f"  Avg step time: {avg_step_time*1000:.0f}ms")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Peak memory: {torch.cuda.max_memory_allocated(device)/1e9:.1f}GB")
        if args.profile:
            print(f"  Profiler traces saved to: {args.profile_dir}/")
        if args.tensorboard:
            print(f"  TensorBoard logs saved to: {args.tb_dir}/")
        print("=" * 80)

    tb.close()
    dist.destroy_process_group()


def download_dataset(dataset_dir):
    """Download LLaVA-Instruct-150k dataset."""
    os.makedirs(dataset_dir, exist_ok=True)
    target = os.path.join(dataset_dir, "llava_instruct_150k.json")
    if os.path.exists(target):
        print(f"Dataset already exists at {target}")
        return

    print("Downloading LLaVA-Instruct-150k...")
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id="liuhaotian/LLaVA-Instruct-150K",
        filename="llava_instruct_150k.json",
        repo_type="dataset",
        local_dir=dataset_dir,
    )
    print(f"Downloaded to {downloaded}")


if __name__ == "__main__":
    main()
