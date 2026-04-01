# Importance Sampling (IS) Ratio Eval

## What it measures

The IS ratio eval quantifies the consistency between a model's **autoregressive sampling** (rollout) and its **teacher-forced forward pass**. In a mathematically exact system, both paths would assign identical log-probabilities to the same tokens. In practice, they diverge due to numerical precision (bf16 accumulation order), kernel differences (KV-cached attention vs full-sequence attention), and any nondeterminism in the compute path.

Tracking this ratio during training serves as a canary: if the ratio drifts significantly from 1.0, something in the forward path has changed in a way that may affect generation quality even if training loss looks fine.

## How it works

The eval runs every N training steps (default: 100) and consists of three phases:

### Phase 1: Rollout (forward_sample)

Given a training batch of token sequences `[t_0, t_1, ..., t_T]`, split each sequence in half. The first half `[t_0, ..., t_{T/2}]` becomes the **prompt**.

The model then generates `max_new_tokens` (default: 128) tokens autoregressively from the prompt using temperature sampling:

```
for each new position i:
    logits = model.forward(all tokens so far)     # full recompute, no KV cache
    log_probs = log_softmax(logits[:, -1, :])     # distribution over next token
    probs = softmax(logits[:, -1, :] / temperature)
    next_token ~ Categorical(probs)               # sample
    rollout_logprob[i] = log_probs[next_token]    # record the log-prob under unscaled distribution
```

This produces:
- `sampled_tokens`: `(B, prompt_len + gen_len)` -- the full sequence (prompt + generated)
- `rollout_logprobs`: `(B, prompt_len + gen_len)` -- log-prob of each token at generation time (prompt positions are 0.0)

**Important detail**: the log-prob recorded is from `log_softmax(logits)` (no temperature scaling), not from the temperature-scaled distribution used for sampling. This means we're recording `log pi(token)` under the true model distribution, even though the *sampling* used temperature. This is intentional -- we want to compare the model's probability assignment, not the sampling procedure.

### Phase 2: Teacher-force (forward)

Take the complete `sampled_tokens` sequence from Phase 1 and run a single forward pass through the model:

```
logits = model.forward(sampled_tokens)            # (B, seq_len, vocab)
log_probs = log_softmax(logits)                    # (B, seq_len, vocab)
teacher_logprob[t] = log_probs[t-1, sampled_tokens[t]]  # shifted by 1
```

The standard autoregressive shift applies: `logits[:, t, :]` predicts the token at position `t+1`. So the teacher-forced log-prob for token at position `t` comes from `logits[:, t-1, :]`.

### Phase 3: Compute ratio

For each **generated** token (positions `prompt_len` to end), compute:

```
ratio = exp(teacher_logprob - rollout_logprob)
```

Or equivalently:

```
ratio = pi_teacher(token) / pi_rollout(token)
```

This is the importance sampling weight. If the model is perfectly self-consistent:
- `teacher_logprob == rollout_logprob` for every token
- `ratio == 1.0` everywhere
- `std == 0.0`

## Why it can deviate from 1.0

In tiny-qwen's current implementation, `forward_sample` does a full forward pass at each step (no KV cache), using the same code path as teacher-forcing. So the ratio is expected to be very close to 1.0 -- deviations come only from:

1. **bf16 floating point non-associativity**: The order of accumulation in matrix multiplications can differ between the full-sequence forward (teacher) and the incremental forward (rollout), since the sequence lengths differ. With bf16, `(a + b) + c != a + (b + c)`.

2. **Attention mask differences**: During rollout, the causal mask is naturally satisfied (the model only sees tokens up to the current position). During teacher-forcing, an explicit causal mask is applied to a longer sequence, which may use a different SDPA kernel.

In production systems with KV caching (like the `ai` repo's `QwenDecoderTrunk.forward_sample`), additional sources of divergence include:
- Different attention kernels for prefill vs decode (FA2/FA3 prefill vs FA2/FA3 KV-cache decode)
- KV cache quantization
- CUDA graph capture changing kernel selection
- Different numerical paths for single-token decode vs batched prefill

## Metrics logged to TensorBoard

Every `--is-ratio-interval` steps (default 100), the following scalars are logged:

| Metric | Description |
|--------|-------------|
| `is_ratio/mean` | Mean ratio across all generated tokens in the batch. Should be ~1.0. |
| `is_ratio/std` | Standard deviation. Should be ~0. Higher values mean more inconsistency. |
| `is_ratio/min` | Minimum ratio. Values << 1.0 mean teacher assigns much lower probability than rollout. |
| `is_ratio/max` | Maximum ratio. Values >> 1.0 mean teacher assigns much higher probability than rollout. |
| `is_ratio/rollout_logprob` | Mean log-prob assigned during autoregressive generation. |
| `is_ratio/teacher_logprob` | Mean log-prob assigned during teacher-forced forward. |

The raw `rollout_logprob` and `teacher_logprob` are logged so you can see which direction the divergence goes -- whether teacher-forcing is more or less confident than rollout.

## Interpreting results

| Observation | Interpretation |
|-------------|----------------|
| `mean ~= 1.0, std ~= 0` | Model is self-consistent. Normal. |
| `mean ~= 1.0, std > 0` | Per-token ratios vary but average out. Check min/max for outliers. |
| `mean > 1.0` | Teacher-forcing assigns higher probability on average. May indicate the full-context forward "sees" patterns that incremental generation misses. |
| `mean < 1.0` | Teacher-forcing assigns lower probability on average. May indicate KV cache or kernel differences favor generation. |
| `mean drifting over training` | The consistency property is changing as the model trains. Worth investigating if the drift is large. |
| `std increasing over training` | The model is becoming less self-consistent. Could indicate numerical issues amplified by sharper distributions as the model trains. |

## Connection to GRPO / RL training

In GRPO (Group Relative Policy Optimization), the IS ratio `pi_new / pi_old` is used to compute the policy gradient with importance sampling correction. The IS ratio eval here is a special case: instead of comparing two different model checkpoints (old policy vs new policy), it compares two *forward paths* of the **same** model.

If the same-model IS ratio is not ~1.0, then the GRPO ratio `pi_new / pi_old` (which relies on `forward_sample` for `pi_old` and `forward` for `pi_new`) will have a systematic bias beyond the actual policy change. The clip bounds in GRPO (`1 - epsilon` to `1 + epsilon`) are meant to bound the policy change, but if there's already a bias from the forward path inconsistency, the effective clip range is shifted.

## Configuration

```bash
# Default: IS ratio eval every 100 steps, 128 generated tokens, temperature 1.0
torchrun ... train_ddp.py --steps 20000 --tensorboard

# Custom interval and generation length
torchrun ... train_ddp.py --is-ratio-interval 200 --is-ratio-tokens 256 --is-ratio-temperature 0.8

# Disable
torchrun ... train_ddp.py --is-ratio-interval 0
```

## Code locations

- `model/model.py:Qwen3VL.forward_sample()` -- autoregressive sampling with log-prob collection
- `train_ddp.py:is_ratio_eval()` -- orchestrates the three phases and computes statistics
- Training loop integration: runs after the optimizer step at the configured interval
