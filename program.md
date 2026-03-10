# Autonomous Optimizer Research — Agent Instructions

You are an autonomous research agent. Your goal is to discover optimizers that achieve the lowest `val_bpb` on a fixed GPT model trained on web text.

The model architecture and training loop are fixed. You only modify the **optimizer** — the algorithm, hyperparameters, learning rate schedules, gradient manipulation, and any custom optimizer logic.

## Setup Phase

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autogradresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autogradresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Contains model (FIXED), optimizer (EDITABLE), and training loop (FIXED).
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Constraints

**What you CAN do:**
- Modify `train.py` ONLY between `# ===== OPTIMIZER START =====` and `# ===== OPTIMIZER END =====`
- Within that section: change optimizer class, hyperparameters, LR schedules, gradient manipulation, add custom optimizer implementations, etc.
- You may add `import` statements at the top of the optimizer section (only stdlib and torch — no external packages)
- Implement custom optimizer classes from scratch within the section

**What you CANNOT do:**
- Modify anything outside the OPTIMIZER START/END markers in `train.py`
- Modify `prepare.py` — it is read-only
- Change the model architecture, training loop, batch size, model size, or evaluation
- Install new packages or add dependencies — everything must use only stdlib + torch

**Required function signatures** (the training loop calls these):
- `create_optimizer(model) -> optimizer` — build and return the optimizer
- `step_optimizer(optimizer, model, step, progress) -> lrm` — apply schedules, step, zero grads, return a float `lrm` (LR multiplier, used for logging)

**The goal is simple: get the lowest val_bpb.** The model, data, and time budget are identical across all experiments — only the optimizer varies.

**VRAM** is a soft constraint. Some increase from optimizer state is acceptable, but don't blow up memory dramatically. The target GPU is an RTX 4090 (24GB).

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so run the training script as-is.

## Understanding the Baseline

The default optimizer is **MuonAdamW** — this is NOT basic Muon. It's an already highly-tuned setup:

**Parameter groups** (the model has 6 types of parameters):
- `model.lm_head` — unembedding matrix → AdamW
- `model.transformer.wte` — token embeddings → AdamW
- `model.value_embeds` — value embeddings (ResFormer) → AdamW
- `model.resid_lambdas` — per-layer residual scalars → AdamW
- `model.x0_lambdas` — per-layer skip connection scalars → AdamW
- `model.transformer.h` — all 2D transformer matrices (attention Q/K/V/proj, MLP) → Muon

**Muon features already in the baseline:**
- Newton-Schulz orthogonalization (5 iterations, "polar express" coefficients)
- Nesterov momentum with warmup (0.85 → 0.95 over 300 steps)
- NorMuon variance reduction (second momentum buffer, beta2=0.95)
- Cautious updates (only update where gradient aligns with parameter direction)
- Weight decay linearly decayed to 0

**Schedules already in the baseline:**
- No warmup, 50% warmdown (linear to 0)
- LR scaling: `1/sqrt(model_dim/768)` for AdamW groups
- Muon LR additionally scaled by `sqrt(max(1, rows/cols))` per weight shape

This means naive changes like "add warmup" or "try cosine decay" may or may not help — the baseline is already sophisticated. Read the code carefully before experimenting.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     15200.3
mfu_percent:      18.50
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract the key metric:

```
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved — use 0.000000 for crashes
3. peak memory in GB, round to .1f (divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	14.8	keep	baseline MuonAdamW
b2c3d4e	0.993200	14.9	keep	increase Muon LR to 0.06
c3d4e5f	1.005000	14.8	discard	pure AdamW (no Muon)
d4e5f6g	0.000000	0.0	crash	custom optimizer bug
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit the optimizer section in `train.py` with an experimental idea
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix
7. Record the results in the tsv (do NOT commit results.tsv)
8. If val_bpb improved (lower), keep the commit and advance
9. If val_bpb is equal or worse, `git reset --hard` back to where you started

**Timeout**: Each experiment takes ~5 minutes (+ startup overhead). If a run exceeds 10 minutes, kill it and treat as failure.

**Crashes**: If it's a typo/import bug, fix and re-run. If fundamentally broken, log as crash and move on.

**NEVER STOP**: Do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, think harder. The loop runs until the human interrupts you, period.

## Research Directions

### Phase 1: Baselines and Hyperparameter Tuning
Start by understanding the current MuonAdamW baseline, then tune:
- Muon learning rate (try 0.02, 0.03, 0.06, 0.08)
- AdamW learning rates for embeddings/unembeddings
- Weight decay values and schedules
- Adam betas
- Warmup/warmdown ratios

### Phase 2: Learning Rate Scheduling
Experiment with different schedules:
- Add warmup (try 0.05, 0.1, 0.2 warmup ratios)
- Cosine decay instead of linear warmdown
- Different warmdown ratios
- OneCycleLR-style schedule
- Final LR fraction (try 0.01 instead of 0.0)

### Phase 3: Muon Internals
The Muon optimizer has several components that can be individually tuned or replaced:
- Newton-Schulz iteration count (try 3, 4, 6, 7)
- Newton-Schulz coefficients (the polar_express_coeffs)
- Momentum warmup schedule (currently 0.85->0.95 over 300 steps)
- NorMuon variance reduction (try disabling it, or tuning beta2)
- Cautious updates mask (try removing it, or different thresholds)
- Weight decay integration (decoupled vs coupled, schedule shape)

### Phase 4: Gradient Manipulation
Add gradient-level modifications within the optimizer:
- Gradient clipping (norm or value-based)
- Gradient noise injection (decaying with step)
- Gradient centralization
- Layer-wise LR scaling (LARS/LAMB-style)
- Gradient normalization per-layer

### Phase 5: Alternative Optimizers
Replace Muon entirely or modify its core (must implement from scratch using only torch):
- SOAP (spectral optimizer — related to Muon, uses SVD/eigendecomposition)
- Shampoo (preconditioned — maintain row/col preconditioners)
- Lion (sign-based — very simple, just sign of momentum)
- Prodigy (learning-rate-free — adapts LR automatically)
- LAMB/LARS for the matrix params
- Pure AdamW with aggressive tuning (is Muon actually helping here?)
- Use the existing MuonAdamW class as a template for the param group structure

### Phase 6: Novel Combinations
Combine the best findings:
- Best optimizer + best schedule
- Lookahead wrapper around Muon
- Stochastic Weight Averaging
- EMA of parameters
- Different optimizers for different layer depths
- Hybrid schedules (e.g., warmup + cosine for AdamW, different schedule for Muon)

## Tips

- The baseline MuonAdamW is already highly tuned — small improvements matter
- Muon handles the 2D matrix parameters, AdamW handles embeddings/scalars — this split is important
- The `step_optimizer` function gives you full control over per-step logic (scheduling, gradient manipulation, etc.)
- `create_optimizer` gives you control over param groups, optimizer class, and initial config
- Weight decay, LR, and momentum schedules interact — change one at a time
- The 5-minute budget means every optimization step counts — avoid adding expensive per-step computation
- The fused optimizer functions use `@torch.compile(dynamic=False, fullgraph=True)` — if you modify them, avoid Python-level data-dependent control flow (use `torch.where` instead of `if`). Or remove `@torch.compile` if needed (slower but more flexible)
- Alternative optimizers (SOAP, Lion, Shampoo, etc.) must be implemented from scratch — you cannot install external packages. Use the existing MuonAdamW as a template
- `step_optimizer` must return a float `lrm` — the training loop uses it for logging. If your optimizer doesn't have a simple LR multiplier, return 1.0
- Record every experiment, even failures — they contain information
