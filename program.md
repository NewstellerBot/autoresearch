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

### Phase 1: Establish Baseline (2-3 experiments max, then move on)
- Run the default MuonAdamW as-is to get a reference val_bpb
- Try 1-2 quick hyperparameter tweaks (Muon LR, warmup ratio) to confirm the baseline is near-optimal
- Do NOT spend more than a few experiments here — the baseline is already well-tuned

### Phase 2: Quasi-Newton Methods
The core research direction. Quasi-Newton methods approximate second-order curvature without computing the full Hessian. The key question: can we beat Muon's orthogonalization with curvature-aware updates?

Ideas to try:
- **Online L-BFGS**: Maintain a limited-memory BFGS approximation using recent gradient pairs (s, y vectors). Apply the two-loop recursion to compute H*g cheaply. Key challenge: stochastic gradients make curvature estimates noisy — try damped updates, skip updates when curvature estimate is bad (y^T s < threshold)
- **Structured quasi-Newton for matrices**: Exploit the 2D structure of weight matrices. Maintain separate row-space and column-space curvature approximations (Kronecker-factored L-BFGS). This is much cheaper than full L-BFGS for large matrices
- **Diagonal quasi-Newton**: Approximate the Hessian as diagonal only — essentially an adaptive per-parameter learning rate learned from curvature. Simpler than full L-BFGS, might work well with Muon's orthogonalization on top
- **L-BFGS + Muon hybrid**: Use L-BFGS curvature to precondition gradients, THEN apply Muon's orthogonalization. Or vice versa: orthogonalize first, then apply curvature correction
- **SR1 (Symmetric Rank-1) updates**: Alternative to BFGS that allows indefinite Hessian approximations — may work better for non-convex landscapes. Cheaper per-step than BFGS
- **Online Newton with sketching**: Use random sketching (count sketch, random projection) to maintain a compressed Hessian approximation that fits in memory

### Phase 3: Curvature-Aware Preconditioning
Methods that use second-order information without full quasi-Newton machinery:

- **K-FAC (Kronecker-Factored Approximate Curvature)**: Approximate the Fisher information matrix as a Kronecker product of layer input/output statistics. Very natural for linear layers — maintain running averages of A = E[a*a^T] and G = E[g*g^T], precondition as G^{-1} * grad * A^{-1}
- **Shampoo**: Maintain left and right preconditioners L, R for each weight matrix W. Update rule: W -= lr * L^{-1/p} @ grad @ R^{-1/p}. Periodically recompute L, R from gradient outer products. Try different recomputation intervals
- **Natural gradient via diagonal Fisher**: Approximate Fisher as diagonal — just the running average of squared gradients (like Adam's v), but use it as a true preconditioner rather than element-wise scaling
- **Layer-wise adaptive preconditioning**: Learn a scalar or low-rank preconditioner per layer based on gradient statistics. Cheaper than K-FAC but captures inter-layer curvature differences

### Phase 4: Novel Gradient Transformations
Go beyond Muon's orthogonalization — invent new ways to transform gradients before applying them:

- **Spectral gradient reshaping**: Compute SVD of the gradient (or approximate via power iteration), then reshape the singular value spectrum. Muon makes all singular values equal (orthogonalization). What if you keep relative magnitudes but compress the spectrum? Or boost small singular values more?
- **Gradient whitening**: Decorrelate gradient components using a running estimate of the gradient covariance. Different from Adam (which only tracks diagonal variance)
- **Polar decomposition variants**: Muon approximates the polar factor U of G = U*S. What about using U*f(S) for some nonlinear f? Or using only the top-k singular components?
- **Hyperbolic gradient transformations**: Apply hyperbolic tangent or other nonlinearities to gradient components to handle heavy-tailed gradient distributions
- **Momentum on the manifold**: Instead of Euclidean momentum (linear interpolation of gradients), try momentum that respects the geometry of the parameter space — e.g., transport previous momentum to current tangent space before combining

### Phase 5: Adaptive and Meta-Learning Approaches
Methods that learn to optimize during training:

- **Hypergradient descent**: Compute the gradient of the loss w.r.t. the learning rate itself (using the chain rule through the optimizer step). Use this to adapt LR online. Can also adapt momentum, weight decay
- **Per-layer learned LR**: Instead of a single schedule, learn a separate LR multiplier per layer (or per param group) that adapts based on gradient statistics
- **Loss-aware updates**: Scale updates based on recent loss trajectory — be more aggressive when loss is decreasing steadily, more conservative near plateaus or after loss spikes
- **Dual averaging / mirror descent**: Use a different geometry for the optimization space — e.g., entropy-regularized updates that naturally keep parameters from growing too large

### Phase 6: Hybrid and Frontier Approaches
Combine the best findings with creative architecture-aware optimization:

- **Best quasi-Newton + Muon orthogonalization**: If a curvature method helps, combine it with the best gradient transformation
- **Cascaded preconditioning**: Apply multiple preconditioning steps in sequence (e.g., K-FAC → spectral reshaping → momentum)
- **Attention-aware optimization**: Different optimization strategies for attention weights vs MLP weights (they have very different loss landscapes)
- **Progressive optimizer switching**: Start with one optimizer (e.g., Adam for stability), switch to a more aggressive one (e.g., quasi-Newton) after warmup
- **Population-based schedule search**: Try random perturbations to the schedule/hyperparameters each run, keep improvements (effectively doing meta-optimization across experiments)

## Tips

- All custom optimizers must be implemented from scratch using only stdlib + torch — no external packages
- Use the existing MuonAdamW class as a template for the param group structure (Muon for 2D matrices, AdamW for embeddings/scalars)
- The `step_optimizer` function gives you full control over per-step logic (scheduling, gradient manipulation, etc.)
- `create_optimizer` gives you control over param groups, optimizer class, and initial config
- The fused optimizer functions use `@torch.compile(dynamic=False, fullgraph=True)` — if you modify them, avoid Python-level data-dependent control flow (use `torch.where` instead of `if`). Or remove `@torch.compile` if needed (slower but more flexible)
- `step_optimizer` must return a float `lrm` — the training loop uses it for logging. If your optimizer doesn't have a simple LR multiplier, return 1.0
- The 5-minute budget means per-step overhead matters. Quasi-Newton methods that do O(n^2) work per step may be too slow for large layers — use low-rank or sketched approximations. Profile if unsure
- For quasi-Newton: the memory budget is ~24GB total. Budget optimizer state carefully — L-BFGS with history size m stores 2m vectors per parameter group
- SVD and eigendecomposition are available via `torch.linalg.svd`, `torch.linalg.eigh` etc. For approximate/truncated SVD, use power iteration (`torch.svd_lowrank`)
- When implementing novel methods, start simple (e.g., diagonal approximation), verify it doesn't crash or diverge, then add complexity
- Record every experiment, even failures — a method that diverges tells you something about the loss landscape
