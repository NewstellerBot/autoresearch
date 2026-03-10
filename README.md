# autogradresearch

Autonomous optimizer research: an AI agent experiments with optimizers to beat MuonAdamW on a fixed GPT model.

Forked from [karpathy/autoresearch](https://github.com/karpathy/autoresearch). In autoresearch, the agent modifies everything (model, optimizer, training loop). Here, the **model and training loop are fixed** — only the **optimizer** is the research target.

## How it works

- **`prepare.py`** — Fixed: data download, BPE tokenizer, dataloader, evaluation (`val_bpb`). Not modified.
- **`train.py`** — Contains the GPT model (FIXED), optimizer section between `OPTIMIZER START`/`OPTIMIZER END` markers (EDITABLE), and training loop (FIXED).
- **`program.md`** — Instructions for the AI agent to run autonomous experiments.

The agent edits only the optimizer section: the optimizer class, hyperparameters, LR schedules, gradient manipulation, and any custom optimization logic.

## Quick start

**Requirements:** Single NVIDIA GPU (tested on RTX 4090), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run prepare.py    # download data + train tokenizer (one-time)
uv run train.py      # baseline run (~5 min)
```

## Running the agent

Point your agent at `program.md`:

```
Read program.md and let's kick off a new experiment run.
```

## Design

| Aspect | autoresearch | autogradresearch |
|--------|-------------|------------------|
| Research target | Everything (model + optimizer) | Optimizer only |
| Fixed | `prepare.py` only | Model + training loop + `prepare.py` |
| Editable | All of `train.py` | Optimizer section in `train.py` |
| Metric | val_bpb (lower=better) | val_bpb (lower=better) |
| Time budget | 5 min | 5 min |
| Baseline optimizer | MuonAdamW | MuonAdamW (target to beat) |

## License

MIT
