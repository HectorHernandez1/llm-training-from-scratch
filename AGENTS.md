# AGENTS.md

Compact guidance for AI agents working in this repo. Longer prose lives in
`CLAUDE.md`; this file is just the high-signal rules you'd otherwise get wrong.

## What this repo is (and isn't)

A personal **lab notebook** for a 4-week LLM-training curriculum — notes,
code-along notebooks, and a training-run log. **It is not a software product.**

- There is **no build, no test suite, no lint, no typecheck.** Don't look for
  them or try to add them. `python scripts/env-check.py` is the only "check."
- `PLAN.md` is the spine (curriculum + progress checkboxes). `README.md` holds
  the **Run log** — add one dated entry per training run using the template there.

## The one behavioral rule that matters most

This is a *learning* repo: the user types the course code themselves while
following Karpathy's videos. **Do not pre-write course/assignment code for
them.** Help with setup, debugging, explanations, and notes instead. (The
`weekN-*` notebooks and `notes/*` are meant to be filled in by hand.)

## Commands

```bash
python scripts/env-check.py                              # first thing on any machine
pip install -r requirements.txt
pip install torch                                        # Mac/CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
bash runpod/setup.sh                                     # fresh RunPod pod (idempotent)
jupyter lab
```

## Non-obvious gotchas

- **`torch` is deliberately NOT in `requirements.txt`** and must never be pinned
  there — it's installed per-machine so each environment (Mac CPU, RunPod CUDA
  12.x, RTX 5090 / CUDA 12.8) gets the right build. Do not "fix" this.
- **`vendor/` is gitignored** except `vendor/README.md`. These are shallow
  clones of Karpathy's reference repos — read-only reference material. **Never
  edit or commit anything under `vendor/`.** `runpod/setup.sh` re-clones any
  missing ones idempotently.
- **Never commit large artifacts.** `data/`, `checkpoints/`, `wandb/`, and all
  `*.pt` / `*.bin` / `*.safetensors` are gitignored (see `.gitignore`).
- Scripts that touch hardware must **degrade gracefully without CUDA** — see
  `scripts/env-check.py` for the pattern (try/except around `import torch`,
  MPS fallback, `sched_getaffinity` for cgroup-aware CPU counts).
- `graphviz` (in requirements.txt) needs the system binary too:
  `brew install graphviz` / `apt install graphviz`.

## Conventions to preserve when editing

- **`notes/*.md`** — one file per video/resource, numbered to match `PLAN.md`
  (`01-micrograd.md` … `05-gpt2-repro.md`, plus `memory-math.md`). Keep the
  header structure: a `**▶ Resume at:**` line tracking video position, then
  sections **Key concepts / Code I wrote / Questions I had / Things that
  surprised me**.
- **`weekN-*/`** — one directory per curriculum week, holding code-along
  notebooks and experiments.
