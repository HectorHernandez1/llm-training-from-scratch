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
python scripts/gpu-smoke-test.py                         # does the GPU actually train? (exit 1 = broken)
pip install -r requirements.txt
pip install torch                                        # Mac/CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
bash runpod/setup.sh                                     # fresh RunPod pod (idempotent)
bash scripts/setup-local.sh                              # Mac or 5090 — auto-detects (idempotent)
jupyter lab
```

Both machines use the same conda env, `llm-training` on Python 3.12 (`conda
activate llm-training`) — not base's 3.13.

## Non-obvious gotchas

- **`torch` is deliberately NOT in `requirements.txt`** and must never be pinned
  there — it's installed per-machine so each environment (Mac CPU, RunPod CUDA
  12.x, RTX 5090 / CUDA 12.8) gets the right build. Do not "fix" this.
- **The RTX 5090 is Blackwell (`sm_120`) and needs the cu128 wheel.** A wheel
  without `sm_120` kernels imports fine and reports `cuda.is_available() ==
  True`, then fails at the first real matmul with "no kernel image is available
  for execution on the device". If someone hits that, it's the wrong wheel, not
  their code — `scripts/gpu-smoke-test.py` checks the arch list first thing.
- **`vendor/` is gitignored** except `vendor/README.md`. These are shallow
  clones of Karpathy's reference repos — read-only reference material. **Never
  edit or commit anything under `vendor/`.** `runpod/setup.sh` re-clones any
  missing ones idempotently.
- **Never commit large artifacts.** `data/`, `checkpoints/`, `wandb/`, and all
  `*.pt` / `*.bin` / `*.safetensors` are gitignored (see `.gitignore`).
- **This repo is used on two machines** (Apple Silicon Mac + RTX 5090 box) and
  synced through git. Anything you add must run on both — don't assume CUDA, and
  don't assume MPS either. See `scripts/env-check.py` and
  `scripts/gpu-smoke-test.py` for the pattern (try/except around `import torch`,
  branch on `cuda.is_available()` / `backends.mps.is_available()`, fall through
  to CPU, `sched_getaffinity` for cgroup-aware CPU counts).
- **Notebooks are edited on both machines and merge with nbdime**, configured by
  `.gitattributes` plus a driver that `scripts/setup-local.sh` enables per
  machine. Cell **outputs stay committed on purpose** — the plots and graphviz
  diagrams are the lab notebook's value, so don't add nbstripout or strip them
  to "clean up" a diff. Resolve conflicts with `nbdiff`/`nbmerge`, never by
  hand-editing notebook JSON.
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
