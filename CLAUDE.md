# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Note: a CLAUDE.md from the parent directory (`~/Desktop/repo/CLAUDE.md`,
describing the CADENCE training-tracker app) may also appear in context. It is a
different project — none of its instructions apply to this repo.

## What this repo is

A personal lab notebook for a 4-week LLM-training curriculum (Karpathy's Zero to
Hero → nanoGPT → QLoRA fine-tuning). It is not a software product: there is no
build, no test suite, no lint. The deliverables are notes, code-along notebooks,
and a log of training runs. **PLAN.md is the spine** — the curriculum with
checkboxes tracking progress. The Run log at the bottom of README.md gets one
entry per training run (template is in the README).

This is a learning repo: the code-alongs are meant to be typed by the user while
following the videos. Don't pre-write course code for them — help with setup,
debugging, explanations, and note-taking instead.

## Commands

```bash
python scripts/env-check.py     # first thing to run on any new machine
pip install -r requirements.txt
pip install torch               # Mac/CPU — torch is deliberately NOT in requirements.txt
# pip install torch --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8
bash runpod/setup.sh            # fresh RunPod pod: deps + torch + vendor clones + wandb + env check (idempotent)
bash scripts/setup-local.sh     # Mac OR 5090 — detects the machine, installs matching torch (idempotent)
python scripts/gpu-smoke-test.py  # does it actually train? sm_120/bf16/compile on CUDA, MPS checks on Mac
jupyter lab                     # notebooks live in the weekN-*/ dirs
```

Both machines use the same conda env name, `llm-training`, on the same Python
3.12 (one version back, because Week 4's bitsandbytes/peft/trl/unsloth stack
trails 3.13 — and matching versions keep the notebook kernel consistent across
the sync). Activate it before doing anything: `conda activate llm-training`.

**Notebooks sync across machines**, so they're set up to merge cell-wise with
nbdime (`.gitattributes` + a driver enabled by `setup-local.sh`). Outputs are
committed deliberately — the plots and diagrams are the point. Never "fix" a
notebook conflict by hand-editing the JSON; use `nbdiff` / `nbmerge`.

Never pin or add `torch` to requirements.txt — it is installed per-machine so
each environment (Mac CPU, RunPod CUDA 12.x, RTX 5090/CUDA 12.8) gets the right build.

## Hardware environments

The user works on **two machines and syncs through this repo**, so anything you
add must run on both:

- **Apple Silicon MacBook** (MPS) — notes and the Week 1–2 code-alongs
- **RTX 5090** (Ubuntu 24.04, CUDA 12.8, `sm_120`) — the actual training runs

Plus **rented RunPod GPUs** for weeks 3–4 overflow (`runpod/setup.sh`).

Scripts must degrade gracefully across all of them — no assuming CUDA, and no
assuming MPS either. See `scripts/env-check.py` and `scripts/gpu-smoke-test.py`
for the pattern (try/except around `import torch`, branch on
`cuda.is_available()` / `backends.mps.is_available()`, fall through to CPU).

## Layout and conventions

- `notes/` — one markdown file per video/resource, numbered to match the plan
  (`01-micrograd.md` … `05-gpt2-repro.md`, plus `memory-math.md` for the Week 2
  exercise). Each has a "**▶ Resume at:**" line tracking video position and the
  sections: Key concepts / Code I wrote / Questions I had / Things that
  surprised me. Keep that structure when updating them.
- `week1-micrograd/` … `week4-finetune/` — code-along notebooks and experiments,
  one directory per curriculum week.
- `vendor/` — shallow clones of the five Karpathy reference repos (micrograd,
  makemore, ng-video-lecture, nanoGPT, build-nanogpt). Gitignored except
  `vendor/README.md`. Read-only reference material: never edit or commit them;
  `runpod/setup.sh` re-clones any that are missing.
- `data/`, `checkpoints/`, `wandb/`, and all `*.pt`/`*.bin`/`*.safetensors` are
  gitignored — large artifacts never get pushed.
