# llm-training-from-scratch

Working through LLM training from first principles — backprop → language models →
transformers → GPT-2 pretraining → QLoRA fine-tuning. This repo is my lab
notebook: notes, code-alongs, training runs, and hardware benchmarks across
several environments (an Apple Silicon laptop, rented GPUs, and a local RTX
5090). The full curriculum lives in **[PLAN.md](PLAN.md)**.

## Hardware I'm running on
I work on two machines and sync through this repo, so everything here has to run
on both:

- **Apple Silicon MacBook** (10-core, 24 GB RAM, MPS) — reading, notes, and the
  Week 1–2 code-alongs, which are small enough to not need a real GPU
- **RTX 5090 box** (32 GB, Ubuntu 24.04, CUDA 12.8) — every actual training run

Plus, when a run needs more card than I own: **rented RunPod GPUs** (RTX 4090
~$0.50/hr, occasionally an A100 80GB) — see `runpod/setup.sh`.

## Quick start
Same three commands on the Mac and on the 5090 box — `setup-local.sh` detects
which machine it's on and installs the matching torch. It's idempotent, so
re-running it is always safe:

```bash
git clone https://github.com/HectorHernandez1/llm-training-from-scratch.git
cd llm-training-from-scratch

conda create -n llm-training python=3.12 -y && conda activate llm-training
bash scripts/setup-local.sh
```

That installs deps, the right torch, the five `vendor/` reference repos, the
graphviz binary, a shared `llm-training` Jupyter kernel, and the notebook merge
driver — then runs both checks. On a rented pod use `bash runpod/setup.sh`.

To do it by hand instead (torch is installed per-machine — see
`requirements.txt`):

```bash
pip install -r requirements.txt
pip install torch                                                      # Mac (MPS) / CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu128 # 5090 / CUDA

python scripts/env-check.py        # what hardware is this?
python scripts/gpu-smoke-test.py   # does it actually train?
```

### Working across both machines
Everything that matters is in git — notes, notebooks, `PLAN.md`, the run log.
`data/`, `checkpoints/`, `wandb/` and `vendor/` are gitignored and stay local, so
each machine re-downloads datasets and re-clones the reference repos once.

**Pull before you open a notebook, commit before you walk away.** Notebooks are
the one thing that genuinely conflicts across machines: re-running a cell bumps
execution counts and rewrites output blobs, so a plain text merge reports a
conflict on cells that didn't meaningfully change. `nbdime` handles this by
merging cell-wise — `.gitattributes` ships the mapping, and `setup-local.sh`
enables the driver on each machine. Outputs stay committed on purpose: the
plots, loss curves and graphviz diagrams *are* the lab notebook.

If a notebook does conflict, don't hand-edit the JSON:
```bash
nbdiff HEAD notebooks/whatever.ipynb   # readable cell-level diff
nbmerge base.ipynb local.ipynb remote.ipynb -o merged.ipynb
```

### Two things that bite, both silent until they aren't
- **The RTX 5090 is Blackwell (`sm_120`).** A default-index torch wheel can
  import fine and report `cuda.is_available() == True`, then die on the first
  real matmul with *"no kernel image is available for execution on the device."*
  Install from the **cu128** index. `gpu-smoke-test.py` checks the arch list up
  front so you find out in a second rather than mid-run.
- **Python 3.12, not 3.13.** Week 4's QLoRA stack (bitsandbytes, peft, trl,
  unsloth) still trails on 3.13, so both machines are pinned a version back —
  matching Python versions also keeps the notebook kernel consistent.

Full 4-week curriculum: **[PLAN.md](PLAN.md)**.

## Repo layout
```
PLAN.md             # the 4-week study plan — the spine of the repo
notes/              # per-topic notes, one file per video/resource
  memory-math.md    #   the Week 2 memory-budget exercise
week1-micrograd/    # micrograd + makemore code-alongs (Videos 1–3)
week2-makemore/     # activations/BatchNorm, build-GPT, tokenizer (Videos 4–8)
week3-gpt/          # nanoGPT / build-nanogpt training runs (Video 9)
week4-finetune/     # LoRA / QLoRA fine-tuning experiments
runpod/             # setup.sh bootstrap + pod configs (rented pods)
scripts/            # setup-local.sh (Mac + 5090), env-check.py, gpu-smoke-test.py
.gitattributes      # nbdime cell-wise notebook merges — see "Working across both machines"
vendor/             # cloned reference repos (gitignored)
data/               # datasets (gitignored)
checkpoints/        # model weights (gitignored)
```

## Run log
The important part — one entry per training run so I can look back and see what I
actually did, what it cost, and what I learned. Template:

### YYYY-MM-DD — &lt;what I ran&gt;
- **Hardware:**
- **Config:**
- **Result:**
- **What broke:**
- **What I learned:**
- **Cost:**

---

### 2026-07-15 — placeholder (repo scaffolded)
- **Hardware:** Apple Silicon MacBook, CPU/MPS
- **Config:** none yet — repo initialized, `env-check.py` passing
- **Result:** —
- **What broke:** —
- **What I learned:** set up the lab notebook; ready to start Week 1
- **Cost:** $0

### 2026-08-08 — 5090 box brought online + repo made two-machine (no training yet)
- **Hardware:** RTX 5090 32 GB, Ubuntu 24.04, driver 595.58.03, CUDA 12.8 toolkit,
  32 CPU / 60 GB RAM
- **Config:** conda env `llm-training` (Python 3.12), torch 2.11.0+cu128 from the
  cu128 index, `requirements.txt`, all five vendor repos cloned,
  graphviz 14.1.2 from conda-forge, Jupyter kernel `llm-training` registered.
  `scripts/setup-local.sh` now detects Mac-vs-5090 and installs the matching
  torch, so the same three commands bootstrap either machine; nbdime wired up
  for cell-wise notebook merges across the sync.
- **Result:** `env-check.py` and `gpu-smoke-test.py` both pass — sm_120 kernels
  present, bf16 + autocast, TF32, torch.compile (Triton) all working.
  **~210 TFLOP/s dense bf16** on an 8192² matmul (207–213 across three runs);
  23.7 GB VRAM free with the desktop running. This is the baseline to compare
  rented 4090/A100 against.
- **What broke:** nothing in the training stack, but three environment traps:
  (1) no passwordless sudo on this box, so graphviz came from conda-forge rather
  than `apt` — same `dot` binary, no password needed;
  (2) `ipykernel install --user` run from inside VS Code lands in the snap
  sandbox (`~/snap/code/240/.local/share/...`), which `jupyter lab` from a normal
  terminal can't see — `setup-local.sh` now forces `XDG_DATA_HOME=$HOME/.local/share`;
  (3) benchmarking with `time.perf_counter` instead of `torch.cuda.Event` reads
  ~8% lower (229 → 210 TFLOP/s) because it includes host-side overhead. Kept
  perf_counter since it works on MPS too — just don't compare across methods.
- **What I learned:** the Blackwell failure mode is nastier than a plain error —
  a wrong-index torch wheel imports fine and reports `cuda.is_available() ==
  True`, then dies at the first matmul with "no kernel image is available for
  execution on the device". Checking `torch.cuda.get_arch_list()` for `sm_120`
  catches it in a second instead of mid-run. Pinned Python to 3.12 rather than
  base's 3.13 because Week 4's bitsandbytes/peft/trl/unsloth stack still trails.
- **Cost:** $0 (own hardware)
