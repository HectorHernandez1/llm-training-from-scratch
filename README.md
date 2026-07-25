# llm-training-from-scratch

Working through LLM training from first principles — backprop → language models →
transformers → GPT-2 pretraining → QLoRA fine-tuning. This repo is my lab
notebook: notes, code-alongs, training runs, and hardware benchmarks across three
environments (CPU laptop, rented GPUs, and a local RTX 5090). The full curriculum
lives in **[PLAN.md](PLAN.md)**.

## Hardware I'm running on
- **Weeks 1–2** — Apple Silicon MacBook (10-core, 24 GB RAM, MPS available) or free Colab
- **Weeks 3–4** — rented RunPod GPUs (RTX 4090 ~$0.50/hr, occasionally an A100 80GB)
- **After** — my own box: RTX 5090, Ubuntu 24.04, CUDA 12.8

## Quick start
```bash
git clone https://github.com/HectorHernandez1/llm-training-from-scratch.git
cd llm-training-from-scratch

# 1. See what you're running on
python scripts/env-check.py

# 2. Install deps (torch is installed per-machine — see requirements.txt)
pip install -r requirements.txt
pip install torch                                                      # Mac / CPU
# pip install torch --index-url https://download.pytorch.org/whl/cu128 # CUDA 12.8

# On a fresh RunPod pod, skip the above and just run:
#   bash runpod/setup.sh
```

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
runpod/             # setup.sh bootstrap + pod configs
scripts/            # env-check.py and other utilities
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
- **Hardware:** 2020 Intel MacBook, CPU
- **Config:** none yet — repo initialized, `env-check.py` passing
- **Result:** —
- **What broke:** —
- **What I learned:** set up the lab notebook; ready to start Week 1
- **Cost:** $0
