# LLM Training: 4-Week Study Plan
Goal: go from "runs local inference" to "can train and fine-tune models" by the
time I'm back with the 5090.

## Week 1 — Foundations: backprop to language models
Hardware: Mac (CPU) or free Colab

Watch (Karpathy: Neural Networks Zero to Hero):
- [ ] Course homepage: https://karpathy.ai/zero-to-hero.html
- [ ] Playlist: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
- [x] Video 1 — The spelled-out intro to neural networks and backpropagation: building micrograd (~2.5 hr)
- [ ] Video 2 — The spelled-out intro to language modeling: building makemore (~2 hr)
- [ ] Video 3 — Building makemore Part 2: MLP (~1.5 hr)

Code along:
- [x] micrograd: https://github.com/karpathy/micrograd
- [ ] makemore: https://github.com/karpathy/makemore
- [ ] Do the exercises in each video description

Optional:
- [ ] 3Blue1Brown Neural Networks: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

## Week 2 — Transformers and the memory math
Hardware: Mac / free Colab

Watch:
- [ ] Video 4 — makemore Part 3: Activations & Gradients, BatchNorm
- [ ] Video 5 — makemore Part 4: Becoming a Backprop Ninja (optional, excellent)
- [ ] Video 7 — Let's build GPT: from scratch, in code, spelled out (~2 hr, the centerpiece)
      Repo: https://github.com/karpathy/ng-video-lecture
      Type the code along, don't just watch. ~4 hr with typing.
- [ ] Video 8 — Let's build the GPT Tokenizer (~2 hr)

Read:
- [ ] Attention Is All You Need: https://arxiv.org/abs/1706.03762
- [ ] The Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
- [ ] The Illustrated GPT-2: https://jalammar.github.io/illustrated-gpt2/

Hardware/systems reading (no compute needed):
- [ ] Making Deep Learning Go Brrrr From First Principles: https://horace.io/brrr_intro.html
- [ ] Transformer Math 101 (EleutherAI): https://blog.eleuther.ai/transformer-math/
- [ ] Model Memory Utility: https://huggingface.co/spaces/hf-accelerate/model-memory-usage

Exercise:
- [ ] Compute training memory for a 7B model in fp32, bf16, and with Adam states
      (weights + gradients + optimizer + activations). Write it up in
      notes/memory-math.md. Then explain why QLoRA exists.

## Week 3 — First real training runs (RunPod)
Hardware: RunPod RTX 4090, ~$0.35-0.70/hr community cloud. Budget ~$10-15.
https://www.runpod.io

Watch:
- [ ] Video 9 — Let's reproduce GPT-2 (124M) (~4 hr, the main event: mixed
      precision, torch.compile, gradient accumulation, LR schedules)

Do:
- [ ] nanoGPT: https://github.com/karpathy/nanoGPT
- [ ] build-nanogpt (cleaned-up repo from Video 9): https://github.com/karpathy/build-nanogpt
- [ ] Dry-run locally first — train tiny Shakespeare end-to-end on Mac/Colab
      before renting anything
- [ ] Rent a 4090: train Shakespeare-GPT, then a small OpenWebText run
- [ ] Log to Weights & Biases: https://wandb.ai
- [ ] Checkpoint to a network volume or push to HF Hub — community pods get preempted

Skim:
- [ ] Ultra-Scale Playbook, intro + Data Parallelism sections:
      https://huggingface.co/spaces/nanotron/ultrascale-playbook

## Week 4 — Fine-tuning + distributed concepts
Hardware: RunPod 4090 for QLoRA; optionally one A100 80GB session (~$1.50-2/hr).
Budget ~$10-15.

Read:
- [ ] LoRA paper: https://arxiv.org/abs/2106.09685
- [ ] QLoRA paper: https://arxiv.org/abs/2305.14314
- [ ] HF PEFT docs: https://huggingface.co/docs/peft
- [ ] Unsloth docs: https://docs.unsloth.ai
- [ ] Finish Ultra-Scale Playbook: ZeRO, tensor parallelism, pipeline parallelism

Do:
- [ ] QLoRA fine-tune a 7-8B model (Llama 3.1 8B or Qwen) on a small instruction
      dataset using Unsloth or HF TRL: https://huggingface.co/docs/trl
- [ ] Optional: rent an A100 80GB for 1-2 hr, run the same fine-tune at higher
      precision, compare speed/memory
- [ ] Evaluate fine-tune vs base model on held-out prompts

Hardware deep-dive:
- [ ] Tim Dettmers — Which GPU(s) for Deep Learning:
      https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/
- [ ] SemiAnalysis (industry/silicon side): https://semianalysis.com

## After: 5090 projects
- [x] Clone this repo onto the 5090 box, run scripts/env-check.py
      (env: `conda activate llm-training`, bootstrap: `bash scripts/setup-local.sh`,
      verify: `python scripts/gpu-smoke-test.py`)
- [ ] Longer GPT-2 pretraining run (multi-day, free since I own the card)
- [ ] LoRA fine-tune on my own data
- [ ] Benchmark 5090 vs rented 4090/A100 — tokens/sec, max batch size, VRAM headroom
- [ ] Stretch: llm.c (GPT-2 in pure C/CUDA): https://github.com/karpathy/llm.c

## Quick reference: the memory equation
Training with Adam in mixed precision, per parameter:
- Weights: 2 bytes (bf16)
- Gradients: 2 bytes
- Optimizer states: 8 bytes (fp32 momentum + variance)
- ≈ 12-16 bytes/param before activations → 7B ≈ 84-112 GB → why you QLoRA on a
  24-32GB card
