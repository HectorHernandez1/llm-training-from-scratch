# vendor/ — cloned reference repos

Karpathy reference repos I code along with. **These are gitignored** (only this
note is tracked) — clone them locally, they never get pushed.

| Directory | Source | Week | Runs on Mac (CPU)? |
|-----------|--------|------|--------------------|
| `micrograd` | https://github.com/karpathy/micrograd | 1 | yes — pure Python |
| `makemore` | https://github.com/karpathy/makemore | 1–2 | yes — tiny |
| `ng-video-lecture` | https://github.com/karpathy/ng-video-lecture | 2 | yes — char-level GPT, slow but fine |
| `nanoGPT` | https://github.com/karpathy/nanoGPT | 3 | dry-run tiny Shakespeare only; real runs want a GPU |
| `build-nanogpt` | https://github.com/karpathy/build-nanogpt | 3 | same — clones anywhere, full run wants a GPU |

Cloning needs no GPU; RunPod only matters for the actual training runs.

## Getting them
On a fresh machine, `runpod/setup.sh` clones all five (idempotent — skips any
already present). To clone locally by hand:

```bash
cd vendor
git clone https://github.com/karpathy/micrograd.git
git clone https://github.com/karpathy/makemore.git
git clone https://github.com/karpathy/ng-video-lecture.git
git clone https://github.com/karpathy/nanoGPT.git
git clone https://github.com/karpathy/build-nanogpt.git
```
