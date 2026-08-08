# vendor/ — cloned reference repos

Karpathy reference repos I code along with. **These are gitignored** (only this
note is tracked) — clone them locally, they never get pushed.

| Directory | Source | Week | Runs on the Mac? |
|-----------|--------|------|------------------|
| `micrograd` | https://github.com/karpathy/micrograd | 1 | yes — pure Python, no torch at all |
| `makemore` | https://github.com/karpathy/makemore | 1–2 | yes — tiny |
| `ng-video-lecture` | https://github.com/karpathy/ng-video-lecture | 2 | yes — char-level GPT, slow but fine on MPS |
| `nanoGPT` | https://github.com/karpathy/nanoGPT | 3 | dry-run tiny Shakespeare only; real runs want the 5090 |
| `build-nanogpt` | https://github.com/karpathy/build-nanogpt | 3 | same — clones anywhere, full run wants the 5090 |

Cloning needs no GPU — only the actual training runs care.

## Getting them
These are **gitignored, so they don't sync between machines** — each box clones
its own copy. The bootstrap scripts do it for you (idempotent, skipping any
already present): `scripts/setup-local.sh` on the Mac or the 5090 box,
`runpod/setup.sh` on a rented pod. To clone by hand:

```bash
cd vendor
git clone https://github.com/karpathy/micrograd.git
git clone https://github.com/karpathy/makemore.git
git clone https://github.com/karpathy/ng-video-lecture.git
git clone https://github.com/karpathy/nanoGPT.git
git clone https://github.com/karpathy/build-nanogpt.git
```
