# memory-math

> Maps to: Week 2 exercise. Reading:
> - Transformer Math 101 — https://blog.eleuther.ai/transformer-math/
> - Making Deep Learning Go Brrrr — https://horace.io/brrr_intro.html
> - Model Memory Utility — https://huggingface.co/spaces/hf-accelerate/model-memory-usage

## The exercise
Compute training memory for a 7B model in fp32, bf16, and with Adam optimizer
states (weights + gradients + optimizer + activations). Fill in the tables below,
then explain in my own words why QLoRA exists.

## Per-parameter cost (mixed-precision + Adam)
| Component        | Bytes/param | Notes                |
|------------------|-------------|----------------------|
| Weights (bf16)   | 2           |                      |
| Gradients (bf16) | 2           |                      |
| Adam m (fp32)    | 4           | first moment         |
| Adam v (fp32)    | 4           | second moment        |
| **Subtotal**     | **~12**     | before activations   |

## 7B worked example
| Precision / setup | Weights | Grads | Optimizer | Total (ex-activations) |
|-------------------|---------|-------|-----------|------------------------|
| fp32              |         |       |           |                        |
| bf16 (mixed)      |         |       |           |                        |
| bf16 + Adam       |         |       |           |                        |

## Key concepts

## Questions I had

## Things that surprised me

## Why QLoRA exists
<!-- one paragraph, in my own words -->
