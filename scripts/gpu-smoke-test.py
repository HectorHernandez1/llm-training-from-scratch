#!/usr/bin/env python3
"""Accelerator smoke test — does this box actually train, not just import torch?

`env-check.py` tells you what hardware you have. This tells you whether the
stack on top of it works. Runs on both my machines and reports what's relevant
to each:

  RTX 5090 (CUDA)     sm_120 kernels, bf16, TF32, torch.compile, TFLOP/s
  Apple Silicon (MPS) MPS backend, fp16/bf16 support, GFLOP/s

The Blackwell gotcha this exists to catch: a torch build with no sm_120 kernels
imports fine and reports `cuda.is_available() == True`, then dies on the first
real matmul with "no kernel image is available for execution on the device".
That means the wrong wheel — reinstall from the cu128 index.

Exits 0 on a healthy box (including a plain CPU one), 1 only when something is
genuinely broken.
"""
import sys

PASS, FAIL, SKIP, WARN = "  ok  ", " FAIL ", " skip ", " warn "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label}" + (f"  —  {detail}" if detail else ""))


def bench_matmul(torch, dev, n: int, dtype) -> float:
    """Sustained matmul throughput in TFLOP/s, after warming up clocks."""
    x = torch.randn(n, n, device=dev, dtype=dtype)
    y = torch.randn(n, n, device=dev, dtype=dtype)
    sync = torch.cuda.synchronize if dev.type == "cuda" else torch.mps.synchronize
    for _ in range(3):
        x @ y
    sync()
    import time

    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        x @ y
    sync()
    secs = (time.perf_counter() - start) / iters
    return (2 * n**3) / secs / 1e12


def check_cuda(torch) -> int:
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    cap = f"sm_{props.major}{props.minor}"
    line(PASS, "CUDA device", f"{props.name} ({cap}, {props.total_memory / 1024**3:.1f} GB)")

    # 1. Kernels compiled for THIS architecture. The one that bites on new cards.
    arch_list = torch.cuda.get_arch_list()
    if any(a.startswith(f"sm_{props.major}{props.minor}") for a in arch_list):
        line(PASS, "kernel support", f"{cap} in torch arch list")
    else:
        line(FAIL, "kernel support", f"{cap} NOT in {arch_list}")
        print("\n  This torch has no kernels for your GPU. Reinstall:")
        print("    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128\n")
        return 1

    failures = 0

    # 2. fp32 matmul — proves kernels launch, and the math is right.
    try:
        a = torch.randn(512, 512, device=dev)
        got = (a @ torch.eye(512, device=dev) - a).abs().max().item()
        assert got < 1e-3, f"identity matmul off by {got}"
        line(PASS, "fp32 matmul", "kernels launch, result correct")
    except Exception as e:  # noqa: BLE001 — smoke test reports, never raises
        line(FAIL, "fp32 matmul", str(e)[:80])
        failures += 1

    # 3. bf16 — the dtype every mixed-precision run in weeks 3-4 uses.
    if torch.cuda.is_bf16_supported():
        try:
            x = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = (x @ x).float().sum().item()
            line(PASS, "bf16 + autocast", "supported")
        except Exception as e:  # noqa: BLE001
            line(FAIL, "bf16 + autocast", str(e)[:80])
            failures += 1
    else:
        line(WARN, "bf16", "not supported on this card — use fp16 instead")

    # 4. TF32 — free speedup on fp32 matmuls; nanoGPT/build-nanogpt turn it on.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    line(PASS, "TF32", "enabled (torch.backends.cuda.matmul.allow_tf32)")

    # 5. torch.compile — needs a working Triton; the usual week-3 blocker.
    try:
        compiled = torch.compile(lambda t: t * 2 + 1)
        out = compiled(torch.randn(64, 64, device=dev))
        torch.cuda.synchronize()
        assert out.shape == (64, 64)
        line(PASS, "torch.compile", "Triton backend works")
    except Exception as e:  # noqa: BLE001
        line(WARN, "torch.compile", f"{str(e)[:70]} (training still works, just slower)")

    # 6. Throughput baseline — the number to compare a rented 4090/A100 against.
    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tflops = bench_matmul(torch, dev, 8192, dtype)
        line(PASS, "matmul throughput", f"{tflops:.1f} TFLOP/s dense {str(dtype).split('.')[-1]} (8192x8192)")
    except Exception as e:  # noqa: BLE001
        line(WARN, "matmul throughput", str(e)[:80])

    # 7. Free VRAM — how big a batch/model you can actually hold right now.
    free, total = torch.cuda.mem_get_info()
    line(PASS, "VRAM headroom", f"{free / 1024**3:.1f} GB free of {total / 1024**3:.1f} GB")
    return failures


def check_mps(torch) -> int:
    dev = torch.device("mps")
    line(PASS, "MPS device", "Apple Silicon GPU (unified memory)")
    failures = 0

    # 1. fp32 matmul — proves the Metal kernels actually run.
    try:
        a = torch.randn(512, 512, device=dev)
        got = (a @ torch.eye(512, device=dev) - a).abs().max().item()
        assert got < 1e-3, f"identity matmul off by {got}"
        line(PASS, "fp32 matmul", "kernels launch, result correct")
    except Exception as e:  # noqa: BLE001
        line(FAIL, "fp32 matmul", str(e)[:80])
        failures += 1

    # 2. Half precision. fp16 is solid on MPS; bf16 needs M2+ / macOS 14+, so a
    #    failure here is a capability limit, not a broken install.
    for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        try:
            x = torch.randn(256, 256, device=dev, dtype=dtype)
            _ = (x @ x).float().sum().item()
            line(PASS, f"{name} matmul", "supported")
        except Exception as e:  # noqa: BLE001
            line(WARN, f"{name} matmul", f"unsupported on this chip/OS ({str(e)[:50]})")

    # 3. torch.compile on MPS is still rough — never fail the run over it.
    try:
        compiled = torch.compile(lambda t: t * 2 + 1)
        out = compiled(torch.randn(64, 64, device=dev))
        torch.mps.synchronize()
        assert out.shape == (64, 64)
        line(PASS, "torch.compile", "works on MPS")
    except Exception as e:  # noqa: BLE001
        line(WARN, "torch.compile", f"{str(e)[:60]} (expected on MPS; eager is fine)")

    # 4. Throughput. Smaller matrix than CUDA — this is a laptop, not the 5090.
    try:
        tflops = bench_matmul(torch, dev, 4096, torch.float32)
        line(PASS, "matmul throughput", f"{tflops * 1000:.0f} GFLOP/s fp32 (4096x4096)")
    except Exception as e:  # noqa: BLE001
        line(WARN, "matmul throughput", str(e)[:80])

    # 5. Unified memory — the GPU shares RAM with the OS, so this is a ceiling
    #    on model+batch size, not a dedicated pool like VRAM.
    try:
        rec = torch.mps.recommended_max_memory() / 1024**3
        used = torch.mps.driver_allocated_memory() / 1024**3
        line(PASS, "unified memory", f"{used:.1f} GB allocated, ~{rec:.1f} GB recommended max")
    except Exception as e:  # noqa: BLE001
        line(WARN, "unified memory", str(e)[:80])
    return failures


def main() -> int:
    print("=" * 62)
    print("Accelerator smoke test")
    print("=" * 62)

    try:
        import torch
    except ImportError:
        line(FAIL, "import torch", "not installed — see scripts/env-check.py")
        return 1

    build = torch.version.cuda or ("mps" if torch.backends.mps.is_built() else "cpu")
    line(PASS, "import torch", f"{torch.__version__} (build: {build})")

    if torch.cuda.is_available():
        failures = check_cuda(torch)
    elif torch.backends.mps.is_available():
        failures = check_mps(torch)
    else:
        why = "torch built without MPS" if sys.platform == "darwin" else "no CUDA device"
        line(SKIP, "accelerator", f"none available ({why}) — CPU only, nothing further to test")
        failures = 0

    print("=" * 62)
    print("Ready to train." if not failures else f"{failures} check(s) FAILED — see above.")
    print("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
