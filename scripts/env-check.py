#!/usr/bin/env python3
"""Environment check — the first thing to run on any new machine.

Prints Python/torch versions, CUDA availability, device name, VRAM, CPU, and RAM.
Runs fine with no torch installed (it tells you how to install it).
"""
import os
import platform


def human_gb(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:.1f} GB"


def main() -> None:
    print("=" * 52)
    print("Environment check")
    print("=" * 52)

    print(f"Python   : {platform.python_version()}  ({platform.system()} {platform.machine()})")

    # CPU count (sched_getaffinity respects cgroup limits inside pods/containers)
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    print(f"CPU count: {cpu_count}")

    # RAM (psutil if available, else best-effort via sysconf on Unix)
    total_ram = None
    try:
        import psutil

        total_ram = psutil.virtual_memory().total
    except ImportError:
        try:
            total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, AttributeError, OSError):
            pass
    print(f"RAM      : {human_gb(total_ram) if total_ram else '(unknown — pip install psutil)'}")

    # torch / accelerators
    try:
        import torch
    except ImportError:
        print("torch    : NOT INSTALLED")
        print("           Mac / CPU:  pip install torch")
        print("           CUDA 12.8:  pip install torch --index-url https://download.pytorch.org/whl/cu128")
        print("=" * 52)
        return

    print(f"torch    : {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA     : True (cuda {torch.version.cuda})")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}  : {p.name}  ({human_gb(p.total_memory)} VRAM)")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        print("CUDA     : False")
        print("MPS      : True (Apple Silicon)")
    else:
        print("CUDA     : False  (CPU-only)")

    print("=" * 52)


if __name__ == "__main__":
    main()
