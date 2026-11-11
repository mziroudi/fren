# Hardware

**This is the owner's own PC** (a local iMac, not a cloud container). The routing profile
was collected here. Routing depends on model + input, not hardware, so the profile also
holds for other machines running the same model file.

Detected 2026-11-11 with `sw_vers`, `sysctl`, `system_profiler`, `diskutil`.

| Item | Value |
|---|---|
| Machine | iMac (iMac17,1, late 2015) |
| OS | macOS 14.8.9 (23J631), x86_64 |
| CPU | Intel Core i5-6500 @ 3.20 GHz, 4 cores / 4 threads, AVX2 + FMA + F16C, no AVX-512 |
| RAM | 16 GB |
| GPU | AMD Radeon R9 M380, 2 GB VRAM, Metal 2 |
| Disk | Samsung SSD 870 QVO 1 TB (SATA 6 Gb/s, QLC NAND) |
| Free disk | 28.6 GB at detection time |

Notes:
- The 2 GB AMD GPU is too small to matter and is not usable from PyTorch on macOS/Intel.
  All inference runs on the CPU.
- The SSD is a SATA QLC drive (~550 MB/s sequential at best, much lower for random reads
  after its SLC cache is full). On this machine "SSD tier" means slow reads.
- With 28.6 GB free, downloads must stay small.

## Model choice

**OLMoE-1B-7B-0125-Instruct** (allenai), **Q8_0 GGUF, run with llama.cpp on CPU**.

- License: Apache-2.0 (checked on the Hugging Face model card,
  `allenai/OLMoE-1B-7B-0125-Instruct`).
- File: `allenai/OLMoE-1B-7B-0125-Instruct-GGUF` / `OLMoE-1B-7B-0125-Instruct-Q8_0.gguf`,
  7,359,942,656 bytes (7.4 GB, checked with an HTTP HEAD request before download).
- Architecture (read from the GGUF metadata, see `results/model_meta.txt`): see the
  "Model" section of `results/WEEK1.md`.

Why this model and this route:
- Qwen3-30B-A3B is out: even at Q4 it is ~18 GB, more than the 16 GB of RAM and close to
  the free disk space.
- CLAUDE.md suggests OLMoE via transformers on CPU for small machines. But in bf16 it needs
  ~14 GB of RAM for the weights alone, which on a 16 GB Mac means heavy swapping. Q8_0 in
  llama.cpp needs ~7.4 GB and is near-lossless, and llama.cpp is the engine the project
  builds on anyway, so the capture tool is reusable later.
- Q8_0 rather than Q4: quantization can flip near-tied router decisions. Q8_0 keeps the
  routing as close to the original weights as fits on this disk.

Caveat: all results are for this model (OLMoE-1B-7B-0125-Instruct, Q8_0). Other MoE models
(different expert counts, top-k, training) may route very differently.
