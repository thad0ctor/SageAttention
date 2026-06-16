# NVFP4 training-attention perf harnesses (perf/nvfp4-train-speed)

Profile-first instruments for the native NVFP4 attention fwd+bwd path on the
RTX 5090 (sm_120). Real Qwen3.5-9B full-attn geometry: d256, h16/hk4 (GQA g4),
packed varlen (z=1, S=8192, ragged cu_seqlens) — the shape the e2e axolotl
8192-token pack feeds the kernel.

Run on idx6 only:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=<fork> <venv>/bin/python scripts/<harness>.py

- prof_train_attn.py : per-CUDA-kernel fwd/bwd breakdown (the profile).
- sweep_bwd_tiles.py : NVFP4_BWD_TILE_OVERRIDE sweep of dkdv/dq tiles.
- ab_dq_tile.py      : interleaved A/B, dq-recompute BM=64 vs 32 (isolated kernel).
- ab_dq_mode.py      : dq recompute vs dscache at varlen geometry.
