"""Vendored FlashAttention-4 CuTe DSL kernels (from the flash-attn-4 wheel).

See VENDOR.md in this directory for provenance and local changes. The
torch-facing `interface.py` is intentionally not vendored; fa4-jax drives the
kernel classes directly through cutejax. Importing this package requires
nvidia-cutlass-dsl but neither torch nor a GPU.
"""
