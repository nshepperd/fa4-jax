"""Vendored subset of quack-kernels 0.6.4 (Dao-AILab/quack, Apache-2.0; see
LICENSE.quack at the repo root): the CuTe DSL copy/layout/SM90 helpers and
ParamsBase that the vendored flash-attention kernels import. Vendored rather
than depended on because quack-kernels pins nvidia-cutlass-dsl==4.6.2
(cutejax needs >=4.8) and requires torch. Each file's header says what was
changed."""
