# Releasing fa4-jax

Tags drive the version (hatch-vcs). Pushing a `v*` tag runs
`.github/workflows/release.yml`: build the wheel and sdist (checked against
the tag), publish to PyPI with trusted publishing, then create a GitHub
release with both attached. The workflow runs no tests — do that first.

## Checklist

1. **Test what users will install.** `uv lock --upgrade` resolves the
   latest compatible dependencies, i.e. what a fresh
   `pip install "fa4-jax[cu13]"` gets; commit the lock if it moved.

2. **Local GPUs**: `uv run pytest` on each local machine (e.g. sm_120 and
   sm_86).

3. **Cross-compile for the GPUs you don't have** (free, any GPU box). Builds
   every kernel for sm_90a/sm_100a without launching anything, so it catches
   tracing and compile errors — e.g. a new cutlass-dsl changing an API:

   ```sh
   uv run python ci/cross_compile.py 90 && uv run python ci/cross_compile.py 100
   ```

4. **Real hardware**: the full suite on rented RunPod H100 + B200 pods, at
   the exact commit being tagged (it refuses a dirty tree), with the API key
   injected by keyward. Roughly an hour and ~$10:

   ```sh
   kw run -e RUNPOD_API_KEY=RUNPOD_API_KEY -- python3 ci/runpod_ci.py
   ```

   Results land in `ci-results/`. Pods are named `fa4-ci-*` and terminated on
   exit; `--cleanup` sweeps any strays. `--only sm86` is a cheap A40 target
   for debugging the script itself.

5. **Tag and push**:

   ```sh
   git push origin main
   git tag -a v0.2.0 -m "fa4-jax 0.2.0" && git push origin v0.2.0
   ```

   Watch the run (`gh run watch`); a tag containing `a`, `b`, `rc` or `dev`
   is marked as a pre-release on GitHub.
