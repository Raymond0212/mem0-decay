# Benchmark integration

Make integration changes on `mem0-decay-py` in the parent repository. Treat the benchmark repository, the `benchmark/` submodule, its pinned commit and `.gitmodules` as read-only.

Use the benchmark's `main` branch to verify Option B compatibility. Run its CLI from a separate writable directory so datasets, logs, bytecode, checkpoints and results stay outside the benchmark checkout. The README includes the commands.
