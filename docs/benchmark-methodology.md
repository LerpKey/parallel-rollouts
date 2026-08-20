# Benchmark methodology

The benchmark compares the same synthetic workload and the same batched policy
under `serial`, `sync`, and `async` execution. It reports environment
transitions per second, not just learner update speed.

Each run records:

- operating system and processor;
- Python, NumPy, and Gymnasium versions;
- number of environments;
- warmup and measured steps;
- median wall time across repeated runs.

Run from a native Windows PowerShell terminal:

```powershell
python -m benchmarks.run_rollout_benchmark --backend serial --num-envs 4
python -m benchmarks.run_rollout_benchmark --backend sync --num-envs 4
python -m benchmarks.run_rollout_benchmark --backend async --num-envs 4
```

For a less lightweight workload, increase the number of simulated users:

```powershell
python -m benchmarks.run_rollout_benchmark --backend async --num-envs 4 --num-users 64
```

The benchmark intentionally does not impose a universal speed threshold.
Process startup and IPC can dominate small workloads. A release report must
show raw measurements and speedup relative to the matching serial baseline,
and may use “faster” only for measured configurations that support it.

The timed interval starts after collector construction and the requested
warmup. It covers repeated `collector.collect()` calls only; environment
construction, worker startup, and policy training are not included. The
reported `warmup_steps_actual` is rounded up to a complete collection horizon.
This benchmark uses the repository's synthetic environment. It is not a
performance claim about a separate application environment such as DRL's
frame-buffer simulator.
