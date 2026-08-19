# Parallel Rollouts

Low-overhead rollout collection for Gymnasium environments.

Parallel Rollouts provides one small interface for collecting reinforcement
learning experience with three interchangeable execution backends:

- `serial`: a simple reference implementation;
- `sync`: multiple environments in one process;
- `async`: multiple environments in Windows-safe spawned worker processes.

The runtime is deliberately separate from any particular RL algorithm. It
supports batched policies, fixed-shape NumPy storage, optional policy values
and log probabilities, deterministic seed allocation, and explicit handling
of terminal observations.

## Quick start on Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,torch]"
python -m examples.vr_edge.train --backend sync --num-envs 4
python -m benchmarks.run_rollout_benchmark --backend async --num-envs 4
```

The `async` backend uses Python's `spawn` context. Training and benchmark
entrypoints include the required Windows `__main__` guard and
`freeze_support()` call. Environment factories must be top-level or otherwise
pickleable; do not pass lambdas or nested functions to the async backend.

## Minimal API

```python
from functools import partial

from parallel_rollouts import ParallelRolloutCollector
from examples.vr_edge.synthetic_env import SyntheticVREdgeEnv

env_fns = [
    partial(SyntheticVREdgeEnv, num_users=8, seed=42 + i)
    for i in range(4)
]

policy = ...  # callable: observations [env, ...] -> actions [env, ...]

with ParallelRolloutCollector(
    env_fns=env_fns,
    policy=policy,
    backend="async",
    horizon=128,
    seed=42,
) as collector:
    batch = collector.collect()
```

`batch.observations`, `batch.actions`, and `batch.rewards` use leading
dimensions `[time, env, ...]`. The `batch.final_observation_mask` identifies
which entries have a terminal observation in `batch.final_observations`.

## Application example

`examples/vr_edge/` contains a fully synthetic multi-user edge-rendering
workload. Users share fluctuating bandwidth, choose bitrate levels, and receive
a quality/stall/smoothness reward. It has no ALVR, MyGO, MTLALVR2, private
trajectory, checkpoint, or external dataset dependency.

The example is intentionally an application demonstration, not a claim that
the package introduces a new RL algorithm. The included PPO trainer is small
enough to inspect and exists to verify end-to-end integration.

## Benchmarking

Run the same workload through each backend:

```powershell
python -m benchmarks.run_rollout_benchmark --backend serial --num-envs 4
python -m benchmarks.run_rollout_benchmark --backend sync --num-envs 4
python -m benchmarks.run_rollout_benchmark --backend async --num-envs 4
```

The benchmark reports transitions per second and records the Python, NumPy,
Gymnasium, CPU, and optional CUDA environment. Results are machine-dependent:
for small environments, process communication can make `async` slower than
`sync`. The project only claims acceleration for configurations where the
measured result supports it.

## Project status

This is an alpha systems component. The public maintenance story is summarized
in [`docs/maintenance-history.md`](docs/maintenance-history.md); private
research logs, raw traces, checkpoints, paper drafts, and intermediate reports
are intentionally not part of this repository.

## License

MIT. See [`LICENSE`](LICENSE).
