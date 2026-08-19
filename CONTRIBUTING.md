# Contributing

1. Create a virtual environment with Python 3.10 or newer.
2. Install development dependencies with `python -m pip install -e ".[dev,torch]"`.
3. Run `python -m pytest` before submitting a change.
4. Keep the collector backend independent of application-specific environments.
5. Add benchmark evidence when changing the rollout hot path, but do not turn
   machine-dependent throughput into a regular CI assertion.

On Windows, keep multiprocessing entrypoints behind an
`if __name__ == "__main__":` guard and use pickleable environment factories.

