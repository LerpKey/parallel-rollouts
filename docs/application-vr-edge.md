# Synthetic VR edge application

The example workload represents multiple clients sharing a fluctuating edge
link. Each client selects a bitrate. The environment tracks bandwidth, buffer
occupancy, current bitrate, and a compact user/time state. Reward combines
quality, stall avoidance, and bitrate smoothness.

The workload is generated from a seed and does not load ALVR, MyGO, MTLALVR2,
real user trajectories, checkpoints, or private telemetry. The small JSON
fixture in `tests/fixtures/synthetic_trace.json` is illustrative only; the
environment itself generates its trajectory at runtime.

The purpose of the example is to demonstrate how a domain environment plugs
into the generic collector. It is not a claim that this repository contains a
new VR streaming algorithm.

