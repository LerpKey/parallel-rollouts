"""Tiny PyTorch actor-critic used by the end-to-end example."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.distributions import Categorical

from parallel_rollouts.types import PolicyOutput, RolloutBatch


class ActorCriticPolicy(nn.Module):
    """Independent categorical bitrate heads sharing one observation encoder."""

    def __init__(self, observation_dim: int, num_users: int, num_actions: int = 4) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_actions = num_actions
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, num_users * num_actions)
        self.value_head = nn.Linear(128, 1)

    def _forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observations)
        logits = self.policy_head(features).view(-1, self.num_users, self.num_actions)
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def __call__(self, observations: npt.NDArray[np.float32]) -> PolicyOutput:
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            logits, values = self._forward(tensor)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions).sum(dim=-1)
        return PolicyOutput(
            actions=actions.cpu().numpy().astype(np.int64),
            log_probs=log_probs.cpu().numpy().astype(np.float32),
            values=values.cpu().numpy().astype(np.float32),
        )

    @torch.no_grad()
    def value(self, observations: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        tensor = torch.as_tensor(observations, dtype=torch.float32)
        _, values = self._forward(tensor)
        return values.cpu().numpy().astype(np.float32)

    def evaluate(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self._forward(observations)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probs, values, entropy


def compute_gae(
    batch: RolloutBatch,
    next_values: npt.NDArray[np.float32],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute GAE without allocating Python lists in the time loop."""

    horizon, num_envs = batch.rewards.shape
    advantages = np.zeros((horizon, num_envs), dtype=np.float32)
    last = np.zeros(num_envs, dtype=np.float32)
    for step in range(horizon - 1, -1, -1):
        next_value = next_values if step == horizon - 1 else batch.values[step + 1]
        not_done = 1.0 - batch.dones[step].astype(np.float32)
        delta = batch.rewards[step] + gamma * next_value * not_done - batch.values[step]
        last = delta + gamma * gae_lambda * not_done * last
        advantages[step] = last
    return advantages, advantages + batch.values


class PPOTrainer:
    """Minimal PPO update loop for the demonstration, not the package core."""

    def __init__(
        self,
        policy: ActorCriticPolicy,
        learning_rate: float = 3e-4,
        clip_range: float = 0.2,
        entropy_coef: float = 0.01,
    ) -> None:
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef

    def update(self, batch: RolloutBatch, epochs: int = 4, minibatch_size: int = 256) -> dict[str, float]:
        observations = torch.as_tensor(batch.observations.reshape(-1, batch.observations.shape[-1]))
        actions = torch.as_tensor(batch.actions.reshape(-1, batch.actions.shape[-1]), dtype=torch.long)
        old_log_probs = torch.as_tensor(batch.log_probs.reshape(-1))
        next_values = self.policy.value(batch.next_observations[-1])
        advantages, returns = compute_gae(batch, next_values)
        advantages_t = torch.as_tensor(advantages.reshape(-1))
        returns_t = torch.as_tensor(returns.reshape(-1))
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0
        total_updates = 0
        sample_count = observations.shape[0]
        for _ in range(epochs):
            permutation = torch.randperm(sample_count)
            for start in range(0, sample_count, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                log_probs, values, entropy = self.policy.evaluate(observations[indices], actions[indices])
                ratio = torch.exp(log_probs - old_log_probs[indices])
                unclipped = ratio * advantages_t[indices]
                clipped = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_t[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = 0.5 * (returns_t[indices] - values).pow(2).mean()
                loss = policy_loss + value_loss - self.entropy_coef * entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                total_loss += float(loss.detach())
                total_updates += 1
        return {"loss": total_loss / max(total_updates, 1), "mean_reward": float(batch.rewards.mean())}
