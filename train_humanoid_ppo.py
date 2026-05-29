import argparse
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


EPS = 1e-6


class RunningMeanStd:
    def __init__(self, shape: tuple[int, ...]):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_low: np.ndarray, act_high: np.ndarray, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        action_scale = (act_high - act_low) / 2.0
        action_bias = (act_high + act_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def dist(self, obs: torch.Tensor) -> Normal:
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def action_from_raw(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action) * self.action_scale + self.action_bias

    def log_prob_from_raw(self, dist: Normal, raw_action: torch.Tensor) -> torch.Tensor:
        squashed = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - squashed.pow(2) + EPS)
        return log_prob.sum(dim=-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.dist(obs)
        raw_action = dist.mean if deterministic else dist.rsample()
        action = self.action_from_raw(raw_action)
        log_prob = self.log_prob_from_raw(dist, raw_action)
        value = self.value(obs)
        return action, raw_action, log_prob, value

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.dist(obs)
        log_prob = self.log_prob_from_raw(dist, raw_action)
        entropy = dist.entropy().sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value


@dataclass
class Rollout:
    obs: torch.Tensor
    raw_actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


def device_from_arg(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_rollout(env, model, obs_rms, obs, args, device) -> tuple[Rollout, np.ndarray, dict]:
    obs_buf = []
    raw_action_buf = []
    log_prob_buf = []
    reward_buf = []
    done_buf = []
    value_buf = []
    episode_returns = []
    episode_lengths = []
    current_return = 0.0
    current_length = 0

    for _ in range(args.steps_per_update):
        obs_rms.update(obs)
        norm_obs = obs_rms.normalize(obs)
        obs_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, raw_action, log_prob, value = model.act(obs_tensor)

        next_obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated

        obs_buf.append(norm_obs)
        raw_action_buf.append(raw_action.squeeze(0).cpu().numpy())
        log_prob_buf.append(log_prob.item())
        reward_buf.append(float(reward))
        done_buf.append(float(done))
        value_buf.append(value.item())

        current_return += float(reward)
        current_length += 1
        obs = next_obs

        if done:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            current_return = 0.0
            current_length = 0
            obs, _ = env.reset()

    obs_rms.update(obs)
    next_obs_tensor = torch.as_tensor(obs_rms.normalize(obs), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        next_value = model.value(next_obs_tensor).item()

    rewards = np.asarray(reward_buf, dtype=np.float32)
    dones = np.asarray(done_buf, dtype=np.float32)
    values = np.asarray(value_buf + [next_value], dtype=np.float32)
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for step in reversed(range(args.steps_per_update)):
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + args.gamma * values[step + 1] * nonterminal - values[step]
        gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
        advantages[step] = gae
    returns = advantages + values[:-1]

    rollout = Rollout(
        obs=torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32, device=device),
        raw_actions=torch.as_tensor(np.asarray(raw_action_buf), dtype=torch.float32, device=device),
        log_probs=torch.as_tensor(np.asarray(log_prob_buf), dtype=torch.float32, device=device),
        returns=torch.as_tensor(returns, dtype=torch.float32, device=device),
        advantages=torch.as_tensor(advantages, dtype=torch.float32, device=device),
        values=torch.as_tensor(values[:-1], dtype=torch.float32, device=device),
    )
    stats = {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "mean_reward": float(rewards.mean()),
    }
    return rollout, obs, stats


def update_policy(model, optimizer, rollout: Rollout, args) -> dict:
    advantages = (rollout.advantages - rollout.advantages.mean()) / (rollout.advantages.std() + 1e-8)
    batch_size = rollout.obs.shape[0]
    indices = np.arange(batch_size)

    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []

    for _ in range(args.epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, args.batch_size):
            batch_idx = torch.as_tensor(indices[start : start + args.batch_size], device=rollout.obs.device)
            new_log_probs, entropy, new_values = model.evaluate(rollout.obs[batch_idx], rollout.raw_actions[batch_idx])
            log_ratio = new_log_probs - rollout.log_probs[batch_idx]
            ratio = log_ratio.exp()

            clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
            policy_loss = -torch.min(ratio * advantages[batch_idx], clipped_ratio * advantages[batch_idx]).mean()
            value_loss = 0.5 * (rollout.returns[batch_idx] - new_values).pow(2).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy_loss.item())
            approx_kls.append(approx_kl.item())

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),
    }


def save_checkpoint(path: Path, model: ActorCritic, optimizer, obs_rms: RunningMeanStd, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "obs_rms": obs_rms.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: ActorCritic, obs_rms: RunningMeanStd, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    obs_rms.load_state_dict(checkpoint["obs_rms"])


def train(args) -> None:
    device = device_from_arg(args.device)
    env = gym.make("Humanoid-v5")
    obs, _ = env.reset(seed=args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = ActorCritic(obs_dim, act_dim, env.action_space.low, env.action_space.high, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_rms = RunningMeanStd((obs_dim,))

    for update in range(1, args.updates + 1):
        rollout, obs, rollout_stats = collect_rollout(env, model, obs_rms, obs, args, device)
        train_stats = update_policy(model, optimizer, rollout, args)

        if update % args.log_interval == 0 or update == 1:
            episode_returns = rollout_stats["episode_returns"]
            episode_lengths = rollout_stats["episode_lengths"]
            mean_ep_return = np.mean(episode_returns) if episode_returns else float("nan")
            mean_ep_length = np.mean(episode_lengths) if episode_lengths else float("nan")
            print(
                f"update={update:04d} "
                f"steps={update * args.steps_per_update} "
                f"mean_ep_return={mean_ep_return:.2f} "
                f"mean_ep_length={mean_ep_length:.1f} "
                f"batch_reward={rollout_stats['mean_reward']:.3f} "
                f"value_loss={train_stats['value_loss']:.3f} "
                f"policy_loss={train_stats['policy_loss']:.3f} "
                f"entropy={train_stats['entropy']:.3f}"
            )

        if update % args.save_interval == 0 or update == args.updates:
            save_checkpoint(Path(args.checkpoint), model, optimizer, obs_rms, args)

    env.close()
    print(f"saved checkpoint: {args.checkpoint}")


def play(args) -> None:
    device = device_from_arg(args.device)
    env = gym.make("Humanoid-v5", render_mode="human" if args.render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model = ActorCritic(obs_dim, act_dim, env.action_space.low, env.action_space.high, args.hidden).to(device)
    obs_rms = RunningMeanStd((obs_dim,))
    load_checkpoint(Path(args.checkpoint), model, obs_rms, device)
    model.eval()

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    for step in range(1, args.play_steps + 1):
        norm_obs = torch.as_tensor(obs_rms.normalize(obs), dtype=torch.float32, device=device).unsqueeze(0)
        action, _, _, _ = model.act(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        total_reward += float(reward)
        if terminated or truncated:
            print(f"episode ended at step={step}, total_reward={total_reward:.3f}")
            break
    else:
        print(f"completed steps={args.play_steps}, total_reward={total_reward:.3f}")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train or play a PPO policy for Gymnasium Humanoid-v5.")
    parser.add_argument("--mode", choices=["train", "play"], default="train")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="checkpoints/humanoid_ppo.pt")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--play-steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
