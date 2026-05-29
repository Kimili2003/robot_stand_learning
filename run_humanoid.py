import argparse

import gymnasium as gym


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gymnasium MuJoCo Humanoid-v5 with random actions.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Open the MuJoCo viewer window.")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = gym.make("Humanoid-v5", render_mode=render_mode)

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    print(f"observation_space={env.observation_space}")
    print(f"action_space={env.action_space}")
    print(f"initial_observation_shape={obs.shape}")

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"episode ended at step={step}, terminated={terminated}, truncated={truncated}")
            break
    else:
        print(f"completed steps={args.steps}")

    print(f"total_reward={total_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()
