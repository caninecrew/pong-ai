import os
from datetime import datetime
from typing import Callable, Optional

import imageio
import pygame
import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from pong import (
    PongEnv,
    simple_tracking_policy,
    Action,
    STAY,
)


class SB3PongEnv(gym.Env):
    """
    Gymnasium wrapper around the custom PongEnv.
    The learning agent controls the left paddle; the right paddle uses a fixed policy.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        opponent_policy: Optional[Callable[[tuple, bool], Action]] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.env = PongEnv(render_mode=render_mode)
        self.opponent_policy = opponent_policy or (lambda obs, is_left: STAY)
        self.last_obs: Optional[tuple] = None

        # Observations are normalized: [bx, by, bvx, bvy, ly, ry]
        low = np.array([0.0, 0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
        obs, info = self.env.reset()
        self.last_obs = obs
        return np.array(obs, dtype=np.float32), info

    def step(self, action: int):
        if self.last_obs is None:
            raise RuntimeError("Call reset() before step().")

        right_action = self.opponent_policy(self.last_obs, is_left=False)
        obs, reward, done, info = self.env.step(action, right_action)
        self.last_obs = obs

        terminated = False  # Episodes end only by truncation (step cap) in PongEnv.
        truncated = bool(done)
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def record_video(model: PPO, video_path: str, steps: int = 400) -> None:
    """
    Roll out a short episode with the trained model and save an mp4.
    """
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="rgb_array")
    obs, _ = env.reset()
    frames = []
    target_size = (320, 192)  # divisible by 16 to keep codecs happy

    frame = env.render()
    if frame is not None:
        frames.append(np.array(Image.fromarray(frame).resize(target_size)))

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(np.array(Image.fromarray(frame).resize(target_size)))
        if terminated or truncated:
            obs, _ = env.reset()

    if frames:
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Saved video: {video_path} (frames: {len(frames)})")
    else:
        print("No frames captured; video not written.")

    env.close()


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    # Vectorize custom Pong environment for PPO.
    env = make_vec_env(
        lambda: SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None),
        n_envs=8,
        seed=0,
    )

    latest_path = "models/ppo_pong_custom_latest.zip"

    if os.path.exists(latest_path):
        print(f"Loading existing model from {latest_path} to continue training...")
        model = PPO.load(latest_path, env=env)
    else:
        print("No existing model found; starting a fresh PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="logs",
            n_steps=256,
            batch_size=512,
            n_epochs=4,
            gamma=0.99,
            learning_rate=2.5e-4,
        )

    # Shorter run for a quick playable model; bump this higher for better skill.
    model.learn(total_timesteps=100_000)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stamped_model_path = f"models/ppo_pong_custom_{timestamp}.zip"

    model.save(stamped_model_path)
    model.save(latest_path)
    print(f"Saved timestamped model: {stamped_model_path}")
    print(f"Updated latest model: {latest_path}")

    video_path = f"videos/ppo_pong_custom_{timestamp}.mp4"
    record_video(model, video_path, steps=400)

    env.close()


if __name__ == "__main__":
    main()
