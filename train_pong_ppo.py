import os
from datetime import datetime
from typing import Callable, Optional, List, Tuple

import math
import imageio
import pygame
import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Change this to control how many timesteps each model trains per run.
TRAIN_TIMESTEPS = 200_000

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
        ball_color: Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()
        self.env = PongEnv(render_mode=render_mode, ball_color=ball_color)
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


def record_video_segment(model: PPO, ball_color: Tuple[int, int, int], steps: int = 400) -> List[np.ndarray]:
    """
    Roll out a short episode with the trained model and return frames.
    """
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="rgb_array", ball_color=ball_color)
    obs, _ = env.reset()
    frames: List[np.ndarray] = []
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

    env.close()
    return frames


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

    # Train multiple model lines in one run; each continues from its own latest checkpoint.
    model_ids = [
        "ppo_pong_custom",   # continues your main model
        "ppo_pong_custom_b", # second line trained in the same run
    ]

    ball_colors = [
        (255, 0, 0),
        (0, 200, 255),
        (255, 200, 0),
        (0, 255, 120),
    ]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    combined_frames_per_model: List[List[np.ndarray]] = []

    for idx, model_id in enumerate(model_ids):
        latest_path = f"models/{model_id}_latest.zip"
        color = ball_colors[idx % len(ball_colors)]

        if os.path.exists(latest_path):
            print(f"[{model_id}] Loading existing model from {latest_path} to continue training...")
            model = PPO.load(latest_path, env=env)
        else:
            print(f"[{model_id}] No existing model found; starting a fresh PPO model...")
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

        # Adjust timesteps as you like; this continues from the previous checkpoint.
        model.learn(total_timesteps=TRAIN_TIMESTEPS)

        stamped_model_path = f"models/{model_id}_{timestamp}.zip"

        model.save(stamped_model_path)
        model.save(latest_path)
        print(f"[{model_id}] Saved timestamped model: {stamped_model_path}")
        print(f"[{model_id}] Updated latest model: {latest_path}")

        segment = record_video_segment(model, ball_color=color, steps=400)
        combined_frames_per_model.append(segment)
        print(f"[{model_id}] Added {len(segment)} frames with ball color {color} to combined video.")

    if combined_frames_per_model and any(combined_frames_per_model):
        max_len = max(len(seg) for seg in combined_frames_per_model)
        num_models = len(combined_frames_per_model)
        cols = math.ceil(math.sqrt(num_models))
        rows = math.ceil(num_models / cols)

        # Use first available frame as placeholder for padding.
        placeholder = None
        for seg in combined_frames_per_model:
            if seg:
                placeholder = np.zeros_like(seg[0])
                break

        grid_frames: List[np.ndarray] = []

        for i in range(max_len):
            row_images = []
            for r in range(rows):
                row_tiles = []
                for c in range(cols):
                    idx = r * cols + c
                    if idx >= num_models:
                        continue
                    seg = combined_frames_per_model[idx]
                    if seg:
                        if i < len(seg):
                            row_tiles.append(seg[i])
                        else:
                            row_tiles.append(seg[-1])  # hold last frame if shorter
                    elif placeholder is not None:
                        row_tiles.append(placeholder)
                if row_tiles:
                    row_images.append(np.concatenate(row_tiles, axis=1))
            if row_images:
                grid_frame = np.concatenate(row_images, axis=0)
                grid_frames.append(grid_frame)

        combined_video_path = f"videos/ppo_pong_combined_{timestamp}.mp4"
        imageio.mimsave(combined_video_path, grid_frames, fps=30)
        print(f"Saved combined video with all models in a grid: {combined_video_path} (frames: {len(grid_frames)})")
    else:
        print("No frames captured; combined video not written.")

    env.close()


if __name__ == "__main__":
    main()
