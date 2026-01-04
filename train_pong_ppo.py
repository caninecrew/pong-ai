import os
from datetime import datetime
import shutil
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
TARGET_FPS = 30
MAX_VIDEO_FRAMES = TARGET_FPS * 120  # 2 minutes at 30 fps

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


def record_video_segment(model: PPO, ball_color: Tuple[int, int, int], steps: int = 400) -> Tuple[List[np.ndarray], bool]:
    """
    Roll out a short episode with the trained model and return frames plus whether the ball was successfully returned.
    """
    env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode="rgb_array", ball_color=ball_color)
    obs, _ = env.reset()
    frames: List[np.ndarray] = []
    target_size = (320, 192)  # divisible by 16 to keep codecs happy
    ponged = False

    frame = env.render()
    if frame is not None:
        frames.append(np.array(Image.fromarray(frame).resize(target_size)))

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, terminated, truncated, _ = env.step(action)
        if rew > 0:
            ponged = True
        frame = env.render()
        if frame is not None:
            frames.append(np.array(Image.fromarray(frame).resize(target_size)))
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    return frames, ponged


def build_grid_frames(segments: List[List[np.ndarray]]) -> List[np.ndarray]:
    """
    Arrange per-model segments into a grid per timestep.
    """
    if not segments or not any(segments):
        return []

    max_len = max(len(seg) for seg in segments)
    num_models = len(segments)
    cols = math.ceil(math.sqrt(num_models))
    rows = math.ceil(num_models / cols)

    placeholder = None
    for seg in segments:
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
                seg = segments[idx]
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

    return grid_frames


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

    all_grid_frames: List[np.ndarray] = []
    failure_detected = False

    while len(all_grid_frames) < MAX_VIDEO_FRAMES and not failure_detected:
        combined_frames_per_model: List[List[np.ndarray]] = []
        scores: List[Tuple[str, float]] = []
        pong_flags: List[bool] = []
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

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

            model.learn(total_timesteps=TRAIN_TIMESTEPS)

            stamped_model_path = f"models/{model_id}_{timestamp}.zip"

            model.save(stamped_model_path)
            model.save(latest_path)
            print(f"[{model_id}] Saved timestamped model: {stamped_model_path}")
            print(f"[{model_id}] Updated latest model: {latest_path}")

            segment, ponged = record_video_segment(model, ball_color=color, steps=400)
            combined_frames_per_model.append(segment)
            pong_flags.append(ponged)
            print(f"[{model_id}] Added {len(segment)} frames with ball color {color} to combined video. Ponged: {ponged}")

            # Evaluate a quick mean episode reward to choose the best model of this cycle.
            eval_env = SB3PongEnv(opponent_policy=simple_tracking_policy, render_mode=None)
            episodes = 3
            total = 0.0
            for _ in range(episodes):
                obs, _ = eval_env.reset()
                done = False
                ep_rew = 0.0
                steps = 0
                while not done and steps < eval_env.env.cfg.max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rew, terminated, truncated, _ = eval_env.step(action)
                    ep_rew += rew
                    steps += 1
                    done = terminated or truncated
                total += ep_rew
            eval_env.close()
            avg_rew = total / episodes if episodes else 0.0
            scores.append((model_id, avg_rew))
            print(f"[{model_id}] Avg reward over {episodes} eval episodes: {avg_rew:.3f}")

        if combined_frames_per_model and any(combined_frames_per_model):
            grid_frames = build_grid_frames(combined_frames_per_model)
            for frame in grid_frames:
                if len(all_grid_frames) >= MAX_VIDEO_FRAMES:
                    break
                all_grid_frames.append(frame)
            print(f"Accumulated {len(all_grid_frames)} frames toward combined video (max {MAX_VIDEO_FRAMES}).")
        else:
            print("No frames captured this cycle; skipping video accumulation.")

        # Choose best model of this cycle and propagate to all latest checkpoints.
        if scores:
            best_id, best_score = max(scores, key=lambda t: t[1])
            best_latest = f"models/{best_id}_latest.zip"
            for model_id in model_ids:
                target_latest = f"models/{model_id}_latest.zip"
                if os.path.exists(best_latest) and best_latest != target_latest:
                    shutil.copy2(best_latest, target_latest)
            print(f"Best model this cycle: {best_id} (avg reward {best_score:.3f}); propagated to all _latest checkpoints.")
        else:
            print("No scores recorded; cannot propagate best model.")

        # Failure condition: no model managed to pong the ball back this cycle.
        if pong_flags and not any(pong_flags):
            print("Failure detected: no model returned the ball this cycle.")
            failure_detected = True

    if all_grid_frames:
        combined_video_path = f"videos/ppo_pong_combined_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4"
        imageio.mimsave(combined_video_path, all_grid_frames, fps=TARGET_FPS)
        print(f"Saved combined video with all models in a grid: {combined_video_path} (frames: {len(all_grid_frames)})")
    else:
        print("No frames captured; combined video not written.")

    env.close()


if __name__ == "__main__":
    main()
