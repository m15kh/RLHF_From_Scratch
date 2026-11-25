#!/usr/bin/env python3
"""
Play one full game with a trained A2C model and record it to an mp4 video.

    ./01_play.py -m save/v0/best.dat -o videos/v0.mp4
"""
import argparse
import pathlib

import cv2
import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Categorical

from lib import atari
from lib.model import AtariA2C

VIDEO_SCALE = 3
TIME_LIMIT = 10_000


class FrameCollector(gym.Wrapper):
    """Keeps every raw RGB frame (60 FPS) so the video plays at real game speed."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.frames: list[np.ndarray] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames.append(obs)
        return obs, info

    def step(self, action):
        obs, r, is_done, is_tr, info = self.env.step(action)
        self.frames.append(obs)
        return obs, r, is_done, is_tr, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="A2C model file to load")
    parser.add_argument("-o", "--output", default="video.mp4", help="Output video file, default=video.mp4")
    parser.add_argument("--dev", default="cpu")
    parser.add_argument("--greedy", action="store_true", help="Take argmax actions instead of sampling")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    dev = torch.device(args.dev)

    recorder = FrameCollector(gym.make(atari.ENV_ID))
    env = gym.wrappers.TimeLimit(atari.wrap_dqn(recorder, episodic_life=False, reward_clipping=False), TIME_LIMIT)

    net = AtariA2C(env.observation_space.shape, env.action_space.n).to(dev)
    net.load_state_dict(torch.load(args.model, map_location=dev, weights_only=True))
    net.eval()

    obs, _ = env.reset(seed=args.seed)
    total_reward, steps = 0.0, 0
    with torch.no_grad():
        while True:
            logits, _ = net(torch.as_tensor(obs, device=dev).unsqueeze(0))
            if args.greedy:
                action = logits.argmax(dim=1).item()
            else:
                action = Categorical(logits=logits).sample().item()
            obs, r, is_done, is_tr, _ = env.step(action)
            total_reward += r
            steps += 1
            if is_done or is_tr:
                break
    print(f"Game over: reward {total_reward:.0f}, {steps} steps, {len(recorder.frames)} frames")

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = recorder.frames[0].shape[:2]
    size = (w * VIDEO_SCALE, h * VIDEO_SCALE)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), 60, size)
    for frame in recorder.frames:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Video saved to {out}")


if __name__ == "__main__":
    main()
