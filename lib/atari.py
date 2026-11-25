"""
Classic DeepMind Atari preprocessing, written from scratch.

This replaces `ptan.common.wrappers.wrap_dqn()` used in the book. The stack is:

    raw env (210x160 RGB, 60 FPS, no frame skip)
      -> NoopResetEnv     random number of NOOPs after reset (diverse starts)
      -> MaxAndSkipEnv    repeat each action 4 frames, max over the last 2 (removes flicker)
      -> EpisodicLifeEnv  losing a life ends the "episode" for the learner (optional)
      -> FireResetEnv     press FIRE after reset, for games that need it
      -> WarpFrame        grayscale + resize to 84x84
      -> ClipRewardEnv    reward -> sign(reward) (optional)
      -> FrameStack       stack the last 4 frames -> observation (4, 84, 84) uint8
"""
import collections

import ale_py
import cv2
import gymnasium as gym
import numpy as np

gym.register_envs(ale_py)

ENV_ID = "SeaquestNoFrameskip-v4"
FRAME_SIZE = 84


def process_frame(frame: np.ndarray) -> np.ndarray:
    """(210, 160, 3) RGB uint8 -> (84, 84) grayscale uint8."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))
        for _ in range(noops):
            obs, _, term, trunc, info = self.env.step(0)
            if term or trunc:
                obs, info = self.env.reset()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._buf = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        term = trunc = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            if i == self._skip - 2:
                self._buf[0] = obs
            if i == self._skip - 1:
                self._buf[1] = obs
            total_reward += float(reward)
            if term or trunc:
                self._buf[:] = obs
                break
        return self._buf.max(axis=0), total_reward, term, trunc, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Signal termination on life loss, but only really reset the game when it's over.
    Helps value estimation: the agent learns that losing a life is bad.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.was_real_done = term or trunc
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            term = True
        self.lives = lives
        return obs, reward, term, trunc, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # life lost but game is not over: just continue with a NOOP step
            obs, _, term, trunc, info = self.env.step(0)
            if term or trunc:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert meanings[1] == "FIRE" and len(meanings) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, info = None, {}
        for action in (1, 2):
            obs, _, term, trunc, info = self.env.step(action)
            if term or trunc:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 255, (FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)

    def observation(self, observation):
        return process_frame(observation)


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return float(np.sign(reward))


class FrameStack(gym.Wrapper):
    """Channels-first frame stack: observation is (k, 84, 84) uint8."""
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self.frames = collections.deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (k,) + shp, dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(self.frames), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.frames.append(obs)
        return np.stack(self.frames), reward, term, trunc, info


def wrap_dqn(env: gym.Env, stack_frames: int = 4, episodic_life: bool = True,
             reward_clipping: bool = True) -> gym.Env:
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if reward_clipping:
        env = ClipRewardEnv(env)
    env = FrameStack(env, stack_frames)
    return env
