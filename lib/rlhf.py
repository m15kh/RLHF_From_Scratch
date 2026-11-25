"""
RLHF building blocks:

* EpisodeRecorderWrapper - randomly records short segments of raw gameplay into a DB directory
* segment DB helpers      - store / load / list segments
* human labels            - labels.json (JSON lines) in the DB root
* reward model helpers    - preprocessing shared by training and inference
* RewardModelWrapper      - replaces the game reward with the learned reward model output

DB layout:
    db-v0/
      00/00012345.npz     <- segment recorded by env #0 at its step 12345
      01/...
      ...
      15/...
      labels.json         <- {"sample1": "14/00023925.npz", "sample2": "10/00606788.npz", "label": 0}
"""
import collections
import json
import os
import pathlib
import random
import typing as tt
from copy import deepcopy
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch

from .atari import process_frame
from .model import RewardModel

# how many transitions to store in episode
EPISODE_STEPS = 50
# probability to start episode recording
START_PROB = 0.00005

SEGMENT_SUFFIX = ".npz"
LABELS_FILE = "labels.json"

# reward model input: last N grayscale 84x84 frames + one-hot action
RM_STACK_FRAMES = 4
# predicted rewards are normalized to zero mean / unit std per raw frame, then multiplied by
# this scale. Atari wrappers sum rewards over 4 frames, so 0.25 keeps a step reward ~[-1, 1].
RM_REWARD_SCALE = 0.25


# ---------------------------------------------------------------------------------------
# Episode segments
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class EpisodeStep:
    obs: np.ndarray
    act: int


def store_segment(root_path: pathlib.Path, step_idx: int, steps: tt.List[EpisodeStep]) -> pathlib.Path:
    """
    Book version uses pickle; we use compressed numpy arrays instead. Atari frames compress
    extremely well (~12 GB for 2,500 segments in the book -> a few hundred MB here), and
    loading .npz files can't execute code the way unpickling can.
    """
    out_path = root_path / f"{step_idx:08d}{SEGMENT_SUFFIX}"
    obs = np.stack([s.obs for s in steps])
    acts = np.array([s.act for s in steps], dtype=np.int64)
    tmp_path = out_path.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(f, obs=obs, acts=acts)
    tmp_path.replace(out_path)
    print(f"Stored {out_path}")
    return out_path


def load_segment(path: pathlib.Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (obs: (N, 210, 160, 3) uint8 RGB, acts: (N,) int64)"""
    with np.load(path) as data:
        return data["obs"], data["acts"]


def list_segments(db_path: pathlib.Path) -> list[str]:
    """Relative paths of all segments in the DB, like '14/00023925.npz'"""
    return sorted(p.relative_to(db_path).as_posix() for p in db_path.glob(f"*/*{SEGMENT_SUFFIX}"))


class EpisodeRecorderWrapper(gym.Wrapper):
    """
    Must be applied to the raw environment (before the Atari wrappers), so we store
    original color frames which are easy for a human to judge.
    """
    def __init__(self, env: gym.Env, db_path: pathlib.Path, env_idx: int,
                 start_prob: float = START_PROB, steps_count: int = EPISODE_STEPS):
        super().__init__(env)
        self._store_path = db_path / f"{env_idx:02d}"
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._start_prob = start_prob
        self._steps_count = steps_count
        self._is_storing = False
        self._steps: tt.List[EpisodeStep] = []
        self._prev_obs = None
        # Small deviation from the book: continue numbering after existing files, so
        # restarting training with the same DB never overwrites already labeled segments.
        existing = [int(p.stem) for p in self._store_path.glob(f"*{SEGMENT_SUFFIX}") if p.stem.isdigit()]
        self._step_idx = max(existing, default=0)

    def reset(self, *, seed: int | None = None, options: dict[str, tt.Any] | None = None):
        self._step_idx += 1
        res = super().reset(seed=seed, options=options)
        if self._is_storing:
            self._prev_obs = deepcopy(res[0])
        return res

    def step(self, action):
        self._step_idx += 1
        obs, r, is_done, is_tr, extra = super().step(action)
        if self._is_storing:
            self._steps.append(EpisodeStep(self._prev_obs, int(action)))
            self._prev_obs = deepcopy(obs)

            if len(self._steps) >= self._steps_count:
                store_segment(self._store_path, self._step_idx, self._steps)
                self._is_storing = False
                self._steps.clear()
        elif random.random() <= self._start_prob:
            # start recording
            self._is_storing = True
            self._prev_obs = deepcopy(obs)
        return obs, r, is_done, is_tr, extra


# ---------------------------------------------------------------------------------------
# Human labels
# ---------------------------------------------------------------------------------------

@dataclass
class HumanLabel:
    sample1: str
    sample2: str
    # 0 - both are equally good (or bad), 1 - first is better, 2 - second is better
    label: int

    @property
    def mu1(self) -> float:
        """Probability that sample1 is preferred, used as the training target."""
        return {0: 0.5, 1: 1.0, 2: 0.0}[self.label]

    @property
    def key(self) -> tuple[str, str]:
        return self.sample1, self.sample2


def load_labels(db_path: pathlib.Path) -> list[HumanLabel]:
    path = db_path / LABELS_FILE
    if not path.exists():
        return []
    res = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            res.append(HumanLabel(**json.loads(line)))
    return res


def save_labels(db_path: pathlib.Path, labels: list[HumanLabel]):
    path = db_path / LABELS_FILE
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text("".join(json.dumps(asdict(lbl)) + "\n" for lbl in labels))
    os.replace(tmp_path, path)


def set_label(db_path: pathlib.Path, sample1: str, sample2: str, label: int):
    """Add a new label or change the existing one for this pair."""
    assert label in (0, 1, 2)
    labels = load_labels(db_path)
    for lbl in labels:
        if lbl.key == (sample1, sample2):
            lbl.label = label
            break
    else:
        labels.append(HumanLabel(sample1, sample2, label))
    save_labels(db_path, labels)


# ---------------------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------------------

def segment_frames(obs: np.ndarray) -> np.ndarray:
    """(N, 210, 160, 3) RGB -> (N, 84, 84) grayscale"""
    return np.stack([process_frame(o) for o in obs])


def stack_frames(frames: np.ndarray, stack: int = RM_STACK_FRAMES) -> np.ndarray:
    """
    (N, 84, 84) -> (N, stack, 84, 84): entry i has frames [i-stack+1, ..., i].
    At the segment start the first frame is repeated, the same way the
    RewardModelWrapper fills its frame buffer after reset.
    """
    idx = np.arange(len(frames))[:, None] + np.arange(-stack + 1, 1)[None, :]
    return frames[np.clip(idx, 0, None)]


def save_reward_model(path: pathlib.Path, model: RewardModel, input_shape: tuple[int, ...],
                      n_actions: int, reward_mean: float, reward_std: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "input_shape": list(input_shape),
        "n_actions": n_actions,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }, path)


def load_reward_model(path: pathlib.Path | str, dev: torch.device | str = "cpu") -> tuple[RewardModel, dict]:
    ckpt = torch.load(path, map_location=dev, weights_only=True)
    model = RewardModel(tuple(ckpt["input_shape"]), ckpt["n_actions"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


class RewardModelWrapper(gym.Wrapper):
    """
    Replaces the game reward with the reward model prediction r_hat(obs, action).

    Applied to the raw env (like EpisodeRecorderWrapper) because the reward model is trained
    on raw frames from the segment DB, so preprocessing has to match exactly.

    The real game score is still useful to watch, so at the end of each game the wrapper
    puts (game_reward, game_steps) into metrics_queue (if given).
    """
    def __init__(self, env: gym.Env, reward_path: pathlib.Path, dev: torch.device | str = "cpu",
                 metrics_queue: tt.Optional[tt.Any] = None, reward_scale: float = RM_REWARD_SCALE):
        super().__init__(env)
        self._dev = torch.device(dev)
        self._model, ckpt = load_reward_model(reward_path, self._dev)
        self._mean = float(ckpt["reward_mean"])
        self._std = max(float(ckpt["reward_std"]), 1e-6)
        self._scale = reward_scale
        self._metrics_queue = metrics_queue
        self._frames = collections.deque(maxlen=ckpt["input_shape"][0])
        self._game_reward = 0.0
        self._game_steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, tt.Any] | None = None):
        obs, info = super().reset(seed=seed, options=options)
        frame = process_frame(obs)
        for _ in range(self._frames.maxlen):
            self._frames.append(frame)
        self._game_reward = 0.0
        self._game_steps = 0
        return obs, info

    @torch.no_grad()
    def _predict(self, action: int) -> float:
        obs_t = torch.from_numpy(np.stack(self._frames)).unsqueeze(0).to(self._dev)
        act_t = torch.tensor([action], dtype=torch.int64, device=self._dev)
        r = self._model(obs_t, act_t).item()
        return (r - self._mean) / self._std * self._scale

    def step(self, action):
        reward = self._predict(int(action))
        obs, game_reward, is_done, is_tr, extra = super().step(action)
        self._frames.append(process_frame(obs))
        self._game_reward += float(game_reward)
        self._game_steps += 1
        if (is_done or is_tr) and self._metrics_queue is not None:
            self._metrics_queue.put((self._game_reward, self._game_steps))
        return obs, reward, is_done, is_tr, extra
