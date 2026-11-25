#!/usr/bin/env python3
"""
Step 3 of the RLHF pipeline: train the reward model on human preference labels.

    ./03_reward_train.py -d db-v0 -o rw/reward-v0.dat --dev cuda

For a labeled pair of segments (s1, s2) the model predicts a reward for every step, and the
segment "score" is the sum of step rewards. The probability that s1 is preferred is
(Bradley-Terry model, as in "Deep RL from Human Preferences"):

    P(s1 > s2) = exp(sum r(s1)) / (exp(sum r(s1)) + exp(sum r(s2))) = sigmoid(sum r(s1) - sum r(s2))

and the loss is the cross-entropy between P and the human label mu1 (1.0, 0.0 or 0.5).
"""
import argparse
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from lib import rlhf
from lib.model import RewardModel

BATCH_SIZE = 16          # pairs per batch -> 16 * 2 * 50 frames
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50


class SegmentCache:
    """Loads segments once and keeps them as (N, 84, 84) grayscale frames + actions."""
    def __init__(self, db_path: pathlib.Path):
        self.db_path = db_path
        self.data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def load(self, rel_paths: set[str]):
        ts = time.time()
        for i, p in enumerate(sorted(rel_paths - self.data.keys())):
            obs, acts = rlhf.load_segment(self.db_path / p)
            self.data[p] = (rlhf.segment_frames(obs), acts)
            if (i + 1) % 100 == 0:
                print(f"  loaded {i + 1} segments, {time.time() - ts:.1f}s")

    def tensors(self, rel_path: str) -> tuple[np.ndarray, np.ndarray]:
        frames, acts = self.data[rel_path]
        return rlhf.stack_frames(frames), acts


def make_batch(cache: SegmentCache, labels: list[rlhf.HumanLabel], dev: torch.device):
    """
    Returns obs (2 * B * N, 4, 84, 84), acts (2 * B * N,), mu1 (B,), N - segment length.
    First half of obs/acts belongs to sample1 of every pair, second half to sample2.
    """
    obs1, acts1, obs2, acts2 = [], [], [], []
    for lbl in labels:
        o, a = cache.tensors(lbl.sample1)
        obs1.append(o)
        acts1.append(a)
        o, a = cache.tensors(lbl.sample2)
        obs2.append(o)
        acts2.append(a)
    seg_len = len(acts1[0])
    obs = torch.from_numpy(np.concatenate(obs1 + obs2)).to(dev)
    acts = torch.from_numpy(np.concatenate(acts1 + acts2)).to(dev)
    mu1 = torch.tensor([lbl.mu1 for lbl in labels], dtype=torch.float32, device=dev)
    return obs, acts, mu1, seg_len


def pair_logits(model: RewardModel, obs: torch.Tensor, acts: torch.Tensor, batch: int, seg_len: int) -> torch.Tensor:
    """sum r(s1) - sum r(s2) for every pair in the batch"""
    r = model(obs, acts).view(2, batch, seg_len).sum(dim=2)
    return r[0] - r[1]


@torch.no_grad()
def evaluate(model: RewardModel, cache: SegmentCache, labels: list[rlhf.HumanLabel],
             dev: torch.device) -> tuple[float, float]:
    """Returns (mean loss, accuracy on labels where one sample is clearly better)"""
    model.eval()
    losses, correct, decisive = [], 0, 0
    for i in range(0, len(labels), BATCH_SIZE):
        batch = labels[i:i + BATCH_SIZE]
        obs, acts, mu1, seg_len = make_batch(cache, batch, dev)
        logits = pair_logits(model, obs, acts, len(batch), seg_len)
        losses.append(F.binary_cross_entropy_with_logits(logits, mu1, reduction="sum").item())
        mask = mu1 != 0.5
        decisive += int(mask.sum().item())
        correct += int(((logits > 0) == (mu1 > 0.5))[mask].sum().item())
    return sum(losses) / max(1, len(labels)), correct / max(1, decisive)


@torch.no_grad()
def reward_stats(model: RewardModel, cache: SegmentCache, dev: torch.device) -> tuple[float, float]:
    """Mean and std of predicted per-step reward over all loaded segments (for normalization)."""
    model.eval()
    rewards = []
    for p in cache.data:
        obs, acts = cache.tensors(p)
        rewards.append(model(torch.from_numpy(obs).to(dev), torch.from_numpy(acts).to(dev)).cpu())
    r = torch.cat(rewards)
    return r.mean().item(), r.std().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--db", required=True, help="DB directory with segments and labels.json")
    parser.add_argument("-o", "--out", required=True, help="File to save the reward model to")
    parser.add_argument("--dev", default="cpu", help="Device to use, default=cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Part of labels held out for validation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.dev)
    db_path = pathlib.Path(args.db)
    out_path = pathlib.Path(args.out)

    labels = rlhf.load_labels(db_path)
    if len(labels) < 2:
        parser.error(f"Need at least 2 labels in {db_path / rlhf.LABELS_FILE}, got {len(labels)}")
    random.shuffle(labels)
    n_test = int(len(labels) * args.test_ratio)
    test_labels, train_labels = labels[:n_test], labels[n_test:]
    print(f"Labels: {len(labels)} total, {len(train_labels)} train, {len(test_labels)} test")

    cache = SegmentCache(db_path)
    paths = {lbl.sample1 for lbl in labels} | {lbl.sample2 for lbl in labels}
    print(f"Loading {len(paths)} segments...")
    cache.load(paths)

    input_shape = (rlhf.RM_STACK_FRAMES,) + cache.data[next(iter(paths))][0].shape[1:]
    n_actions = 18   # full Atari action set, SeaQuest uses all of them
    model = RewardModel(input_shape, n_actions).to(dev)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_loss, best_state = None, None
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_labels)
        train_losses = []
        for i in range(0, len(train_labels), BATCH_SIZE):
            batch = train_labels[i:i + BATCH_SIZE]
            obs, acts, mu1, seg_len = make_batch(cache, batch, dev)
            logits = pair_logits(model, obs, acts, len(batch), seg_len)
            loss = F.binary_cross_entropy_with_logits(logits, mu1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss, train_acc = evaluate(model, cache, train_labels, dev)
        msg = f"Epoch {epoch}: train loss {train_loss:.4f}, acc {train_acc:.3f}"
        # with no test set (tiny label count) select by train loss
        sel_loss = train_loss
        if test_labels:
            test_loss, test_acc = evaluate(model, cache, test_labels, dev)
            msg += f" | test loss {test_loss:.4f}, acc {test_acc:.3f}"
            sel_loss = test_loss
        if best_loss is None or sel_loss < best_loss:
            best_loss = sel_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            msg += "  <- best"
        print(msg)

    model.load_state_dict(best_state)
    mean, std = reward_stats(model, cache, dev)
    print(f"Predicted step reward: mean {mean:.4f}, std {std:.4f} (used for normalization)")
    rlhf.save_reward_model(out_path, model, input_shape, n_actions, mean, std)
    print(f"Reward model saved to {out_path}")


if __name__ == "__main__":
    main()
