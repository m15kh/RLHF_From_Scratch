#!/usr/bin/env python3
"""
Alternative to steps 3 + 4: train the policy directly on the human labels with DPO
(Direct Preference Optimization, Rafailov et al. 2023). No reward model, no RL loop.

    ./04_dpo.py --dev cuda -n dpo-v0 -d db-v0 -m save/v0/best.dat --save save/dpo

For a labeled pair of segments (s1, s2):

    logp(s) = sum over the agent decisions in s of log pi(a | o)
    h       = beta * [(logp_theta(s1) - logp_ref(s1)) - (logp_theta(s2) - logp_ref(s2))]
    loss    = BCE_with_logits(h, mu1)

pi_ref is a frozen copy of the start model. With mu1 = 1 or 0 this is the standard DPO loss.
With mu1 = 0.5 ("both are equal") it pulls both segments to the same score.
beta * log(pi_theta / pi_ref) plays the role of the reward model ("implicit reward").

Segments store raw frames (60 FPS) and the raw action of every frame, but the policy decides
once per 4 frames and sees 4 stacked 84x84 frames. segment_to_decisions() rebuilds the
policy's view from the raw frames, the same way the Atari wrappers do it.
"""
import argparse
import copy
import pathlib
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from lib import atari, rlhf
from lib.model import AtariA2C

BETA = 0.5             # higher = stay closer to the reference policy (a segment has only ~9 decisions)
LEARNING_RATE = 1e-5
BATCH_SIZE = 16        # pairs per batch
EPOCHS = 20
TEST_EPISODES = 5
TIME_LIMIT = 10_000    # agent steps, same as 01_a2c.py

FRAME_SKIP = 4         # MaxAndSkipEnv: the agent action is repeated for 4 raw frames
STACK = 4              # FrameStack: the policy sees the last 4 agent frames


# ---------------------------------------------------------------------------------------
# Raw segment -> policy decisions
# ---------------------------------------------------------------------------------------

def find_offset(acts: np.ndarray) -> int:
    """
    The agent action stays the same for FRAME_SKIP raw frames. Find where these windows start
    (0..3): the right offset gives the most windows with a single action inside.
    """
    scores = []
    for off in range(FRAME_SKIP):
        starts = range(off, len(acts) - FRAME_SKIP + 1, FRAME_SKIP)
        scores.append(sum(np.all(acts[s:s + FRAME_SKIP] == acts[s]) for s in starts))
    return int(np.argmax(scores))


def segment_to_decisions(obs: np.ndarray, acts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    obs: (N, 210, 160, 3) raw RGB, obs[k] is on screen when the raw action acts[k] is taken.
    Returns the policy observations (K, 4, 84, 84) uint8 and the actions (K,) of the agent
    decisions inside the segment.

    A window that starts at raw frame s is one agent decision. Its observation is made of 4 agent
    frames, at raw frames k = s-12, s-8, s-4, s. Each agent frame is max(obs[k-1], obs[k])
    (MaxAndSkipEnv), converted to 84x84 grayscale (WarpFrame). So a decision needs 13 raw frames
    of history. Windows with mixed actions (around a reset or NOOPs) are skipped.
    """
    off = find_offset(acts)
    n = len(acts)
    agent_frame = {k: atari.process_frame(np.maximum(obs[k - 1], obs[k]))
                   for k in range(off, n, FRAME_SKIP) if k >= 1}
    first_s = FRAME_SKIP * (STACK - 1) + 1
    res_obs, res_acts = [], []
    for s in range(off, n - FRAME_SKIP + 1, FRAME_SKIP):
        if s < first_s or not np.all(acts[s:s + FRAME_SKIP] == acts[s]):
            continue
        ks = range(s - FRAME_SKIP * (STACK - 1), s + 1, FRAME_SKIP)
        res_obs.append(np.stack([agent_frame[k] for k in ks]))
        res_acts.append(acts[s])
    if not res_obs:
        return np.empty((0, STACK, atari.FRAME_SIZE, atari.FRAME_SIZE), np.uint8), np.empty(0, np.int64)
    return np.stack(res_obs), np.array(res_acts, dtype=np.int64)


class DecisionCache:
    """Loads segments once and keeps them as policy decisions (obs, acts)."""
    def __init__(self, db_path: pathlib.Path):
        self.db_path = db_path
        self.data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def load(self, rel_paths: set[str]):
        for i, p in enumerate(sorted(rel_paths - self.data.keys())):
            self.data[p] = segment_to_decisions(*rlhf.load_segment(self.db_path / p))
            if (i + 1) % 100 == 0:
                print(f"  loaded {i + 1} segments")


# ---------------------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------------------

def segments_logp(net: AtariA2C, cache: DecisionCache, paths: list[str], dev: torch.device) -> torch.Tensor:
    """log pi(segment) = sum of log pi(a | o) over the segment's decisions, for every path -> (len(paths),)"""
    obs = torch.from_numpy(np.concatenate([cache.data[p][0] for p in paths])).to(dev)
    acts = torch.from_numpy(np.concatenate([cache.data[p][1] for p in paths])).to(dev)
    counts = torch.tensor([len(cache.data[p][1]) for p in paths], device=dev)
    seg_idx = torch.repeat_interleave(torch.arange(len(paths), device=dev), counts)
    logits, _ = net(obs)
    logp = F.log_softmax(logits, dim=1).gather(1, acts.unsqueeze(1)).squeeze(1)
    return torch.zeros(len(paths), device=dev).index_add(0, seg_idx, logp)


@torch.no_grad()
def reference_logp(ref_net: AtariA2C, cache: DecisionCache, dev: torch.device) -> dict[str, float]:
    """The reference policy is frozen, so its log-probs are computed once."""
    paths = sorted(cache.data)
    res = {}
    for i in range(0, len(paths), 64):
        chunk = paths[i:i + 64]
        res.update(zip(chunk, segments_logp(ref_net, cache, chunk, dev).tolist()))
    return res


def pair_logits(net: AtariA2C, cache: DecisionCache, ref_logp: dict[str, float],
                labels: list[rlhf.HumanLabel], beta: float, dev: torch.device) -> torch.Tensor:
    """h for every pair in the batch, (B,)"""
    paths = [lbl.sample1 for lbl in labels] + [lbl.sample2 for lbl in labels]
    logp = segments_logp(net, cache, paths, dev)
    ref = torch.tensor([ref_logp[p] for p in paths], device=dev)
    ratio = (logp - ref).view(2, len(labels))      # implicit reward / beta, per segment
    return beta * (ratio[0] - ratio[1])


@torch.no_grad()
def evaluate(net: AtariA2C, cache: DecisionCache, ref_logp: dict[str, float],
             labels: list[rlhf.HumanLabel], beta: float, dev: torch.device) -> tuple[float, float]:
    """Returns (mean loss, accuracy on labels where one sample is clearly better)"""
    losses, correct, decisive = [], 0, 0
    for i in range(0, len(labels), BATCH_SIZE):
        batch = labels[i:i + BATCH_SIZE]
        h = pair_logits(net, cache, ref_logp, batch, beta, dev)
        mu1 = torch.tensor([lbl.mu1 for lbl in batch], device=dev)
        losses.append(F.binary_cross_entropy_with_logits(h, mu1, reduction="sum").item())
        mask = mu1 != 0.5
        decisive += int(mask.sum().item())
        correct += int(((h > 0) == (mu1 > 0.5))[mask].sum().item())
    return sum(losses) / max(1, len(labels)), correct / max(1, decisive)


# ---------------------------------------------------------------------------------------
# Real game test (same as 01_a2c.py: full games, unclipped score)
# ---------------------------------------------------------------------------------------

def make_test_env() -> gym.Env:
    e = atari.wrap_dqn(gym.make(atari.ENV_ID), episodic_life=False, reward_clipping=False)
    return gym.wrappers.TimeLimit(e, TIME_LIMIT)


@torch.no_grad()
def test_model(net: AtariA2C, dev: torch.device, episodes: int) -> tuple[float, float]:
    """Returns (mean reward, mean steps)"""
    env = make_test_env()
    rewards, steps = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward, n_steps = 0.0, 0
        while True:
            logits, _ = net(torch.as_tensor(obs, device=dev).unsqueeze(0))
            obs, r, is_done, is_tr, _ = env.step(Categorical(logits=logits).sample().item())
            total_reward += r
            n_steps += 1
            if is_done or is_tr:
                break
        rewards.append(total_reward)
        steps.append(n_steps)
    env.close()
    return float(np.mean(rewards)), float(np.mean(steps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run, used in TensorBoard")
    parser.add_argument("-d", "--db", required=True, help="DB directory with segments and labels.json")
    parser.add_argument("-m", "--model", required=True, help="A2C model to start from, also the reference policy")
    parser.add_argument("--save", help="Directory to save models")
    parser.add_argument("--dev", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Part of labels held out for validation")
    parser.add_argument("--test-every", type=int, default=2, help="Play test games every N epochs")
    parser.add_argument("--test-episodes", type=int, default=TEST_EPISODES)
    parser.add_argument("--freeze-conv", action="store_true", help="Train only the policy head")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.dev)
    db_path = pathlib.Path(args.db)
    save_path = pathlib.Path(args.save) if args.save else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    # ---------------- data ----------------
    labels = rlhf.load_labels(db_path)
    cache = DecisionCache(db_path)
    paths = {lbl.sample1 for lbl in labels} | {lbl.sample2 for lbl in labels}
    print(f"Loading {len(paths)} segments...")
    cache.load(paths)
    n_dec = [len(a) for _, a in cache.data.values()]
    print(f"Decisions per segment: mean {np.mean(n_dec):.1f}, min {min(n_dec)}, max {max(n_dec)}")
    # a segment without decisions has logp = 0 for both policies, so the pair teaches nothing
    labels = [lbl for lbl in labels if len(cache.data[lbl.sample1][1]) and len(cache.data[lbl.sample2][1])]
    if len(labels) < 2:
        parser.error(f"Need at least 2 usable labels in {db_path / rlhf.LABELS_FILE}, got {len(labels)}")
    random.shuffle(labels)
    n_test = int(len(labels) * args.test_ratio)
    test_labels, train_labels = labels[:n_test], labels[n_test:]
    print(f"Labels: {len(labels)} usable, {len(train_labels)} train, {len(test_labels)} test")

    # ---------------- policy + frozen reference ----------------
    env = make_test_env()
    net = AtariA2C(env.observation_space.shape, env.action_space.n).to(dev)
    env.close()
    net.load_state_dict(torch.load(args.model, map_location=dev, weights_only=True))
    ref_net = copy.deepcopy(net).eval()
    for p in ref_net.parameters():
        p.requires_grad_(False)
    ref_logp = reference_logp(ref_net, cache, dev)
    if args.freeze_conv:
        for p in net.conv.parameters():
            p.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)
    writer = SummaryWriter(log_dir=f"runs/dpo-{args.name}")

    # the start model is the baseline: h = 0 for every pair, so the loss is ln(2) = 0.693
    best_reward, _ = test_model(net, dev, args.test_episodes)
    print(f"Start model: game reward {best_reward:.1f}")

    try:
        for epoch in range(1, args.epochs + 1):
            random.shuffle(train_labels)
            for i in range(0, len(train_labels), BATCH_SIZE):
                batch = train_labels[i:i + BATCH_SIZE]
                h = pair_logits(net, cache, ref_logp, batch, args.beta, dev)
                mu1 = torch.tensor([lbl.mu1 for lbl in batch], device=dev)
                loss = F.binary_cross_entropy_with_logits(h, mu1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss, train_acc = evaluate(net, cache, ref_logp, train_labels, args.beta, dev)
            writer.add_scalar("train_loss", train_loss, epoch)
            writer.add_scalar("train_acc", train_acc, epoch)
            msg = f"Epoch {epoch}: train loss {train_loss:.4f}, acc {train_acc:.3f}"
            if test_labels:
                test_loss, test_acc = evaluate(net, cache, ref_logp, test_labels, args.beta, dev)
                writer.add_scalar("test_loss", test_loss, epoch)
                writer.add_scalar("test_acc", test_acc, epoch)
                msg += f" | test loss {test_loss:.4f}, acc {test_acc:.3f}"

            if epoch % args.test_every == 0 or epoch == args.epochs:
                game_reward, game_steps = test_model(net, dev, args.test_episodes)
                writer.add_scalar("game_reward", game_reward, epoch)
                writer.add_scalar("game_steps", game_steps, epoch)
                msg += f" | game reward {game_reward:.1f}, steps {game_steps:.0f}"
                if save_path is not None:
                    torch.save(net.state_dict(), save_path / f"dpo-e{epoch:03d}-r{game_reward:.1f}-s{game_steps:.0f}.dat")
                if game_reward > best_reward:
                    best_reward = game_reward
                    msg += "  <- best"
                    if save_path is not None:
                        torch.save(net.state_dict(), save_path / "best.dat")
            print(msg)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if save_path is not None:
            torch.save(net.state_dict(), save_path / "last.dat")
            print(f"Saved {save_path / 'last.dat'}")
        writer.close()


if __name__ == "__main__":
    main()