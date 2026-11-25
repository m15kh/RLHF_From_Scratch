#!/usr/bin/env python3
"""
Step 1 (and step 4) of the RLHF pipeline: A2C training on SeaQuest.

Initial training ("v0"), recording random segments for labeling:
    ./01_a2c.py --dev cuda -n v0 --save save/v0 --db-path db-v0

Fine-tuning with the learned reward model instead of the game score:
    ./01_a2c.py --dev cuda -n v1 --save save/v1 --finetune save/v0/best.dat --reward rw/reward-v0.dat
"""
import argparse
import collections
import multiprocessing as mp
import pathlib
import queue
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from lib import atari, rlhf
from lib.model import AtariA2C

NUM_ENVS = 16
N_STEPS = 5               # rollout length per env -> batch of 16 * 5 = 80 transitions
GAMMA = 0.99
LEARNING_RATE = 7e-4      # decayed linearly to 0 over --max-steps
ENTROPY_BETA = 0.01
VALUE_COEF = 0.25
CLIP_GRAD = 0.5
TIME_LIMIT = 10_000       # agent steps (each one is 4 game frames)
TEST_EPISODES = 10
REPORT_EVERY_SEC = 10


def make_env(env_idx: int, dev: torch.device, db_path: str | None = None,
             reward_path: str | None = None, metrics_queue: queue.SimpleQueue | None = None,
             test: bool = False) -> gym.Env:
    
    e = gym.make(atari.ENV_ID)
    if reward_path is not None:
        e = rlhf.RewardModelWrapper(e, pathlib.Path(reward_path), dev=dev, metrics_queue=metrics_queue)
    if db_path is not None:
        p = pathlib.Path(db_path)
        p.mkdir(parents=True, exist_ok=True)
        e = rlhf.EpisodeRecorderWrapper(e, p, env_idx=env_idx)
    # The learned reward is already normalized, clipping it to sign() would destroy it.
    # Test env uses the original game score and full games (no life-loss episodes).
    e = atari.wrap_dqn(e, episodic_life=not test, reward_clipping=not test and reward_path is None)
    # add time limit after all wrappers
    e = gym.wrappers.TimeLimit(e, TIME_LIMIT)
    return e


@torch.no_grad()
def test_model(net: AtariA2C, dev: torch.device, episodes: int = TEST_EPISODES) -> tuple[float, float, float]:
    """Play full games with the original (unclipped) score. Returns (mean reward, mean steps, max reward)."""
    env = make_env(0, dev, test=True)
    rewards, steps = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward, n_steps = 0.0, 0
        while True:
            logits, _ = net(torch.as_tensor(obs, device=dev).unsqueeze(0))
            action = Categorical(logits=logits).sample().item()
            obs, r, is_done, is_tr, _ = env.step(action)
            total_reward += r
            n_steps += 1
            if is_done or is_tr:
                break
        rewards.append(total_reward)
        steps.append(n_steps)
    env.close()
    return float(np.mean(rewards)), float(np.mean(steps)), float(np.max(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda", help="Device to use, default=cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run, used in TensorBoard")
    parser.add_argument("--save", help="Directory to save best models after testing")
    parser.add_argument("--db-path", help="Directory to store random episode segments for labeling")
    parser.add_argument("--reward", help="Reward model checkpoint: train on the learned reward instead of the game score")
    parser.add_argument("--finetune", help="A2C model to start from")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Initial learning rate, default={LEARNING_RATE}")
    parser.add_argument("--max-steps", type=int, default=12_000_000,
                        help="Total agent steps (over all envs) to train; LR goes to 0 at this point")
    parser.add_argument("--test-every", type=int, default=1000, help="Test the model every N batches, default=1000")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dev = torch.device(args.dev)
    torch.manual_seed(args.seed)
    save_path = None
    if args.save:
        save_path = pathlib.Path(args.save)
        save_path.mkdir(parents=True, exist_ok=True)

    # Each env runs in its own process. A Manager queue can be shared between processes.
    manager = mp.Manager()
    metrics_queue = manager.Queue()
    envs = gym.vector.AsyncVectorEnv(
        [lambda i=i: make_env(i, dev, db_path=args.db_path, reward_path=args.reward, metrics_queue=metrics_queue)
         for i in range(NUM_ENVS)],
        context="spawn",  # safe with CUDA in the child processes
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,  # finished env returns the new episode's first obs
    )
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    net = AtariA2C(obs_shape, n_actions).to(dev)
    if args.finetune:
        net.load_state_dict(torch.load(args.finetune, map_location=dev, weights_only=True))
        print(f"Loaded model from {args.finetune}")
    print(net)

    optimizer = torch.optim.RMSprop(net.parameters(), lr=args.lr, alpha=0.99, eps=1e-5)
    writer = SummaryWriter(log_dir=f"runs/a2c-{args.name}")

    obs, _ = envs.reset(seed=args.seed)  # env i gets seed + i
    ep_rewards = np.zeros(NUM_ENVS)
    ep_steps = np.zeros(NUM_ENVS, dtype=np.int64)
    done_rewards = collections.deque(maxlen=100)
    done_steps = collections.deque(maxlen=100)
    game_rewards = collections.deque(maxlen=100)
    games_done = 0
    best_test_reward = best_test_steps = None

    step_idx = 0
    batch_idx = 0
    ts = ts_report = time.time()
    step_ts = 0

    try:
        while step_idx < args.max_steps:
            # ---------------- collect N_STEPS transitions from every env ----------------
            mb_obs = np.empty((N_STEPS, NUM_ENVS) + obs_shape, dtype=np.uint8)
            mb_actions = np.empty((N_STEPS, NUM_ENVS), dtype=np.int64)
            mb_rewards = np.empty((N_STEPS, NUM_ENVS), dtype=np.float32)
            mb_dones = np.empty((N_STEPS, NUM_ENVS), dtype=np.float32)

            for t in range(N_STEPS):
                with torch.no_grad():
                    logits, _ = net(torch.as_tensor(obs, device=dev))
                    actions = Categorical(logits=logits).sample().cpu().numpy()
                mb_obs[t] = obs
                mb_actions[t] = actions
                # all 16 envs step at the same time, each in its own process
                obs, r, is_done, is_tr, _ = envs.step(actions)
                # truncation (time limit) is treated as terminal - small bias, much simpler code
                dones = is_done | is_tr
                ep_rewards += r
                ep_steps += 1
                mb_rewards[t] = r
                mb_dones[t] = dones
                for i in np.flatnonzero(dones):
                    done_rewards.append(ep_rewards[i])
                    done_steps.append(ep_steps[i])
                    games_done += 1
                    writer.add_scalar("episode_reward", ep_rewards[i], step_idx)
                    writer.add_scalar("episode_steps", ep_steps[i], step_idx)
                    ep_rewards[i] = 0.0
                    ep_steps[i] = 0
                step_idx += NUM_ENVS

            # real game score when training on the learned reward
            while not metrics_queue.empty():
                g_reward, g_steps = metrics_queue.get()
                game_rewards.append(g_reward)
                writer.add_scalar("game_reward", g_reward, step_idx)
                writer.add_scalar("game_steps", g_steps, step_idx)

            # ---------------- n-step returns ----------------
            with torch.no_grad():
                _, last_values = net(torch.as_tensor(obs, device=dev))
            returns = np.empty_like(mb_rewards)
            ret = last_values.cpu().numpy()
            for t in reversed(range(N_STEPS)):
                ret = mb_rewards[t] + GAMMA * ret * (1.0 - mb_dones[t])
                returns[t] = ret

            # ---------------- A2C update ----------------
            lr = args.lr * max(0.0, 1.0 - step_idx / args.max_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            obs_t = torch.as_tensor(mb_obs.reshape((-1,) + obs_shape), device=dev)
            actions_t = torch.as_tensor(mb_actions.reshape(-1), device=dev)
            returns_t = torch.as_tensor(returns.reshape(-1), device=dev)

            logits, values = net(obs_t)
            dist = Categorical(logits=logits)
            adv_t = returns_t - values.detach()
            loss_policy = -(dist.log_prob(actions_t) * adv_t).mean()
            loss_value = F.mse_loss(values, returns_t)
            entropy = dist.entropy().mean()
            loss = loss_policy + VALUE_COEF * loss_value - ENTROPY_BETA * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
            optimizer.step()
            batch_idx += 1

            # ---------------- reporting ----------------
            now = time.time()
            if now - ts_report > REPORT_EVERY_SEC:
                speed = (step_idx - step_ts) / (now - ts)
                ts, step_ts, ts_report = now, step_idx, now
                mean_r = np.mean(done_rewards) if done_rewards else 0.0
                mean_s = np.mean(done_steps) if done_steps else 0.0
                msg = (f"{step_idx}: done {games_done} episodes, mean reward {mean_r:.3f}, "
                       f"steps {mean_s:.0f}, speed {speed:.2f} f/s")
                if game_rewards:
                    msg += f", game reward {np.mean(game_rewards):.1f}"
                print(msg)
                writer.add_scalar("speed", speed, step_idx)
                writer.add_scalar("lr", lr, step_idx)
                writer.add_scalar("loss_policy", loss_policy.item(), step_idx)
                writer.add_scalar("loss_value", loss_value.item(), step_idx)
                writer.add_scalar("entropy", entropy.item(), step_idx)
                writer.add_scalar("grad_norm", grad_norm.item(), step_idx)
                writer.add_scalar("advantage", adv_t.mean().item(), step_idx)

            if batch_idx % args.test_every == 0:
                print(f"{step_idx}: Testing model...")
                test_reward, test_steps, test_max = test_model(net, dev)
                print(f"Got mean reward {test_reward:.2f} (max {test_max:.0f}) "
                      f"and steps {test_steps:.1f} in {TEST_EPISODES} episodes")
                writer.add_scalar("test_reward", test_reward, step_idx)
                writer.add_scalar("test_steps", test_steps, step_idx)
                better_reward = best_test_reward is None or test_reward > best_test_reward
                better_steps = best_test_steps is None or test_steps > best_test_steps
                if save_path is not None and (better_reward or better_steps):
                    name = f"a2c-{step_idx:09d}-r{test_reward:.1f}-s{test_steps:.0f}.dat"
                    torch.save(net.state_dict(), save_path / name)
                    if better_reward:
                        torch.save(net.state_dict(), save_path / "best.dat")
                    print(f"Saved {save_path / name}")
                if better_reward:
                    best_test_reward = test_reward
                if better_steps:
                    best_test_steps = test_steps
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if save_path is not None:
            torch.save(net.state_dict(), save_path / "last.dat")
            print(f"Saved {save_path / 'last.dat'}")
        writer.close()
        envs.close()
        manager.shutdown()


if __name__ == "__main__":
    main()