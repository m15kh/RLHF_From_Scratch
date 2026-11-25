# RLHF from scratch (SeaQuest)

This project implements the RLHF pipeline from *Deep Reinforcement Learning Hands-On*, Chapter 19. It is based on
"Deep RL from Human Preferences" (Christiano et al., 2017). The code is self-contained and doesn't use `ptan`.

```
 1. 01_a2c.py         train A2C on the game score (v0) and record random 50-frame segments into a DB
 2. 02_label_ui.py    a human labels pairs of segments in the browser: "which one is better?"
 3. 03_reward_train.py train a reward model r(obs, action) on those preferences
 4. 01_a2c.py --reward fine-tune v0 using the reward model instead of the game score
    (and optionally record a new DB, label again, and repeat)
```

## Setup

```bash
python3 -m venv --system-site-packages .venv   # reuses an already installed torch
.venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

## 1. Initial training (v0)

```bash
./01_a2c.py --dev cuda -n v0 --save save/v0 --db-path db-v0
```

* `--dev`: device (`cpu` or `cuda`).
* `-n`: run name for TensorBoard (`tensorboard --logdir runs`).
* `--save`: each test (10 full games, unclipped score) writes a model file if the mean reward or the episode length
  beats the previous record. `best.dat` always holds the model with the best reward.
* `--db-path`: where random episode segments are stored. Each env has its own subdirectory, `00` to `15`.
* `--max-steps`: total agent steps (default 12M). The learning rate decays linearly to 0 over this many steps.

Segments: on every raw frame there's a `START_PROB = 0.00005` chance to start recording. A recording is 50 raw RGB
frames plus the actions taken. It runs across episode boundaries and is saved as `db-v0/NN/XXXXXXXX.npz`.

Watch the agent play: `./01_play.py -m save/v0/best.dat -o videos/v0.mp4`

## 2. Labeling

```bash
./02_label_ui.py -d db-v0                 # open http://localhost:8080
./02_label_ui.py -d db-v0 --host 0.0.0.0  # on a remote server (there is NO authentication!)
```

* **Overview** shows the DB path, the number of segments and the number of labels.
* **Label new data** shows 20 random unlabeled pairs. Press `1` if the first is better, `2` if the second is better,
  or `0` if they are equally good or bad. The arrow keys move through the list. **Resample list** loads 20 new pairs.
* **Existing labels** lets you review your labels and change them.

Labels are stored in `db-v0/labels.json`, one JSON object per line. Label 1 means μ1 = 1, μ2 = 0;
label 2 means μ1 = 0, μ2 = 1; label 0 means μ1 = μ2 = 0.5:
```
{"sample1": "14/00023925.npz", "sample2": "10/00606788.npz", "label": 0}
```

Tips: judge what you *want* the agent to do, not the score. In SeaQuest, surfacing to refill oxygen and rescuing
divers is something v0 never learns (it sits on the bottom). Prefer segments that move toward that.

## 3. Reward model training

```bash
./03_reward_train.py -d db-v0 -o rw/reward-v0.dat --dev cuda
```

The model takes 4 stacked 84×84 grayscale frames and a one-hot action, and predicts a per-frame reward. A segment's
score is the sum of its 50 rewards, and the probability that segment 1 is preferred is
`sigmoid(score1 - score2)`. The loss is the cross-entropy against μ1. Held-out accuracy is reported on the
decisive labels (`1`/`2`). The checkpoint also stores the mean and std of the predicted reward, which are used to
normalize the reward during RL.

## 4. Fine-tuning with the reward model

```bash
./01_a2c.py --dev cuda -n v1 --save save/v1 --finetune save/v0/best.dat --reward rw/reward-v0.dat \
            [--db-path db-v1] [--lr 1e-4] [--max-steps 5000000]
```

`RewardModelWrapper` replaces the game reward with the model's prediction, and reward clipping is turned off.
The real game score is still reported as `game reward` in the console and TensorBoard. Tests always use the real
score, so you can see whether your preferences actually helped.

## Differences from the book

* There's no `ptan` dependency. The Atari wrappers (`lib/atari.py`) and the A2C loop are written out in full.
* Segments are saved as compressed `.npz` instead of pickle. They take about 40 KB each instead of about 5 MB,
  and loading them can't run code.
* When a DB already holds segments, new segment numbers continue after the existing ones. That way, restarting a
  run never overwrites segments you've labeled.
* The label UI is written for NiceGUI 3 and 4, not 1.4. It listens on localhost by default.
* The reward model follows the paper's architecture (4 conv layers × 16 filters, batch norm, dropout, leaky ReLU).
