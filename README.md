# Ms. Pac-Man DQN Agent

## Overview
This repository trains a Deep Q-Network (DQN) agent to play Ms. Pac-Man
using reinforcement learning. Open `pacman_dqn.ipynb` in Google Colab
(Runtime → Change runtime type → T4 GPU) and select Runtime → Run all.

## My Hyperparameters
- **Exploration:** 0.20 (kept at the notebook's default)
- **Episodes:** 500
- **Learning rate:** 0.0001 (kept at the notebook's default)

I ran two shorter experiments first (5 episodes, then 100 episodes) before
settling on 500, to isolate the effect of episode count while keeping
exploration and learning rate fixed.

## Prediction vs Result
**My prediction:** I expected 500 episodes to produce steadily improving
scores, since more episodes should mean more training updates for the
network to learn from.

**What happened:** Mean score before training = 492.0; after training = 780.0
(a +288 point improvement, or about +59%). See `comparison.json` for all
five before/after scores.

However, looking at the training log (`training.csv`) and dashboard
(`training_dashboard.png`), the recent-25-episode mean score did not
improve steadily. It peaked around episode 94 (~1004), then declined and
fluctuated for the rest of training, ending around 608 by episode 500.

| Game | Before | After |
|---|---|---|
| 1 | 350 | 800 |
| 2 | 500 | 790 |
| 3 | 320 | 740 |
| 4 | 800 | 650 |
| 5 | 490 | 920 |
| **Mean** | **492.0** | **780.0** |

## What the Agent Observes, Does, and Is Rewarded For
The agent observes four stacked game screens (grayscale pixel frames) as
its input. It chooses one of nine joystick actions (e.g. NOOP, UP, RIGHT,
LEFT, DOWN, and diagonals). It receives the in-game score as its reward
signal — points for eating pellets and ghosts, encouraging it to learn
which moves lead to higher scores over time.

## Gameplay GIFs
- `demos/episode_0000.gif` — before any training (essentially random play)
- `demos/episode_0100.gif` — near the training peak (~episode 94)
- `demos/episode_0250.gif` — during a mid-training dip
- `demos/episode_0500.gif` — final episode of training
- `demos/final_best.gif` — best of the 5 final evaluation games
- Additional intermediate GIFs (every 25 episodes) are included in `demos/`

## One Limitation
Training was not stable over 500 episodes. Rather than improving steadily,
the recent-mean score rose and fell repeatedly, suggesting the agent had
not converged and may have been sensitive to the fixed learning rate or
the small replay buffer size used in this notebook.

## My Next Experiment
I would change only the learning rate (e.g. lower it to 0.00005) while
keeping episodes at 500 and exploration at 0.20, to see whether a smaller
learning rate produces more stable, steadily improving performance over
the same training budget.

## Files
- `pacman_dqn.ipynb` — executed notebook with all outputs visible
- `comparison.json` — all 5 before/after evaluation scores
- `config.json` — the three chosen hyperparameters
- `training.csv` / `training_summary.json` — full training log
- `training_dashboard.png` — score/loss/exploration curves
- `demo_scores.json` — periodic demonstration scores during training
- `demos/` — gameplay GIFs (untrained, intermediate, and best trained)

## Model Checkpoints
Model weight files (`trained.pt`, `untrained.pt`, `episode_00XX.pt`) are
kept locally due to their size (~6MB each) and are not included in this
repository.

## Hardware Used
- Platform: Google Colab, T4 GPU (CUDA)
- Python 3.13.15, PyTorch 2.11.0+cu128
- Total training time: ~19.3 minutes for 500 episodes
- Total decisions: 304,000 | Total learning updates: 75,751
