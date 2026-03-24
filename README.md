# RL Challenge 2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sam-packer/reinforcement-learning-challenge-two/blob/master/colab.ipynb)

RLHF training pipeline built around three stages:

1. supervised fine-tuning on preferred responses
2. reward-model training on chosen vs. rejected pairs
3. PPO alignment against the learned reward model

The repo keeps the pipeline small and direct: a shared data loader, one top-level training entrypoint, one reward-model module, and one PPO module.

## Pipeline

`preference CSV -> filtering -> shared train/validation split -> SFT -> reward model -> PPO`

Each stage has a separate role:

- SFT trains the assistant to imitate preferred completions
- the reward model learns to rank preferred responses above rejected ones
- PPO updates the assistant policy using reward-model scores while penalizing drift from the SFT reference policy

## Project Structure

- `src/load_data.py`: loads preference CSVs, filters noisy rows, and builds SFT, reward-model, and PPO datasets
- `src/train.py`: runs the full training pipeline end to end
- `src/reward_model.py`: reward-model training and evaluation
- `src/ppo.py`: PPO rollout, reward scoring, and policy optimization
- `src/learning_curve.py`: learning curve analysis — trains reward models at increasing data fractions to show how accuracy scales with dataset size
- `docs/rlhf_learning_guide.md`: detailed explanation of the pipeline, metrics, and design decisions

## Requirements

- Python 3.13
- `uv`
- NVIDIA GPU with CUDA

## Setup

Requires a CUDA GPU (Linux or Windows). Recommended to run on Google Colab or a machine with an NVIDIA GPU.

```bash
uv sync
```

## Run

```bash
uv run train
```

## Checkpoints And Logs

- SFT checkpoint: `checkpoints/sft`
- reward-model checkpoint: `checkpoints/reward_model`
- PPO checkpoint: `checkpoints/ppo`
- SFT metrics: `logs/sft/sft_metrics.json`
- reward-model metrics: `logs/reward_model/reward_metrics.json`
- PPO metrics: `logs/ppo/ppo_metrics.json`
- learning curve results: `logs/learning_curve/learning_curve.json`

## Metrics

### SFT

- training loss
- validation loss
- validation perplexity

### Reward Model

- training loss
- validation loss
- pairwise accuracy on held-out comparisons
- mean chosen-minus-rejected score margin

### PPO

- mean reward-model score on PPO rollouts
- mean KL from the PPO policy to the frozen SFT reference
- mean total reward after KL penalty
- mean PPO policy loss
- mean clip fraction
- held-out PPO win rate vs the frozen SFT policy

### Learning Curve

Trains the reward model at 20%, 40%, 60%, 80%, and 100% of the training data against a fixed validation set. Reports:

- overall pairwise accuracy at each fraction
- per-category accuracy (e.g. factual vs empathy) to show which categories are data-starved
- mean chosen-minus-rejected margin

If accuracy is still climbing steeply at 100%, more data is needed. If it's flattening, you're near saturation.

## PPO Implementation Note

- rollout generation and frozen-policy scoring use `torch.no_grad()`, not `torch.inference_mode()`
- PPO reuses sampled rollout tensors in later gradient-tracked policy updates, and `inference_mode` tensors are too restrictive for that use

## Inference Roles

- `checkpoints/reward_model` is a scorer, not the chat model
- `checkpoints/ppo` is the final aligned policy checkpoint for generation
- `checkpoints/sft` is the pre-PPO reference policy and also a usable generation checkpoint
