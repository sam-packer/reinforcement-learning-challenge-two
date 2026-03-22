# RL Challenge 2

Minimal toy RLHF pipeline for an assignment.

The project currently does two steps:

1. Supervised fine-tuning (SFT) on the preferred responses from the dataset
2. Reward-model training on chosen vs. rejected response pairs

## Project Structure

- `src/load_data.py`: loads and filters preference data
- `src/train.py`: runs the SFT step, then the reward-model step
- `src/reward_model.py`: trains the reward model
- `data/`: local preference CSVs
- `checkpoints/`: saved model outputs
- `logs/`: saved training metrics

## Requirements

- Python 3.13
- `uv`
- CUDA

## Run

Install dependencies with `uv`, then run:

```bash
uv run train
```

## Outputs

- SFT model: `checkpoints/sft`
- Reward model: `checkpoints/reward_model`
- SFT metrics: `logs/sft/sft_metrics.json`
- Reward metrics: `logs/reward_model/reward_metrics.json`
