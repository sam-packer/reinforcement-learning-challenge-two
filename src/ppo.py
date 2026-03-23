"""
Minimal PPO loop for a toy RLHF pipeline.

This implementation keeps the policy and frozen reference as two adapters on the
same base model so PPO stays explainable and memory-aware.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from load_data import load_preference_data


@dataclass(frozen=True)
class PPOTrainConfig:
    data_path: str = "data"
    policy_path: Path = Path("checkpoints/sft")
    reward_model_path: Path = Path("checkpoints/reward_model")
    output_dir: Path = Path("checkpoints/ppo")
    log_dir: Path = Path("logs/ppo")
    num_epochs: int = 2
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_prompt_length: int = 512
    max_response_length: int = 128
    ppo_updates_per_batch: int = 2
    clip_range: float = 0.2
    kl_coefficient: float = 0.05
    max_grad_norm: float = 1.0
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 42


@dataclass(frozen=True)
class PrecisionConfig:
    model_dtype: torch.dtype
    bf16: bool
    fp16: bool


def select_precision() -> PrecisionConfig:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return PrecisionConfig(model_dtype=torch.bfloat16, bf16=True, fp16=False)
    if torch.cuda.is_available():
        return PrecisionConfig(model_dtype=torch.float16, bf16=False, fp16=True)
    return PrecisionConfig(model_dtype=torch.float32, bf16=False, fp16=False)


def load_ppo_data(config: PPOTrainConfig):
    datasets, metadata = load_preference_data(config.data_path, seed=config.seed)
    return datasets["ppo_train"], datasets["ppo_val"], metadata


def load_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer must define either a pad token or an eos token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_model(config: PPOTrainConfig, tokenizer, precision: PrecisionConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_config = PeftConfig.from_pretrained(config.policy_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        dtype=precision.model_dtype,
    )
    policy_model = PeftModel.from_pretrained(
        base_model,
        config.policy_path,
        adapter_name="ppo",
        is_trainable=True,
    )
    policy_model.load_adapter(
        config.policy_path,
        adapter_name="reference",
        is_trainable=False,
    )
    policy_model.set_adapter("ppo")
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    if policy_model.generation_config is not None:
        policy_model.generation_config.pad_token_id = tokenizer.pad_token_id
    return policy_model.to(device)


def load_reward_model(config: PPOTrainConfig, precision: PrecisionConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    peft_config = PeftConfig.from_pretrained(config.reward_model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.reward_model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer must define either a pad token or an eos token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=1,
        dtype=precision.model_dtype,
    )
    reward_model = PeftModel.from_pretrained(
        base_model,
        config.reward_model_path,
        is_trainable=False,
    )
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad = False
    return reward_model.to(device), tokenizer


def format_prompts(prompts: list[str], tokenizer) -> list[str]:
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return formatted_prompts


def tokenize_prompts(
    tokenizer, prompts: list[str], max_prompt_length: int, device: torch.device
):
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


def generate_responses(model, tokenizer, prompts: list[str], config: PPOTrainConfig):
    model_device = next(model.parameters()).device
    formatted_prompts = format_prompts(prompts, tokenizer)
    prompt_batch = tokenize_prompts(
        tokenizer,
        formatted_prompts,
        config.max_prompt_length,
        model_device,
    )
    prompt_width = prompt_batch["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **prompt_batch,
            max_new_tokens=config.max_response_length,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response_texts = []
    for response_ids in outputs[:, prompt_width:]:
        trimmed_ids = response_ids[response_ids != tokenizer.pad_token_id]
        response_texts.append(tokenizer.decode(trimmed_ids, skip_special_tokens=True))

    return formatted_prompts, outputs, prompt_width, response_texts


def response_logprobs(
    model, sequences: torch.Tensor, response_start_index: int, pad_token_id: int
):
    attention_mask = (sequences != pad_token_id).long()
    logits = model(input_ids=sequences, attention_mask=attention_mask).logits
    shifted_logits = logits[:, :-1, :]
    shifted_targets = sequences[:, 1:]
    shifted_positions = torch.arange(
        1, sequences.shape[1], device=sequences.device
    ).unsqueeze(0)
    response_mask = (shifted_positions >= response_start_index) & (
        shifted_targets != pad_token_id
    )

    token_logprobs = (
        torch.log_softmax(shifted_logits, dim=-1)
        .gather(
            dim=-1,
            index=shifted_targets.unsqueeze(-1),
        )
        .squeeze(-1)
    )
    return token_logprobs, response_mask


def sequence_means(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.float()
    counts = mask_float.sum(dim=1).clamp_min(1.0)
    return (values * mask_float).sum(dim=1) / counts


def score_completions(
    reward_model,
    reward_tokenizer,
    formatted_prompts: list[str],
    completions: list[str],
    max_length: int,
):
    device = next(reward_model.parameters()).device
    texts = [
        prompt + completion
        for prompt, completion in zip(formatted_prompts, completions, strict=False)
    ]
    batch = reward_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        scores = reward_model(**batch).logits.squeeze(-1)
    return scores.detach().float()


def build_advantages(
    rewards: torch.Tensor, running_reward_mean: float
) -> tuple[torch.Tensor, float]:
    centered_rewards = rewards - running_reward_mean
    if rewards.numel() > 1:
        centered_rewards = (centered_rewards - centered_rewards.mean()) / (
            centered_rewards.std(unbiased=False) + 1e-8
        )
    updated_running_mean = (0.9 * running_reward_mean) + (0.1 * rewards.mean().item())
    return centered_rewards, updated_running_mean


def rollout_step(
    policy_model,
    tokenizer,
    reward_model,
    reward_tokenizer,
    prompts: list[str],
    config: PPOTrainConfig,
):
    formatted_prompts, sequences, response_start_index, completions = (
        generate_responses(
            policy_model,
            tokenizer,
            prompts,
            config,
        )
    )

    policy_model.set_adapter("ppo")
    with torch.no_grad():
        old_token_logprobs, response_mask = response_logprobs(
            policy_model,
            sequences,
            response_start_index,
            tokenizer.pad_token_id,
        )
        policy_model.set_adapter("reference")
        reference_token_logprobs, _ = response_logprobs(
            policy_model,
            sequences,
            response_start_index,
            tokenizer.pad_token_id,
        )
    policy_model.set_adapter("ppo")

    reward_scores = score_completions(
        reward_model,
        reward_tokenizer,
        formatted_prompts,
        completions,
        config.max_prompt_length + config.max_response_length,
    )
    mean_kl = sequence_means(
        old_token_logprobs - reference_token_logprobs, response_mask
    )
    total_rewards = reward_scores.to(sequences.device) - (
        config.kl_coefficient * mean_kl
    )

    return {
        "sequences": sequences.detach().clone(),
        "response_start_index": response_start_index,
        "completions": completions,
        "formatted_prompts": formatted_prompts,
        "old_token_logprobs": old_token_logprobs.detach().clone(),
        "reference_token_logprobs": reference_token_logprobs.detach().clone(),
        "response_mask": response_mask.detach().clone(),
        "reward_scores": reward_scores.to(sequences.device),
        "mean_kl": mean_kl.detach().clone(),
        "total_rewards": total_rewards.detach().clone(),
        "response_lengths": response_mask.sum(dim=1).detach().clone(),
    }


def ppo_update(
    policy_model,
    optimizer,
    rollout: dict[str, torch.Tensor | list[str]],
    advantages: torch.Tensor,
    config: PPOTrainConfig,
):
    losses = []
    clip_fractions = []

    sequences = rollout["sequences"]
    response_start_index = rollout["response_start_index"]
    old_token_logprobs = rollout["old_token_logprobs"]
    response_mask = rollout["response_mask"]

    for _ in range(config.ppo_updates_per_batch):
        policy_model.train()
        policy_model.set_adapter("ppo")
        new_token_logprobs, _ = response_logprobs(
            policy_model,
            sequences,
            response_start_index=response_start_index,
            pad_token_id=policy_model.config.pad_token_id,
        )

        logprob_ratio = torch.exp(new_token_logprobs - old_token_logprobs)
        token_advantages = advantages.unsqueeze(1)
        unclipped_objective = logprob_ratio * token_advantages
        clipped_ratio = torch.clamp(
            logprob_ratio,
            1.0 - config.clip_range,
            1.0 + config.clip_range,
        )
        clipped_objective = clipped_ratio * token_advantages

        mask_float = response_mask.float()
        policy_loss = -(
            torch.minimum(unclipped_objective, clipped_objective) * mask_float
        ).sum() / mask_float.sum().clamp_min(1.0)

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
        optimizer.step()

        losses.append(policy_loss.detach().item())
        clipped_tokens = (logprob_ratio > 1.0 + config.clip_range) | (
            logprob_ratio < 1.0 - config.clip_range
        )
        clip_fraction = (
            clipped_tokens & response_mask
        ).float().sum() / mask_float.sum().clamp_min(1.0)
        clip_fractions.append(clip_fraction.item())

    return {
        "policy_loss": sum(losses) / len(losses),
        "clip_fraction": sum(clip_fractions) / len(clip_fractions),
    }


def evaluate_policy(
    policy_model,
    tokenizer,
    reward_model,
    reward_tokenizer,
    eval_dataset,
    config: PPOTrainConfig,
) -> dict[str, float] | None:
    if len(eval_dataset) == 0:
        return None

    prompts = [eval_dataset[index]["prompt"] for index in range(len(eval_dataset))]
    model_device = next(policy_model.parameters()).device
    formatted_prompts = format_prompts(prompts, tokenizer)
    prompt_batch = tokenize_prompts(
        tokenizer,
        formatted_prompts,
        config.max_prompt_length,
        model_device,
    )
    prompt_width = prompt_batch["input_ids"].shape[1]

    def generate_with_adapter(adapter_name: str) -> list[str]:
        policy_model.eval()
        policy_model.set_adapter(adapter_name)
        with torch.inference_mode():
            outputs = policy_model.generate(
                **prompt_batch,
                max_new_tokens=config.max_response_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = []
        for response_ids in outputs[:, prompt_width:]:
            trimmed_ids = response_ids[response_ids != tokenizer.pad_token_id]
            completions.append(tokenizer.decode(trimmed_ids, skip_special_tokens=True))
        return completions

    reference_completions = generate_with_adapter("reference")
    ppo_completions = generate_with_adapter("ppo")
    policy_model.set_adapter("ppo")

    reference_scores = score_completions(
        reward_model,
        reward_tokenizer,
        formatted_prompts,
        reference_completions,
        config.max_prompt_length + config.max_response_length,
    )
    ppo_scores = score_completions(
        reward_model,
        reward_tokenizer,
        formatted_prompts,
        ppo_completions,
        config.max_prompt_length + config.max_response_length,
    )

    return {
        "reference_reward_mean": reference_scores.mean().item(),
        "ppo_reward_mean": ppo_scores.mean().item(),
        "win_rate_vs_sft": (ppo_scores > reference_scores).float().mean().item(),
    }


def save_metrics(
    metrics: dict[str, float | int | dict[str, float]], config: PPOTrainConfig
):
    config.log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.log_dir / "ppo_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"PPO metrics saved to {metrics_path}")


def train_ppo(config: PPOTrainConfig | None = None):
    config = config or PPOTrainConfig()
    precision = select_precision()
    train_dataset, eval_dataset, metadata = load_ppo_data(config)

    if len(train_dataset) == 0:
        raise ValueError("PPO needs at least one training prompt.")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    tokenizer = load_tokenizer(config.policy_path)
    policy_model = load_policy_model(config, tokenizer, precision)
    reward_model, reward_tokenizer = load_reward_model(config, precision)
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in policy_model.parameters()
            if parameter.requires_grad
        ],
        lr=config.learning_rate,
    )

    print("\nStarting PPO training...")
    print(f"  Train prompts:    {len(train_dataset)}")
    print(f"  Validation set:   {len(eval_dataset)}")
    print(f"  Epochs:           {config.num_epochs}")
    print(f"  Batch size:       {config.batch_size}")
    print(f"  Learning rate:    {config.learning_rate}")
    print(f"  KL coefficient:   {config.kl_coefficient}")
    print()

    running_reward_mean = 0.0
    metric_history = {
        "reward_model_score": [],
        "mean_kl": [],
        "total_reward": [],
        "response_tokens": [],
        "policy_loss": [],
        "clip_fraction": [],
    }

    prompt_indices = list(range(len(train_dataset)))
    total_steps = 0
    steps_per_epoch = (len(prompt_indices) + config.batch_size - 1) // config.batch_size
    progress_bar = tqdm(
        total=config.num_epochs * steps_per_epoch,
        desc="PPO",
        dynamic_ncols=True,
    )
    try:
        for epoch in range(config.num_epochs):
            random.shuffle(prompt_indices)
            for start in range(0, len(prompt_indices), config.batch_size):
                batch_indices = prompt_indices[start : start + config.batch_size]
                prompts = [train_dataset[index]["prompt"] for index in batch_indices]
                rollout = rollout_step(
                    policy_model,
                    tokenizer,
                    reward_model,
                    reward_tokenizer,
                    prompts,
                    config,
                )
                advantages, running_reward_mean = build_advantages(
                    rollout["total_rewards"],
                    running_reward_mean,
                )
                update_metrics = ppo_update(
                    policy_model,
                    optimizer,
                    rollout,
                    advantages,
                    config,
                )

                reward_score = rollout["reward_scores"].mean().item()
                mean_kl = rollout["mean_kl"].mean().item()
                total_reward = rollout["total_rewards"].mean().item()
                response_tokens = rollout["response_lengths"].float().mean().item()

                metric_history["reward_model_score"].append(reward_score)
                metric_history["mean_kl"].append(mean_kl)
                metric_history["total_reward"].append(total_reward)
                metric_history["response_tokens"].append(response_tokens)
                metric_history["policy_loss"].append(update_metrics["policy_loss"])
                metric_history["clip_fraction"].append(update_metrics["clip_fraction"])
                total_steps += 1

                progress_bar.set_postfix(
                    epoch=epoch + 1,
                    reward=f"{reward_score:.3f}",
                    kl=f"{mean_kl:.4f}",
                    clip=f"{update_metrics['clip_fraction']:.3f}",
                )
                progress_bar.update(1)
    finally:
        progress_bar.close()

    eval_metrics = evaluate_policy(
        policy_model,
        tokenizer,
        reward_model,
        reward_tokenizer,
        eval_dataset,
        config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    policy_model.save_pretrained(str(config.output_dir), selected_adapters=["ppo"])
    tokenizer.save_pretrained(str(config.output_dir))
    print(f"PPO policy saved to {config.output_dir}")

    metrics = {
        "base_policy": str(config.policy_path),
        "reward_model": str(config.reward_model_path),
        "ppo_train_prompts": len(train_dataset),
        "ppo_val_prompts": len(eval_dataset),
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "clip_range": config.clip_range,
        "kl_coefficient": config.kl_coefficient,
        "total_steps": total_steps,
        "mean_reward_model_score": sum(metric_history["reward_model_score"])
        / max(len(metric_history["reward_model_score"]), 1),
        "mean_kl": sum(metric_history["mean_kl"])
        / max(len(metric_history["mean_kl"]), 1),
        "mean_total_reward": sum(metric_history["total_reward"])
        / max(len(metric_history["total_reward"]), 1),
        "mean_response_tokens": sum(metric_history["response_tokens"])
        / max(len(metric_history["response_tokens"]), 1),
        "mean_policy_loss": sum(metric_history["policy_loss"])
        / max(len(metric_history["policy_loss"]), 1),
        "mean_clip_fraction": sum(metric_history["clip_fraction"])
        / max(len(metric_history["clip_fraction"]), 1),
        "usable_pairs": metadata["usable_pairs"],
    }
    if eval_metrics is not None:
        metrics["eval"] = eval_metrics

    save_metrics(metrics, config)

    return {
        "output_dir": config.output_dir,
        "steps": total_steps,
        "eval_metrics": eval_metrics,
    }
