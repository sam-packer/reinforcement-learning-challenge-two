"""
Minimal reward-model training for preference pairs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from load_data import load_preference_data


@dataclass(frozen=True)
class RewardModelTrainConfig:
    data_path: str = "data"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: Path = Path("checkpoints/reward_model")
    log_dir: Path = Path("logs/reward_model")
    num_epochs: int = 1
    max_steps: int = -1
    batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_length: int = 1024
    warmup_fraction: float = 0.1
    logging_steps: int = 5
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


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


def load_reward_data(config: RewardModelTrainConfig):
    _, reward_train, reward_val, _, metadata = load_preference_data(
        config.data_path,
        seed=config.seed,
    )
    return reward_train, reward_val, metadata


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define either a pad token or an eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def apply_chat_template_to_reward_dataset(dataset, tokenizer):
    """Format prompts so the reward model scores assistant completions in chat context."""

    def format_example(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example["prompt"]},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": formatted_prompt}

    return dataset.map(format_example)


def load_model(config: RewardModelTrainConfig, tokenizer, precision: PrecisionConfig):
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=1,
        torch_dtype=precision.model_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def estimate_warmup_steps(config: RewardModelTrainConfig, train_dataset_size: int) -> int:
    if config.max_steps > 0:
        total_steps = config.max_steps
    else:
        total_steps = max(
            math.ceil(train_dataset_size / config.effective_batch_size) * config.num_epochs,
            1,
        )
    return max(int(total_steps * config.warmup_fraction), 0)


def build_peft_config(config: RewardModelTrainConfig) -> LoraConfig | None:
    if not config.use_lora:
        return None

    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )


def build_training_args(
    config: RewardModelTrainConfig,
    precision: PrecisionConfig,
    train_dataset_size: int,
) -> RewardConfig:
    warmup_steps = estimate_warmup_steps(config, train_dataset_size)
    return RewardConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_length=config.max_length,
        warmup_steps=warmup_steps,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        lr_scheduler_type="cosine",
        bf16=precision.bf16,
        fp16=precision.fp16,
        gradient_checkpointing=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        seed=config.seed,
    )


def save_metrics(
    train_result,
    eval_metrics: dict[str, float] | None,
    config: RewardModelTrainConfig,
    reward_train_size: int,
    reward_val_size: int,
    metadata: dict,
):
    metrics = {
        "train_loss": train_result.training_loss,
        "train_runtime_s": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        "reward_train_examples": reward_train_size,
        "reward_val_examples": reward_val_size,
        "num_epochs": config.num_epochs,
        "max_steps": config.max_steps,
        "base_model": config.base_model,
        "effective_batch_size": config.effective_batch_size,
        "learning_rate": config.learning_rate,
        "usable_pairs": metadata["usable_pairs"],
    }
    if eval_metrics is not None:
        metrics["eval"] = eval_metrics

    config.log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.log_dir / "reward_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Reward metrics saved to {metrics_path}")


def train_reward_model(config: RewardModelTrainConfig | None = None):
    config = config or RewardModelTrainConfig()
    precision = select_precision()
    reward_train, reward_val, metadata = load_reward_data(config)
    tokenizer = load_tokenizer(config.base_model)
    reward_train = apply_chat_template_to_reward_dataset(reward_train, tokenizer)
    reward_val = apply_chat_template_to_reward_dataset(reward_val, tokenizer)
    model = load_model(config, tokenizer, precision)
    peft_config = build_peft_config(config)
    training_args = build_training_args(config, precision, len(reward_train))

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=reward_train,
        eval_dataset=reward_val if len(reward_val) > 0 else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("\nStarting reward-model training...")
    print(f"  Train pairs:      {len(reward_train)}")
    print(f"  Validation pairs: {len(reward_val)}")
    print(f"  Learning rate:    {config.learning_rate}")
    print(f"  Max length:       {config.max_length}")
    print()

    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if len(reward_val) > 0 else None

    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    print(f"Reward model saved to {config.output_dir}")

    save_metrics(
        train_result=train_result,
        eval_metrics=eval_metrics,
        config=config,
        reward_train_size=len(reward_train),
        reward_val_size=len(reward_val),
        metadata=metadata,
    )

    return {
        "output_dir": config.output_dir,
        "train_size": len(reward_train),
        "val_size": len(reward_val),
        "eval_metrics": eval_metrics,
    }
