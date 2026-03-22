"""
Minimal supervised fine-tuning pipeline for preference data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from load_data import load_preference_data
from reward_model import RewardModelTrainConfig, train_reward_model


@dataclass(frozen=True)
class TrainConfig:
    data_path: str = "data"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: Path = Path("checkpoints/sft")
    log_dir: Path = Path("logs/sft")
    num_epochs: int = 3
    max_steps: int = -1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_length: int = 1024
    warmup_fraction: float = 0.1
    logging_steps: int = 5
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    sanity_check_tokens: int = 200

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


def print_runtime_info(precision: PrecisionConfig):
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Runtime device: CUDA ({device_name})")
        print(f"PyTorch CUDA build: {torch.version.cuda}")
    else:
        print("Runtime device: CPU")
        print("PyTorch CUDA build: None")
    print(f"Tensor dtype: {precision.model_dtype}")


def load_training_data(config: TrainConfig):
    print(f"Loading preference data from {config.data_path}")
    sft_dataset, _, _, _, metadata = load_preference_data(
        config.data_path, seed=config.seed
    )
    print(f"Source CSVs: {metadata['num_source_files']}")
    print(f"SFT dataset: {len(sft_dataset)} examples")
    print(f"Categories: {metadata['categories']}")
    return sft_dataset, metadata


def apply_chat_template_to_dataset(train_dataset, tokenizer):
    """Format prompts so instruction-tuned models see a proper chat prefix."""

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

    return train_dataset.map(format_example)


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer must define either a pad token or an eos token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(config: TrainConfig, tokenizer, precision: PrecisionConfig):
    print(f"\nLoading base model: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        dtype=precision.model_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    print(f"Model parameters: {model.num_parameters():,}")
    return model


def build_training_args(
    config: TrainConfig,
    precision: PrecisionConfig,
    train_dataset_size: int,
) -> SFTConfig:
    warmup_steps = estimate_warmup_steps(config, train_dataset_size)
    return SFTConfig(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_length=config.max_length,
        warmup_steps=warmup_steps,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        lr_scheduler_type="cosine",
        bf16=precision.bf16,
        fp16=precision.fp16,
        gradient_checkpointing=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        seed=config.seed,
        completion_only_loss=True,
    )


def estimate_warmup_steps(config: TrainConfig, train_dataset_size: int) -> int:
    if config.max_steps > 0:
        total_steps = config.max_steps
    else:
        total_steps = max(
            math.ceil(train_dataset_size / config.effective_batch_size)
            * config.num_epochs,
            1,
        )
    return max(int(total_steps * config.warmup_fraction), 0)


def build_peft_config(config: TrainConfig) -> LoraConfig | None:
    if not config.use_lora:
        print("Full fine-tuning (no LoRA)")
        return None

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    print(f"Using LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    return peft_config


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    config: TrainConfig,
    precision: PrecisionConfig,
    peft_config: LoraConfig | None,
) -> SFTTrainer:
    training_args = build_training_args(config, precision, len(train_dataset))
    return SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def print_training_summary(config: TrainConfig, train_dataset_size: int):
    print("\nStarting SFT training...")
    print(f"  Epochs:           {config.num_epochs}")
    if config.max_steps > 0:
        print(f"  Max steps:        {config.max_steps}")
    print(
        f"  Batch size:       {config.batch_size} x "
        f"{config.gradient_accumulation_steps} = {config.effective_batch_size} effective"
    )
    print(f"  Learning rate:    {config.learning_rate}")
    print(f"  Max length:       {config.max_length}")
    print(f"  Training samples: {train_dataset_size}")
    print()


def save_metrics(
    result,
    config: TrainConfig,
    train_dataset_size: int,
    metadata: dict,
):
    metrics = {
        "train_loss": result.training_loss,
        "train_runtime_s": result.metrics["train_runtime"],
        "train_samples_per_second": result.metrics["train_samples_per_second"],
        "num_examples": train_dataset_size,
        "num_epochs": config.num_epochs,
        "max_steps": config.max_steps,
        "base_model": config.base_model,
        "effective_batch_size": config.effective_batch_size,
        "learning_rate": config.learning_rate,
        "usable_pairs": metadata["usable_pairs"],
    }
    config.log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.log_dir / "sft_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Metrics saved to {metrics_path}")


def move_batch_to_model_device(batch, model):
    model_device = next(model.parameters()).device
    return {key: value.to(model_device) for key, value in batch.items()}


def run_sanity_check(model, tokenizer, prompt: str, max_new_tokens: int):
    print("\n=== Sanity Check: Sample Generation ===")
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = move_batch_to_model_device(inputs, model)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated_text[len(prompt) : len(prompt) + 300]
    print(f"Prompt:    {prompt[:100]}...")
    print(f"Generated: {completion}...")


DEFAULT_CONFIG = TrainConfig()


def train_sft(config: TrainConfig | None = None):
    config = config or DEFAULT_CONFIG
    precision = select_precision()
    print_runtime_info(precision)
    train_dataset, metadata = load_training_data(config)
    tokenizer = load_tokenizer(config.base_model)
    sample_prompt = train_dataset[0]["prompt"]
    train_dataset = apply_chat_template_to_dataset(train_dataset, tokenizer)
    model = load_model(config, tokenizer, precision)
    peft_config = build_peft_config(config)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        config=config,
        precision=precision,
        peft_config=peft_config,
    )

    print_training_summary(config, len(train_dataset))
    result = trainer.train()

    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    print(f"\nModel saved to {config.output_dir}")

    save_metrics(result, config, len(train_dataset), metadata)
    run_sanity_check(
        model=trainer.model,
        tokenizer=tokenizer,
        prompt=tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": sample_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ),
        max_new_tokens=config.sanity_check_tokens,
    )

    return {
        "output_dir": config.output_dir,
        "num_examples": len(train_dataset),
    }


def main():
    sft_config = DEFAULT_CONFIG
    reward_config = RewardModelTrainConfig(
        data_path=sft_config.data_path,
        base_model=sft_config.base_model,
    )

    train_sft(sft_config)
    train_reward_model(reward_config)


if __name__ == "__main__":
    main()
