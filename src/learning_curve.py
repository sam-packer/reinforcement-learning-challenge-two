"""
Learning curve analysis: train reward models at increasing data fractions
to see how pairwise accuracy scales with dataset size.

Runs after the main pipeline. Uses a fixed validation set across all fractions
so results are directly comparable. Reports overall and per-category accuracy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl.trainer.reward_config import RewardConfig
from trl.trainer.reward_trainer import RewardTrainer

from load_data import (
    build_reward_dataset,
    filter_preference_rows,
    load_preference_frame,
    split_preference_frame,
)
from reward_model import apply_chat_template_to_reward_dataset, batch_reward_scores

DATA_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]


def select_precision():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


def evaluate_per_category(model, tokenizer, eval_dataset, max_length=1024, batch_size=8):
    """Pairwise accuracy overall and per category on a fixed validation set."""
    model.eval()
    chosen_scores = []
    rejected_scores = []
    categories = []

    for start in range(0, len(eval_dataset), batch_size):
        stop = min(start + batch_size, len(eval_dataset))
        batch = eval_dataset[start:stop]
        chosen_texts = [
            p + c for p, c in zip(batch["prompt"], batch["chosen"], strict=False)
        ]
        rejected_texts = [
            p + r for p, r in zip(batch["prompt"], batch["rejected"], strict=False)
        ]
        chosen_scores.append(
            batch_reward_scores(model, tokenizer, chosen_texts, max_length)
        )
        rejected_scores.append(
            batch_reward_scores(model, tokenizer, rejected_texts, max_length)
        )
        categories.extend(batch["category"])

    chosen = torch.cat(chosen_scores)
    rejected = torch.cat(rejected_scores)
    margins = chosen - rejected
    correct = margins > 0

    category_results = {}
    for cat in set(categories):
        mask = torch.tensor([c == cat for c in categories])
        cat_correct = correct[mask]
        category_results[cat] = {
            "accuracy": cat_correct.float().mean().item(),
            "count": int(cat_correct.shape[0]),
        }

    return {
        "overall_accuracy": correct.float().mean().item(),
        "mean_margin": margins.mean().item(),
        "per_category": category_results,
    }


def subsample_by_category(train_df, fraction, seed):
    """Stratified subsample preserving category distribution."""
    sampled_parts = []
    for _, group in train_df.groupby("prompt_category"):
        n = max(int(len(group) * fraction), 1)
        sampled_parts.append(group.sample(n=n, random_state=seed))
    import pandas as pd
    return pd.concat(sampled_parts).reset_index(drop=True)


def train_reward_model_at_fraction(
    train_subset, tokenizer, base_model, model_dtype, bf16, fp16, seed, fraction,
):
    """Train a throwaway reward model on a data subset and return the trained model."""
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=1, dtype=model_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["score"],
    )

    effective_batch_size = 16  # 4 * 4
    total_steps = max(math.ceil(len(train_subset) / effective_batch_size) * 3, 1)
    warmup_steps = max(int(total_steps * 0.1), 0)

    training_args = RewardConfig(
        output_dir=f"checkpoints/learning_curve/{int(fraction * 100)}pct",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_length=1024,
        warmup_steps=warmup_steps,
        logging_steps=999999,  # suppress per-step logs
        save_strategy="no",
        eval_strategy="no",
        lr_scheduler_type="cosine",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
        seed=seed,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    return trainer.model


def print_results_table(results):
    """Print a clear summary table."""
    categories = sorted(results[0]["per_category"].keys())

    header = f"{'Fraction':>10} {'Pairs':>6} {'Overall':>8}"
    for cat in categories:
        label = cat[:14]
        header += f"  {label:>14}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = f"{r['fraction']:>9.0%} {r['train_pairs']:>6} {r['overall_accuracy']:>7.1%}"
        for cat in categories:
            cat_data = r["per_category"].get(cat, {})
            acc = cat_data.get("accuracy", 0)
            n = cat_data.get("count", 0)
            line += f"  {acc:>7.1%} (n={n:<3})"
        print(line)


def run_learning_curve(
    data_path: str = "data",
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    seed: int = 42,
    log_dir: Path = Path("logs/learning_curve"),
):
    """Train reward models at increasing data fractions and measure accuracy."""
    model_dtype, bf16, fp16 = select_precision()

    df, _ = load_preference_frame(data_path)
    preference_df, _ = filter_preference_rows(
        df, min_reading_time=2.0, max_reading_time=600.0,
    )
    train_df, val_df = split_preference_frame(
        preference_df, val_fraction=0.2, seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    val_dataset = build_reward_dataset(val_df)
    val_dataset = apply_chat_template_to_reward_dataset(val_dataset, tokenizer)

    print("\n" + "=" * 60)
    print("LEARNING CURVE ANALYSIS")
    print("=" * 60)
    print(f"Total usable pairs: {len(preference_df)}")
    print(f"Training pairs:     {len(train_df)}")
    print(f"Validation pairs:   {len(val_df)} (fixed across all fractions)")
    print(f"Data fractions:     {DATA_FRACTIONS}")

    train_categories = train_df["prompt_category"].value_counts().to_dict()
    print(f"Train categories:   {train_categories}")
    print()

    results = []

    for fraction in DATA_FRACTIONS:
        if fraction >= 1.0:
            sampled_df = train_df
        else:
            sampled_df = subsample_by_category(train_df, fraction, seed)

        train_subset = build_reward_dataset(sampled_df)
        train_subset = apply_chat_template_to_reward_dataset(train_subset, tokenizer)

        print(f"Training with {fraction:.0%} of data ({len(sampled_df)} pairs)...")

        model = train_reward_model_at_fraction(
            train_subset, tokenizer, base_model,
            model_dtype, bf16, fp16, seed, fraction,
        )

        eval_result = evaluate_per_category(model, tokenizer, val_dataset)
        eval_result["fraction"] = fraction
        eval_result["train_pairs"] = len(sampled_df)
        results.append(eval_result)

        print(f"  -> Overall accuracy: {eval_result['overall_accuracy']:.1%}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("LEARNING CURVE RESULTS")
    print("=" * 60)
    print_results_table(results)
    print()

    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = log_dir / "learning_curve.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results
