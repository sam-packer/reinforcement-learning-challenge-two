"""
Utilities for turning preference annotations into train/validation datasets.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split


def load_preference_data(
    data_path: str,
    val_fraction: float = 0.2,
    min_reading_time: float = 2.0,
    max_reading_time: float = 600.0,
    seed: int = 42,
) -> tuple[dict[str, Dataset], dict[str, Any]]:
    """
    Load preference annotations, filter noisy rows, and build train/validation views
    for SFT, reward modeling, and PPO.
    """
    df, source_files = load_preference_frame(data_path)
    preference_df, filter_stats = filter_preference_rows(
        df,
        min_reading_time=min_reading_time,
        max_reading_time=max_reading_time,
    )
    train_df, val_df = split_preference_frame(
        preference_df,
        val_fraction=val_fraction,
        seed=seed,
    )

    datasets = {
        "sft_train": build_sft_dataset(train_df),
        "sft_val": build_sft_dataset(val_df),
        "reward_train": build_reward_dataset(train_df),
        "reward_val": build_reward_dataset(val_df),
        "ppo_train": build_ppo_dataset(train_df),
        "ppo_val": build_ppo_dataset(val_df),
    }
    metadata = build_metadata(
        full_df=df,
        preference_df=preference_df,
        train_df=train_df,
        val_df=val_df,
        filter_stats=filter_stats,
        datasets=datasets,
        source_files=source_files,
        val_fraction=val_fraction,
    )

    return datasets, metadata


def load_preference_frame(data_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load either a single CSV or every CSV in a directory, concatenated in name order.
    """
    path = Path(data_path)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")
        frames = [pd.read_csv(csv_file) for csv_file in csv_files]
        return pd.concat(frames, ignore_index=True), [
            str(csv_file) for csv_file in csv_files
        ]

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file or directory of CSVs, got: {path}")
        return pd.read_csv(path), [str(path)]

    raise FileNotFoundError(f"Data path does not exist: {path}")


def filter_preference_rows(
    df: pd.DataFrame,
    min_reading_time: float,
    max_reading_time: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only non-tie, reasonably trustworthy annotations."""
    preference_df = df[df["is_tie"] == 0].copy()
    ties_removed = int((df["is_tie"] == 1).sum())

    too_fast = preference_df["reading_time_s"] < min_reading_time
    too_slow = preference_df["reading_time_s"] > max_reading_time
    quality_mask = ~(too_fast | too_slow)
    filtered_df = preference_df.loc[quality_mask].copy().reset_index(drop=True)

    stats = {
        "ties_removed": ties_removed,
        "quality_filtered": int((~quality_mask).sum()),
        "too_fast": int(too_fast.sum()),
        "too_slow": int(too_slow.sum()),
    }
    return filtered_df, stats


def split_preference_frame(
    preference_df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one shared train/validation split for the whole pipeline."""
    if len(preference_df) < 2 or val_fraction <= 0:
        empty_df = preference_df.iloc[0:0].copy()
        return preference_df.reset_index(drop=True), empty_df.reset_index(drop=True)

    stratify_labels = get_stratify_labels(
        preference_df["prompt_category"], val_fraction
    )
    train_df, val_df = train_test_split(
        preference_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify_labels,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def get_stratify_labels(categories: pd.Series, val_fraction: float) -> list[str] | None:
    """
    Stratify when the split is large enough to represent each category.
    Fall back to a random split for tiny datasets or sparse categories.
    """
    category_counts = categories.value_counts()
    if category_counts.empty or int(category_counts.min()) < 2:
        return None

    val_size = math.ceil(len(categories) * val_fraction)
    train_size = len(categories) - val_size
    num_categories = len(category_counts)
    if val_size < num_categories or train_size < num_categories:
        return None

    return categories.tolist()


def build_sft_dataset(preference_df: pd.DataFrame) -> Dataset:
    """Use chosen responses as supervised targets."""
    records = preference_df.loc[:, ["prompt", "chosen", "prompt_category"]].rename(
        columns={"chosen": "completion", "prompt_category": "category"}
    )
    return Dataset.from_list(records.to_dict("records"))


def build_reward_dataset(preference_df: pd.DataFrame) -> Dataset:
    """Create chosen/rejected pairs for reward-model training and evaluation."""
    reward_columns = [
        "prompt",
        "chosen",
        "rejected",
        "prompt_category",
        "chosen_temperature",
        "rejected_temperature",
        "chosen_output_tokens",
        "rejected_output_tokens",
    ]
    reward_records = preference_df.loc[:, reward_columns].rename(
        columns={"prompt_category": "category"}
    )
    return Dataset.from_list(reward_records.to_dict("records"))


def build_ppo_dataset(preference_df: pd.DataFrame) -> Dataset:
    """Deduplicate prompts for on-policy sampling."""
    unique_prompts = preference_df.drop_duplicates(subset="prompt_id")
    records = unique_prompts.loc[:, ["prompt", "prompt_category"]].rename(
        columns={"prompt_category": "category"}
    )
    return Dataset.from_list(records.to_dict("records"))


def build_metadata(
    full_df: pd.DataFrame,
    preference_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    filter_stats: dict[str, int],
    datasets: dict[str, Dataset],
    source_files: list[str],
    val_fraction: float,
) -> dict[str, Any]:
    """Collect simple reporting numbers for the full pipeline."""
    return {
        "source_files": source_files,
        "num_source_files": len(source_files),
        "total_rows": len(full_df),
        "val_fraction": val_fraction,
        **filter_stats,
        "usable_pairs": len(preference_df),
        "train_pairs": len(train_df),
        "val_pairs": len(val_df),
        "sft_train_examples": len(datasets["sft_train"]),
        "sft_val_examples": len(datasets["sft_val"]),
        "reward_train_examples": len(datasets["reward_train"]),
        "reward_val_examples": len(datasets["reward_val"]),
        "ppo_train_prompts": len(datasets["ppo_train"]),
        "ppo_val_prompts": len(datasets["ppo_val"]),
        "categories": preference_df["prompt_category"].value_counts().to_dict(),
        "train_categories": train_df["prompt_category"].value_counts().to_dict(),
        "val_categories": val_df["prompt_category"].value_counts().to_dict(),
        "avg_chosen_tokens": preference_df["chosen_output_tokens"].mean(),
        "avg_rejected_tokens": preference_df["rejected_output_tokens"].mean(),
        "median_reading_time_s": preference_df["reading_time_s"].median(),
    }


if __name__ == "__main__":
    import json

    datasets, metadata = load_preference_data("data")

    print("=== Dataset Sizes ===")
    print(f"  SFT train:      {len(datasets['sft_train'])} examples")
    print(f"  SFT val:        {len(datasets['sft_val'])} examples")
    print(f"  Reward train:   {len(datasets['reward_train'])} pairs")
    print(f"  Reward val:     {len(datasets['reward_val'])} pairs")
    print(f"  PPO train:      {len(datasets['ppo_train'])} prompts")
    print(f"  PPO val:        {len(datasets['ppo_val'])} prompts")

    print("\n=== Metadata ===")
    print(json.dumps(metadata, indent=2))

    print("\n=== SFT sample ===")
    sample = datasets["sft_train"][0]
    print(f"  Prompt:     {sample['prompt'][:80]}...")
    print(f"  Completion: {sample['completion'][:80]}...")

    print("\n=== Reward sample ===")
    sample = datasets["reward_train"][0]
    print(f"  Prompt:     {sample['prompt'][:80]}...")
    print(f"  Chosen:     {sample['chosen'][:80]}...")
    print(f"  Rejected:   {sample['rejected'][:80]}...")

    print("\n=== PPO sample ===")
    sample = datasets["ppo_train"][0]
    print(f"  Prompt:     {sample['prompt'][:80]}...")
