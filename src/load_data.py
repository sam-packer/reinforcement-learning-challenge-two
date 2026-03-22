"""
Utilities for turning preference annotations into training datasets.
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
    reward_val_fraction: float = 0.2,
    min_reading_time: float = 2.0,
    max_reading_time: float = 600.0,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset, Dataset, dict[str, Any]]:
    """
    Load and split preference data for simple RLHF-style experiments.
    """
    df, source_files = load_preference_frame(data_path)
    preference_df, filter_stats = filter_preference_rows(
        df,
        min_reading_time=min_reading_time,
        max_reading_time=max_reading_time,
    )

    sft_dataset = build_sft_dataset(preference_df)
    reward_train, reward_val = build_reward_datasets(
        preference_df,
        reward_val_fraction=reward_val_fraction,
        seed=seed,
    )
    ppo_dataset = build_ppo_dataset(preference_df)
    metadata = build_metadata(
        full_df=df,
        preference_df=preference_df,
        filter_stats=filter_stats,
        sft_dataset=sft_dataset,
        reward_train=reward_train,
        reward_val=reward_val,
        ppo_dataset=ppo_dataset,
        source_files=source_files,
    )

    return sft_dataset, reward_train, reward_val, ppo_dataset, metadata


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
        return pd.concat(frames, ignore_index=True), [str(csv_file) for csv_file in csv_files]

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
    filtered_df = preference_df.loc[quality_mask].copy()

    stats = {
        "ties_removed": ties_removed,
        "quality_filtered": int((~quality_mask).sum()),
        "too_fast": int(too_fast.sum()),
        "too_slow": int(too_slow.sum()),
    }
    return filtered_df, stats


def build_sft_dataset(preference_df: pd.DataFrame) -> Dataset:
    """Use chosen responses as supervised targets."""
    records = preference_df.loc[:, ["prompt", "chosen", "prompt_category"]].rename(
        columns={"chosen": "completion", "prompt_category": "category"}
    )
    return Dataset.from_list(records.to_dict("records"))


def build_reward_datasets(
    preference_df: pd.DataFrame,
    reward_val_fraction: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Create chosen/rejected pairs for reward-model style training."""
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
    ).to_dict("records")

    if len(reward_records) < 2 or reward_val_fraction <= 0:
        return Dataset.from_list(reward_records), Dataset.from_list([])

    stratify_labels = get_reward_stratify_labels(reward_records, reward_val_fraction)
    train_records, val_records = train_test_split(
        reward_records,
        test_size=reward_val_fraction,
        random_state=seed,
        stratify=stratify_labels,
    )
    return Dataset.from_list(train_records), Dataset.from_list(val_records)


def get_reward_stratify_labels(
    reward_records: list[dict[str, Any]],
    reward_val_fraction: float,
) -> list[str] | None:
    """
    Stratify when the split is large enough to represent each category.
    Fall back to a random split for tiny datasets or sparse categories.
    """
    categories = pd.Series(record["category"] for record in reward_records)
    category_counts = categories.value_counts()
    if category_counts.empty or int(category_counts.min()) < 2:
        return None

    val_size = math.ceil(len(reward_records) * reward_val_fraction)
    train_size = len(reward_records) - val_size
    num_categories = len(category_counts)
    if val_size < num_categories or train_size < num_categories:
        return None

    return categories.tolist()


def build_ppo_dataset(preference_df: pd.DataFrame) -> Dataset:
    """Deduplicate prompts for later on-policy sampling."""
    unique_prompts = preference_df.drop_duplicates(subset="prompt_id")
    records = unique_prompts.loc[:, ["prompt", "prompt_category"]].rename(
        columns={"prompt_category": "category"}
    )
    return Dataset.from_list(records.to_dict("records"))


def build_metadata(
    full_df: pd.DataFrame,
    preference_df: pd.DataFrame,
    filter_stats: dict[str, int],
    sft_dataset: Dataset,
    reward_train: Dataset,
    reward_val: Dataset,
    ppo_dataset: Dataset,
    source_files: list[str],
) -> dict[str, Any]:
    """Collect a few sanity-check numbers for debugging and reporting."""
    return {
        "source_files": source_files,
        "num_source_files": len(source_files),
        "total_rows": len(full_df),
        **filter_stats,
        "usable_pairs": len(preference_df),
        "sft_examples": len(sft_dataset),
        "reward_train": len(reward_train),
        "reward_val": len(reward_val),
        "ppo_prompts": len(ppo_dataset),
        "categories": preference_df["prompt_category"].value_counts().to_dict(),
        "avg_chosen_tokens": preference_df["chosen_output_tokens"].mean(),
        "avg_rejected_tokens": preference_df["rejected_output_tokens"].mean(),
        "median_reading_time_s": preference_df["reading_time_s"].median(),
    }


if __name__ == "__main__":
    import json

    sft_ds, r_train, r_val, ppo_ds, meta = load_preference_data(
        "data"
    )

    print("=== Dataset Sizes ===")
    print(f"  SFT:           {len(sft_ds)} examples")
    print(f"  Reward train:  {len(r_train)} pairs")
    print(f"  Reward val:    {len(r_val)} pairs")
    print(f"  PPO prompts:   {len(ppo_ds)} prompts")

    print("\n=== Metadata ===")
    print(json.dumps(meta, indent=2))

    print("\n=== SFT sample ===")
    sample = sft_ds[0]
    print(f"  Prompt:     {sample['prompt'][:80]}...")
    print(f"  Completion: {sample['completion'][:80]}...")

    print("\n=== Reward sample ===")
    sample = r_train[0]
    print(f"  Prompt:     {sample['prompt'][:80]}...")
    print(f"  Chosen:     {sample['chosen'][:80]}...")
    print(f"  Rejected:   {sample['rejected'][:80]}...")

    print("\n=== PPO sample ===")
    sample = ppo_ds[0]
    print(f"  Prompt:     {sample['prompt'][:80]}...")
