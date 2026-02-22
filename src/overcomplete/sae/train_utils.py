"""
Training utilities for Sparse Auto Encoder (SAE).

This module contains utility functions for feature extraction, saving/loading,
and logging for SAE training.
"""

import torch
import json
from pathlib import Path


def save_features(features, labels, save_path):
    """Save extracted features and labels."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "features": features,
            "labels": labels,
            "feature_dim": features.shape[1],
            "num_samples": features.shape[0],
        },
        save_path,
    )
    print(f"Features saved to {save_path}")


def load_features(load_path):
    """Load extracted features and labels."""
    load_path = Path(load_path)
    if not load_path.exists():
        return None, None

    data = torch.load(load_path, map_location="cpu")
    features = data["features"]
    labels = data["labels"]

    print(f"Features loaded from {load_path}")
    print(f"Feature shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")

    return features, labels


def save_logs_to_file(logs, output_path):
    """Save training logs to a text file."""
    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Training Logs\n")
        f.write("=" * 80 + "\n\n")

        # Get the number of epochs
        if not logs:
            f.write("No training logs available.\n")
            return

        num_epochs = len(list(logs.values())[0])

        # Write header
        header = "Epoch"
        for key in logs.keys():
            header += f"\t{key}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")

        # Write data for each epoch
        for epoch in range(num_epochs):
            row = f"{epoch + 1}"
            for key, values in logs.items():
                if epoch < len(values):
                    row += f"\t{values[epoch]:.6f}"
                else:
                    row += f"\t-"
            f.write(row + "\n")

    print(f"Logs saved to {output_path}")


def save_args_to_file(args, output_path):
    """Save command line arguments to a JSON file."""
    output_path = Path(output_path)

    # Convert args to dictionary and handle non-serializable types
    args_dict = vars(args)

    # Convert Path objects to strings for JSON serialization
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)

    print(f"Arguments saved to {output_path}")
