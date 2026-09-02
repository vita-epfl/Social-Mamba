import argparse
import random

import numpy as np
import torch

from dataset import create_dataset


def generate_data_cache(dataset_name, split="all", chunk_size=4, obs_len=10, max_agents=8):
    splits = ["train", "val", "test"] if split == "all" else [split]
    for current_split in splits:
        print(f"Generate {current_split} data cache for {dataset_name}...")
        create_dataset(
            dataset_name,
            split=current_split,
            chunk_size=chunk_size,
            obs_len=obs_len,
            max_agents=max_agents,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--UniHuMotion_dataset",
        type=str,
        default="UHM_NBA",
        help="Name of the dataset directory under data/UniHuMotion, e.g. UHM_NBA",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--obs-len", type=int, default=10)
    parser.add_argument("--max-agents", type=int, default=8)
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    generate_data_cache(
        dataset_name=args.UniHuMotion_dataset,
        split=args.split,
        chunk_size=args.chunk_size,
        obs_len=args.obs_len,
        max_agents=args.max_agents,
    )


if __name__ == "__main__":
    main()
