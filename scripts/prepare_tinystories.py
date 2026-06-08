"""
Downloads and tokenizes a subset of TinyStories to data/tinystories.bin.
This allows for a 'Real' language modeling experiment in Yubo.
"""

import os

import numpy as np
import requests
from tqdm import tqdm


def download_and_tokenize():
    import tiktoken

    # 1. Setup paths
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    bin_path = os.path.join(data_dir, "tinystories.bin")

    if os.path.exists(bin_path):
        print(f"Data already exists at {bin_path}")
        return

    # 2. Download a small shard of TinyStories (Validation set)
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    print(f"Downloading TinyStories validation set from {url}...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    text = ""
    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                text += chunk.decode("utf-8", errors="ignore")
                pbar.update(len(chunk))

    # 3. Tokenize using tiktoken (GPT-2 encoding used by nanochat)
    print("Tokenizing text...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    tokens_np = np.array(tokens, dtype=np.uint16)

    # 4. Save to .bin
    print(f"Saving {len(tokens_np):,} tokens to {bin_path}...")
    tokens_np.tofile(bin_path)
    print("Done!")


if __name__ == "__main__":
    download_and_tokenize()
