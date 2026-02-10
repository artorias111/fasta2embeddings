# /usr/bin/env python
import torch
from safetensors.torch import load_file

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--safetensor', type = str, required = True)
args = parser.parse_args()

file_path = args.safetensor
embeddings = load_file(file_path)

for key, tensor in embeddings.items():
    print(f"Key: {key}")
    print(f"Shape: {tensor.shape}")