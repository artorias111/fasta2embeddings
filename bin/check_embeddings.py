#!/usr/bin/env python
import argparse
from safetensors import safe_open

parser = argparse.ArgumentParser()
parser.add_argument('--safetensor', type=str, nargs='+', required=True)
args = parser.parse_args()

for file_path in args.safetensor:
    print(f"\n--- Checking {file_path} ---")
    
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"Key: {key} | Shape: {tensor.shape}")
