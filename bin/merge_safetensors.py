#!/usr/bin/env python3
import torch
from safetensors.torch import load_file, save_file
import glob
import os
import argparse
import sys

def merge_safetensors(prefix, output_file, delete_partials=False):
    # Find all partial files (e.g., prefix.100.safetensors, prefix.200.safetensors)
    # We exclude the output file name itself to avoid accidental recursive loading
    search_pattern = f"{prefix}.*.safetensors"
    files = sorted([
        f for f in glob.glob(search_pattern) 
        if os.path.basename(f) != os.path.basename(output_file)
    ])

    if not files:
        print(f"Error: No partial files found matching pattern: {search_pattern}")
        sys.exit(1)

    print(f"Found {len(files)} partial files. Merging...")

    combined = {}
    for f in files:
        print(f"  Loading {f}...")
        try:
            # load_file returns a dict-like object mapping keys to tensors
            part = load_file(f)
            combined.update(part)
        except Exception as e:
            print(f"  FAILED to load {f}: {e}")
            sys.exit(1)

    print(f"Merge complete. Total sequences/tensors: {len(combined)}")
    
    try:
        save_file(combined, output_file)
        print(f"Successfully saved combined file to: {output_file}")
    except Exception as e:
        print(f"  FAILED to save {output_file}: {e}")
        sys.exit(1)

    if delete_partials:
        print("Cleaning up partial files...")
        for f in files:
            os.remove(f)

def main():
    parser = argparse.ArgumentParser(description="Merge partial safetensor files into a single file.")
    parser.add_argument("--prefix", required=True, help="The prefix of the partial files (e.g., 'results_part')")
    parser.add_argument("--output", required=True, help="The name of the final combined file (e.g., 'final.safetensors')")
    parser.add_argument("--cleanup", action="store_true", help="Delete partial files after a successful merge")

    args = parser.parse_args()

    merge_safetensors(args.prefix, args.output, args.cleanup)

if __name__ == "__main__":
    main()

