#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from safetensors.torch import load_file, save_file

def main():
    parser = argparse.ArgumentParser(description="Merge safetensor dictionaries.")
    parser.add_argument("--prefix", help="Prefix of files to merge")
    parser.add_argument("--files", nargs='+', help="Exact list of files to merge")
    parser.add_argument("--output", required=True, help="Final output file")
    parser.add_argument("--cleanup", action="store_true", help="Delete merged files")
    args = parser.parse_args()

    # Determine files to merge based on arguments
    if args.files:
        files = args.files
    elif args.prefix:
        search_pattern = f"{args.prefix}*.safetensors"
        files = [f for f in glob.glob(search_pattern) if os.path.basename(f) != os.path.basename(args.output)]
    else:
        print("Must provide --prefix or --files")
        sys.exit(1)

    if not files:
        print("No partial files found.")
        sys.exit(0)

    print(f"Found {len(files)} files to merge.")
    
    # Safely merge dictionaries (your original, correct logic!)
    combined = {}
    for f in files:
        try:
            part = load_file(f)
            combined.update(part)
        except Exception as e:
            print(f"FAILED to load {f}: {e}")
            sys.exit(1)

    print(f"Merge complete. Total sequences: {len(combined)}")
    try:
        save_file(combined, args.output)
        print(f"Saved to {args.output}")
    except Exception as e:
        print(f"FAILED to save: {e}")
        sys.exit(1)

    # Cleanup only what was merged
    if args.cleanup:
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

if __name__ == "__main__":
    main()
