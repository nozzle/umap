#!/usr/bin/env python3
"""
Run Python UMAP on input CSV and output embeddings to a CSV file.

This script is used by Go tests to compare the output of the native Python
umap-learn library against the Go implementation.

Usage:
    python run_umap.py --input input.csv --output output.csv [options]

Options match the Go implementation's CLI flags for consistency.
"""

import argparse
import csv
import sys

import numpy as np
import umap


def load_csv(filename: str) -> np.ndarray:
    """Load data from a CSV file (no header, numeric values only)."""
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data, dtype=np.float32)


def save_csv(filename: str, embedding: np.ndarray) -> None:
    """Save embedding to a CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in embedding:
            writer.writerow([f"{x:.6f}" for x in row])


def main():
    parser = argparse.ArgumentParser(
        description="Run UMAP on input CSV and output embeddings"
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--neighbors", type=int, default=15, help="Number of neighbors for k-NN"
    )
    parser.add_argument(
        "--components", type=int, default=2, help="Number of output dimensions"
    )
    parser.add_argument("--metric", default="euclidean", help="Distance metric")
    parser.add_argument(
        "--min-dist", type=float, default=0.1, help="Minimum distance between points"
    )
    parser.add_argument(
        "--spread", type=float, default=1.0, help="Spread of embedded points"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load data
    try:
        data = load_csv(args.input)
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Loaded {data.shape[0]} samples with {data.shape[1]} features")

    # Configure and run UMAP
    reducer = umap.UMAP(
        n_neighbors=args.neighbors,
        n_components=args.components,
        metric=args.metric,
        min_dist=args.min_dist,
        spread=args.spread,
        n_epochs=args.epochs,
        random_state=args.seed,
        verbose=args.verbose,
    )

    embedding = reducer.fit_transform(data)

    # Save output
    try:
        save_csv(args.output, embedding)
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Saved embedding to {args.output}")


if __name__ == "__main__":
    main()
