#!/usr/bin/env python3
"""
Debug script to output intermediate UMAP values for comparison with Go implementation.
"""

import csv
import json
import sys
import numpy as np
from scipy.sparse import coo_matrix
import umap
from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample
from pynndescent import NNDescent


def load_csv(filename: str) -> np.ndarray:
    """Load data from a CSV file (no header, numeric values only)."""
    data = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data, dtype=np.float32)


def main():
    if len(sys.argv) < 2:
        print("Usage: debug_umap.py <input.csv>", file=sys.stderr)
        sys.exit(1)

    data = load_csv(sys.argv[1])
    print(f"Data shape: {data.shape}")

    # Parameters matching our test
    n_neighbors = 5
    metric = "euclidean"
    min_dist = 0.1
    spread = 1.0
    n_epochs = 200
    seed = 42

    # Set random state
    random_state = np.random.RandomState(seed)

    # Step 1: Build k-NN graph using pynndescent (what UMAP uses internally)
    print("\n=== Step 1: k-NN Graph ===")

    # For small datasets, UMAP uses brute force
    n_samples = data.shape[0]
    if n_samples < 4096:
        print("Using brute force k-NN (small dataset)")
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(data)
        knn_distances, knn_indices = nn.kneighbors(data)
    else:
        print("Using NNDescent")
        nnd = NNDescent(
            data,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            n_jobs=1,
        )
        knn_indices, knn_distances = nnd.neighbor_graph

    print(f"knn_indices shape: {knn_indices.shape}")
    print(f"knn_distances shape: {knn_distances.shape}")
    print(f"knn_indices:\n{knn_indices}")
    print(f"knn_distances:\n{knn_distances}")

    # Step 2: Compute fuzzy simplicial set
    print("\n=== Step 2: Fuzzy Simplicial Set ===")

    # Need to reset random state for reproducibility
    random_state = np.random.RandomState(seed)

    graph, sigmas, rhos = fuzzy_simplicial_set(
        X=data,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        local_connectivity=1.0,
        set_op_mix_ratio=1.0,
    )

    print(f"Graph shape: {graph.shape}")
    print(f"Graph nnz: {graph.nnz}")
    print(f"Sigmas: {sigmas}")
    print(f"Rhos: {rhos}")

    # Convert to COO for easier inspection
    coo = coo_matrix(graph)
    print(f"\nGraph edges (row, col, data):")
    for i in range(min(20, len(coo.row))):
        print(f"  ({coo.row[i]}, {coo.col[i]}): {coo.data[i]:.6f}")
    if len(coo.row) > 20:
        print(f"  ... ({len(coo.row)} total edges)")

    # Step 3: Compute a, b parameters
    print("\n=== Step 3: Curve Parameters (a, b) ===")
    from umap.umap_ import find_ab_params

    a, b = find_ab_params(spread, min_dist)
    print(f"a = {a}")
    print(f"b = {b}")

    # Step 4: Compute epochs per sample
    print("\n=== Step 4: Epochs Per Sample ===")
    epochs_per_sample = make_epochs_per_sample(graph.tocoo().data, n_epochs)
    print(f"epochs_per_sample (first 20): {epochs_per_sample[:20]}")

    # Step 5: Initial embedding
    print("\n=== Step 5: Initial Embedding ===")

    # Run full UMAP to get the embedding
    random_state = np.random.RandomState(seed)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric=metric,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        random_state=seed,
        init="spectral",  # default
        verbose=False,
    )
    embedding = reducer.fit_transform(data)

    print(f"Final embedding shape: {embedding.shape}")
    print(f"Final embedding:\n{embedding}")

    # Also try with random init for comparison
    print("\n=== With Random Init ===")
    random_state = np.random.RandomState(seed)
    reducer_random = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        metric=metric,
        min_dist=min_dist,
        spread=spread,
        n_epochs=n_epochs,
        random_state=seed,
        init="random",
        verbose=False,
    )
    embedding_random = reducer_random.fit_transform(data)
    print(f"Random init embedding:\n{embedding_random}")


if __name__ == "__main__":
    main()
