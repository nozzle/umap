#!/usr/bin/env python3
"""
Debug script to output all intermediate UMAP values for exact comparison.
"""

import csv
import sys
import numpy as np
from scipy.sparse import coo_matrix
import umap
from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample, find_ab_params
from sklearn.neighbors import NearestNeighbors


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
        print("Usage: debug_umap_detailed.py <input.csv>", file=sys.stderr)
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

    # Step 1: k-NN (brute force for small datasets)
    print("\n=== Step 1: k-NN Graph ===")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nn.fit(data)
    knn_distances, knn_indices = nn.kneighbors(data)

    print(f"knn_indices:\n{knn_indices}")
    print(f"knn_distances:\n{knn_distances}")

    # Step 2: Fuzzy simplicial set
    print("\n=== Step 2: Fuzzy Simplicial Set ===")
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

    print(f"Sigmas: {sigmas}")
    print(f"Rhos: {rhos}")

    # Convert to COO for inspection
    coo = coo_matrix(graph)
    print(f"\nGraph edges (first 30):")
    for i in range(min(30, len(coo.row))):
        print(f"  ({coo.row[i]}, {coo.col[i]}): {coo.data[i]:.10f}")

    # Step 3: a, b parameters
    print("\n=== Step 3: Curve Parameters ===")
    a, b = find_ab_params(spread, min_dist)
    print(f"a = {a:.15f}")
    print(f"b = {b:.15f}")

    # Step 4: Epochs per sample
    print("\n=== Step 4: Epochs Per Sample ===")
    epochs_per_sample = make_epochs_per_sample(coo.data, n_epochs)
    print(f"epochs_per_sample (first 30): {epochs_per_sample[:30]}")

    # Step 5: Random initial embedding
    print("\n=== Step 5: Random Initial Embedding ===")
    random_state = np.random.RandomState(seed)
    n_samples = data.shape[0]
    init_embedding = random_state.uniform(
        low=-10.0, high=10.0, size=(n_samples, 2)
    ).astype(np.float32)
    print(f"Initial embedding (from seed {seed}):")
    print(init_embedding)

    # Step 6: Full UMAP with random init
    print("\n=== Step 6: Final Embedding ===")
    reducer = umap.UMAP(
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
    embedding = reducer.fit_transform(data)
    print(f"Final embedding:\n{embedding}")

    # Also output the random state sequence for debugging
    print("\n=== Random State Sequence ===")
    rs = np.random.RandomState(seed)
    print("First 20 uniform values from RandomState(42):")
    for i in range(20):
        print(f"  {i}: {rs.uniform(-10.0, 10.0):.15f}")


if __name__ == "__main__":
    main()
