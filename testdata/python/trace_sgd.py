#!/usr/bin/env python3
"""Trace SGD optimization for debugging."""

import csv
import numpy as np
from scipy.sparse import coo_matrix
from umap.umap_ import fuzzy_simplicial_set, make_epochs_per_sample, find_ab_params
from umap.utils import tau_rand_int
from sklearn.neighbors import NearestNeighbors


def load_csv(filename):
    data = []
    with open(filename) as f:
        for row in csv.reader(f):
            data.append([float(x) for x in row])
    return np.array(data, dtype=np.float32)


def main():
    data = load_csv('../test_data.csv')
    n_samples = data.shape[0]
    n_neighbors = 5
    n_epochs = 200
    seed = 42
    negative_sample_rate = 5.0
    
    # Build k-NN
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(data)
    knn_distances, knn_indices = nn.kneighbors(data)
    
    # Build fuzzy simplicial set
    random_state = np.random.RandomState(seed)
    graph, _, _ = fuzzy_simplicial_set(
        X=data,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric='euclidean',
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        local_connectivity=1.0,
        set_op_mix_ratio=1.0,
    )
    
    # Get edges
    coo = coo_matrix(graph)
    head = coo.row.astype(np.int32)
    tail = coo.col.astype(np.int32)
    weights = coo.data.astype(np.float32)
    
    print(f"Number of edges: {len(head)}")
    print(f"First 10 edges: {list(zip(head[:10], tail[:10], weights[:10]))}")
    
    # Curve parameters
    a, b = find_ab_params(1.0, 0.1)
    print(f"\na = {a}, b = {b}")
    
    # Initialize embedding
    random_state = np.random.RandomState(seed)
    embedding = random_state.uniform(-10.0, 10.0, (n_samples, 2)).astype(np.float32)
    print(f"\nInitial embedding (first 3):")
    print(embedding[:3])
    
    # RNG state
    rng_state = random_state.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max + 1, 3).astype(np.int64)
    print(f"\nrng_state: {rng_state}")
    
    # rng_state_per_sample
    rng_state_per_sample = np.full(
        (n_samples, len(rng_state)), rng_state, dtype=np.int64
    ) + embedding[:, 0].astype(np.float64).view(np.int64).reshape(-1, 1)
    
    print(f"\nrng_state_per_sample (first 3):")
    print(rng_state_per_sample[:3])
    
    # epochs_per_sample
    epochs_per_sample = make_epochs_per_sample(weights, n_epochs)
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_sample = epochs_per_sample.copy()
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    
    print(f"\nepochs_per_sample (first 10): {epochs_per_sample[:10]}")
    print(f"epochs_per_negative_sample (first 10): {epochs_per_negative_sample[:10]}")
    
    # Trace first few iterations
    alpha = 1.0
    print(f"\n=== Tracing first 3 epochs ===")
    
    for epoch in range(3):
        print(f"\n--- Epoch {epoch} ---")
        edge_count = 0
        for i in range(len(head)):
            if epoch_of_next_sample[i] <= epoch:
                j = head[i]
                k = tail[i]
                
                if edge_count < 3:  # Only print first 3 edges per epoch
                    print(f"  Edge {i}: ({j}, {k})")
                    print(f"    epoch_of_next_sample[{i}] = {epoch_of_next_sample[i]}")
                    
                    # Calculate negative samples
                    n_neg = int((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
                    print(f"    n_neg_samples = {n_neg}")
                    
                    if n_neg > 0:
                        # Show first random sample
                        rng = rng_state_per_sample[j].copy()
                        k_neg = tau_rand_int(rng) % n_samples
                        print(f"    First negative sample: {k_neg}")
                
                epoch_of_next_sample[i] += epochs_per_sample[i]
                n_neg = int((epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i])
                epoch_of_next_negative_sample[i] += n_neg * epochs_per_negative_sample[i]
                
                edge_count += 1
        
        print(f"  Total edges processed: {edge_count}")
        alpha = 1.0 * (1.0 - epoch / n_epochs)


if __name__ == '__main__':
    main()
