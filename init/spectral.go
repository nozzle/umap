// Package init provides initialization methods for UMAP embeddings.
// This includes spectral embedding and random initialization.
package init

import (
	"math"
	"sort"

	"github.com/nozzle/umap/graph"
	"github.com/nozzle/umap/internal/rand"
	"gonum.org/v1/gonum/mat"
)

// InitMethod specifies the initialization method.
type InitMethod int

const (
	// Spectral uses spectral embedding from the graph Laplacian
	Spectral InitMethod = iota
	// Random uses random initialization
	Random
	// PCA uses PCA of the input data (not yet implemented)
	PCA
)

// SpectralEmbedding computes a spectral embedding of the graph.
// This provides a good initialization for the UMAP optimization.
func SpectralEmbedding(g *graph.CSRMatrix, dim int, seed int64) [][]float32 {
	n := g.NRows
	if n == 0 {
		return nil
	}

	// Build the normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
	// where A is the adjacency matrix and D is the degree matrix

	// Compute degrees
	degrees := make([]float64, n)
	for i := 0; i < n; i++ {
		start := g.Indptr[i]
		end := g.Indptr[i+1]
		for j := start; j < end; j++ {
			degrees[i] += float64(g.Data[j])
		}
	}

	// Compute D^(-1/2)
	dInvSqrt := make([]float64, n)
	for i := 0; i < n; i++ {
		if degrees[i] > 0 {
			dInvSqrt[i] = 1.0 / math.Sqrt(degrees[i])
		}
	}

	// Build normalized Laplacian as dense matrix (for small n)
	// For large n, we should use sparse eigensolvers
	if n > 5000 {
		// Fall back to random initialization for large graphs
		return RandomEmbedding(n, dim, seed)
	}

	// Build L = I - D^(-1/2) * A * D^(-1/2) as a symmetric matrix
	// First build as dense, then convert to symmetric
	lData := make([]float64, n*n)
	for i := 0; i < n; i++ {
		lData[i*n+i] = 1.0 // Identity diagonal

		start := g.Indptr[i]
		end := g.Indptr[i+1]
		for idx := start; idx < end; idx++ {
			j := int(g.Indices[idx])
			val := float64(g.Data[idx])
			normalized := -val * dInvSqrt[i] * dInvSqrt[j]
			lData[i*n+j] = normalized
		}
	}

	// Symmetrize by averaging L[i,j] and L[j,i]
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			avg := (lData[i*n+j] + lData[j*n+i]) / 2
			lData[i*n+j] = avg
			lData[j*n+i] = avg
		}
	}

	L := mat.NewSymDense(n, lData)

	// Compute eigendecomposition
	var eig mat.EigenSym
	ok := eig.Factorize(L, true)
	if !ok {
		// Eigendecomposition failed, fall back to random
		return RandomEmbedding(n, dim, seed)
	}

	// Get eigenvalues and eigenvectors
	eigenvalues := eig.Values(nil)
	var eigenvectors mat.Dense
	eig.VectorsTo(&eigenvectors)

	// Sort by eigenvalue (ascending) and take the smallest non-zero ones
	type eigenPair struct {
		value  float64
		vector []float64
	}
	pairs := make([]eigenPair, n)
	for i := 0; i < n; i++ {
		vec := make([]float64, n)
		for j := 0; j < n; j++ {
			vec[j] = eigenvectors.At(j, i)
		}
		pairs[i] = eigenPair{eigenvalues[i], vec}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value < pairs[j].value
	})

	// Skip the first eigenvector (constant) and take the next 'dim' ones
	result := make([][]float32, n)
	for i := 0; i < n; i++ {
		result[i] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			if d+1 < len(pairs) {
				result[i][d] = float32(pairs[d+1].vector[i])
			}
		}
	}

	// Scale and center the embedding
	scaleAndCenter(result)

	return result
}

// RandomEmbedding generates a random embedding.
func RandomEmbedding(n, dim int, seed int64) [][]float32 {
	rng := rand.New(seed)

	result := make([][]float32, n)
	for i := 0; i < n; i++ {
		result[i] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			// Uniform in [-10, 10] as in the original UMAP
			result[i][d] = (rand.Float32(&rng) - 0.5) * 20
		}
	}

	return result
}

// InitializeEmbedding creates an initial embedding based on the method.
func InitializeEmbedding(g *graph.CSRMatrix, n, dim int, method InitMethod, seed int64) [][]float32 {
	switch method {
	case Spectral:
		embedding := SpectralEmbedding(g, dim, seed)
		if embedding != nil {
			return embedding
		}
		// Fall back to random if spectral fails
		return RandomEmbedding(n, dim, seed)
	case Random:
		return RandomEmbedding(n, dim, seed)
	default:
		return RandomEmbedding(n, dim, seed)
	}
}

// scaleAndCenter scales the embedding to have reasonable values.
func scaleAndCenter(embedding [][]float32) {
	if len(embedding) == 0 {
		return
	}

	n := len(embedding)
	dim := len(embedding[0])

	// Compute mean for each dimension
	means := make([]float64, dim)
	for i := 0; i < n; i++ {
		for d := 0; d < dim; d++ {
			means[d] += float64(embedding[i][d])
		}
	}
	for d := 0; d < dim; d++ {
		means[d] /= float64(n)
	}

	// Center
	for i := 0; i < n; i++ {
		for d := 0; d < dim; d++ {
			embedding[i][d] -= float32(means[d])
		}
	}

	// Compute max absolute value
	maxAbs := float32(0)
	for i := 0; i < n; i++ {
		for d := 0; d < dim; d++ {
			val := embedding[i][d]
			if val < 0 {
				val = -val
			}
			if val > maxAbs {
				maxAbs = val
			}
		}
	}

	// Scale to [-10, 10]
	if maxAbs > 0 {
		scale := 10.0 / maxAbs
		for i := 0; i < n; i++ {
			for d := 0; d < dim; d++ {
				embedding[i][d] *= scale
			}
		}
	}
}

// NormalizeCoordsTo01 normalizes embedding coordinates to [0, 1] range.
func NormalizeCoordsTo01(embedding [][]float32) {
	if len(embedding) == 0 {
		return
	}

	n := len(embedding)
	dim := len(embedding[0])

	// Find min/max for each dimension
	mins := make([]float32, dim)
	maxs := make([]float32, dim)
	for d := 0; d < dim; d++ {
		mins[d] = embedding[0][d]
		maxs[d] = embedding[0][d]
	}

	for i := 1; i < n; i++ {
		for d := 0; d < dim; d++ {
			if embedding[i][d] < mins[d] {
				mins[d] = embedding[i][d]
			}
			if embedding[i][d] > maxs[d] {
				maxs[d] = embedding[i][d]
			}
		}
	}

	// Normalize
	for d := 0; d < dim; d++ {
		spread := maxs[d] - mins[d]
		if spread > 0 {
			for i := 0; i < n; i++ {
				embedding[i][d] = (embedding[i][d] - mins[d]) / spread
			}
		}
	}
}
