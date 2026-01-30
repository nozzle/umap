// Package graph provides fuzzy simplicial set construction for UMAP.
// This implements the core topological representation of the high-dimensional data.
package graph

import (
	"math"
	"sort"

	"github.com/nozzle/umap/internal/parallel"
)

// SparseMatrix represents a sparse matrix in COO format.
type SparseMatrix struct {
	Rows  []int32   // Row indices
	Cols  []int32   // Column indices
	Data  []float32 // Values
	NRows int       // Number of rows
	NCols int       // Number of columns
	NNZ   int       // Number of non-zero elements
}

// CSRMatrix represents a sparse matrix in CSR format.
type CSRMatrix struct {
	Indptr  []int32   // Row pointers
	Indices []int32   // Column indices
	Data    []float32 // Values
	NRows   int       // Number of rows
	NCols   int       // Number of columns
	NNZ     int       // Number of non-zero elements
}

// FuzzySimplicialSetConfig configures fuzzy simplicial set construction.
type FuzzySimplicialSetConfig struct {
	// LocalConnectivity controls how local the connectivity estimate is
	LocalConnectivity float64
	// SetOpMixRatio controls the blend between fuzzy set union and intersection
	// 0.0 = pure intersection, 1.0 = pure union
	SetOpMixRatio float64
	// ApplySetOperations whether to apply fuzzy set operations
	ApplySetOperations bool
	// NumWorkers for parallel processing (0 = auto)
	NumWorkers int
}

// DefaultFuzzySimplicialSetConfig returns default configuration.
func DefaultFuzzySimplicialSetConfig() FuzzySimplicialSetConfig {
	return FuzzySimplicialSetConfig{
		LocalConnectivity:  1.0,
		SetOpMixRatio:      1.0,
		ApplySetOperations: true,
		NumWorkers:         0,
	}
}

// FuzzySimplicialSet constructs a fuzzy simplicial set from k-NN data.
// This is the core of UMAP's topological representation.
//
// IMPORTANT: knnIndices and knnDistances are expected to include self as the first
// neighbor (matching sklearn's NearestNeighbors output). This function will skip
// self when computing sigma/rho (matching Python's knn_dists[:, 1:] behavior)
// and also skip self-edges when building the graph.
func FuzzySimplicialSet(
	knnIndices [][]int32,
	knnDistances [][]float32,
	config FuzzySimplicialSetConfig,
) *CSRMatrix {
	n := len(knnIndices)
	if n == 0 {
		return &CSRMatrix{}
	}
	k := len(knnIndices[0])

	numWorkers := config.NumWorkers
	if numWorkers <= 0 {
		numWorkers = parallel.NumWorkers()
	}

	// Compute smooth k-NN distances (sigma and rho for each point)
	// Python UMAP uses knn_dists[:, 1:] to skip self-distance
	sigmas := make([]float32, n)
	rhos := make([]float32, n)

	parallel.ParallelFor(0, n, numWorkers, func(i int) {
		// Skip self-distance (first element) - matches Python's knn_dists[:, 1:]
		distsNoSelf := knnDistances[i][1:]
		sigmas[i], rhos[i] = smoothKNNDist(
			distsNoSelf,
			float64(k-1), // k-1 because we excluded self
			config.LocalConnectivity,
		)
	})

	// Compute membership strengths
	// Skip self-edges (where neighbor == point itself)
	rows := make([]int32, 0, n*k)
	cols := make([]int32, 0, n*k)
	data := make([]float32, 0, n*k)

	for i := range n {
		for j := range k {
			neighbor := knnIndices[i][j]
			// Skip self-edges and invalid indices
			if neighbor < 0 || neighbor == int32(i) {
				continue
			}
			dist := knnDistances[i][j]

			var membership float32
			if dist <= rhos[i] || sigmas[i] == 0 {
				membership = 1.0
			} else {
				membership = float32(math.Exp(-float64(dist-rhos[i]) / float64(sigmas[i])))
			}

			if membership > 0 {
				rows = append(rows, int32(i))
				cols = append(cols, neighbor)
				data = append(data, membership)
			}
		}
	}

	// Convert to CSR for efficient row access
	graph := cooToCSR(rows, cols, data, n, n)

	// Apply fuzzy set operations (symmetrize the graph)
	if config.ApplySetOperations {
		graph = fuzzySetUnion(graph, config.SetOpMixRatio)
	}

	return graph
}

// smoothKNNDist computes the smooth distance normalization parameters.
// Returns sigma (bandwidth) and rho (distance to nearest neighbor).
//
// This matches Python UMAP's smooth_knn_dist function exactly:
// - distances should be the k nearest neighbor distances (excluding self)
// - rho is computed from non-zero distances based on local_connectivity
// - The sum for sigma is computed over distances[1:] (skipping first distance)
// - When binary search doesn't converge, sigma may be very large (Python returns inf)
func smoothKNNDist(distances []float32, k, localConnectivity float64) (float32, float32) {
	const (
		nIter           = 64   // Binary search iterations
		bandwidth       = 1.0  // Target bandwidth in perplexity-like units
		smoothTolerance = 1e-5 // Convergence tolerance
		minKDistScale   = 1e-3 // Minimum sigma as fraction of mean distance
	)

	// Filter non-zero distances (matching Python's non_zero_dists = ith_distances[ith_distances > 0.0])
	nonZeroDists := make([]float32, 0, len(distances))
	for _, d := range distances {
		if d > 0 {
			nonZeroDists = append(nonZeroDists, d)
		}
	}

	// Compute rho based on local connectivity (matching Python exactly)
	// Python: index = int(np.floor(local_connectivity))
	//         if index > 0: rho = non_zero_dists[index - 1]
	//         else: rho = interpolation * non_zero_dists[0]
	var rho float32
	index := int(math.Floor(localConnectivity))
	interpolation := float32(localConnectivity - float64(index))

	if len(nonZeroDists) >= int(localConnectivity) {
		if index > 0 {
			rho = nonZeroDists[index-1]
			if interpolation > smoothTolerance && index < len(nonZeroDists) {
				rho += interpolation * (nonZeroDists[index] - nonZeroDists[index-1])
			}
		} else {
			rho = interpolation * nonZeroDists[0]
		}
	} else if len(nonZeroDists) > 0 {
		// Fallback: use max non-zero distance
		rho = nonZeroDists[len(nonZeroDists)-1]
		for _, d := range nonZeroDists {
			if d > rho {
				rho = d
			}
		}
	}

	// Compute mean distance for minimum sigma calculation
	var meanDist float32
	for _, d := range distances {
		meanDist += d
	}
	if len(distances) > 0 {
		meanDist /= float32(len(distances))
	}

	// Target sum of membership strengths
	target := math.Log2(k) * bandwidth

	// Binary search for sigma (matching Python's algorithm)
	// Python uses inf for hi and mid *= 2 when hi is still inf
	lo := float64(0.0)
	hi := math.Inf(1)
	mid := 1.0

	for range nIter {
		// Compute current sum of membership strengths
		// IMPORTANT: Python's loop is "for j in range(1, distances.shape[1])"
		// which skips the first distance (j=0)
		sum := 0.0
		for i := 1; i < len(distances); i++ {
			d := float64(distances[i]) - float64(rho)
			if d > 0 {
				sum += math.Exp(-d / mid)
			} else {
				sum += 1.0
			}
		}

		if math.Abs(sum-target) < smoothTolerance {
			break
		}

		if sum > target {
			hi = mid
			mid = (lo + hi) / 2.0
		} else {
			lo = mid
			if math.IsInf(hi, 1) {
				mid *= 2
			} else {
				mid = (lo + hi) / 2.0
			}
		}
	}

	sigma := float32(mid)

	// Apply minimum sigma threshold (matching Python)
	if rho > 0 {
		minSigma := minKDistScale * meanDist
		if sigma < minSigma {
			sigma = minSigma
		}
	}

	return sigma, rho
}

// cooToCSR converts COO format to CSR format.
func cooToCSR(rows, cols []int32, data []float32, nrows, ncols int) *CSRMatrix {
	nnz := len(rows)

	// Sort by row
	type entry struct {
		row, col int32
		val      float32
	}
	entries := make([]entry, nnz)
	for i := range entries {
		entries[i] = entry{rows[i], cols[i], data[i]}
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].row != entries[j].row {
			return entries[i].row < entries[j].row
		}
		return entries[i].col < entries[j].col
	})

	// Build CSR
	indptr := make([]int32, nrows+1)
	indices := make([]int32, nnz)
	vals := make([]float32, nnz)

	for i, e := range entries {
		indices[i] = e.col
		vals[i] = e.val
		indptr[e.row+1]++
	}

	// Cumulative sum for indptr
	for i := 1; i <= nrows; i++ {
		indptr[i] += indptr[i-1]
	}

	return &CSRMatrix{
		Indptr:  indptr,
		Indices: indices,
		Data:    vals,
		NRows:   nrows,
		NCols:   ncols,
		NNZ:     nnz,
	}
}

// fuzzySetUnion computes the fuzzy set union (symmetrization) of the graph.
// result = mix * union + (1 - mix) * intersection
// where union = A + A^T - A * A^T
// and intersection = A * A^T
func fuzzySetUnion(graph *CSRMatrix, mixRatio float64) *CSRMatrix {
	n := graph.NRows

	// Convert to map for easier manipulation
	type edgeKey struct {
		i, j int32
	}
	edges := make(map[edgeKey]float32)

	// Add original edges
	for i := range n {
		start := graph.Indptr[i]
		end := graph.Indptr[i+1]
		for idx := start; idx < end; idx++ {
			j := graph.Indices[idx]
			val := graph.Data[idx]
			edges[edgeKey{int32(i), j}] = val
		}
	}

	// Symmetrize: for each edge (i,j), compute symmetric value
	symmetric := make(map[edgeKey]float32)
	for key, val := range edges {
		i, j := key.i, key.j
		reverse := edgeKey{j, i}
		revVal, hasReverse := edges[reverse]

		if i <= j { // Only process each pair once
			if hasReverse {
				// Both directions exist: apply set operation
				// union = a + b - a*b
				// intersection = a*b
				// result = mix * union + (1-mix) * intersection
				union := float32(float64(val) + float64(revVal) - float64(val)*float64(revVal))
				intersection := val * revVal
				result := float32(mixRatio)*union + float32(1-mixRatio)*intersection

				symmetric[edgeKey{i, j}] = result
				symmetric[edgeKey{j, i}] = result
			} else {
				// Only forward direction: use as-is for union
				result := float32(mixRatio) * val
				symmetric[edgeKey{i, j}] = result
				symmetric[edgeKey{j, i}] = result
			}
		}
	}

	// Convert back to CSR
	rows := make([]int32, 0, len(symmetric))
	cols := make([]int32, 0, len(symmetric))
	data := make([]float32, 0, len(symmetric))

	for key, val := range symmetric {
		if val > 0 {
			rows = append(rows, key.i)
			cols = append(cols, key.j)
			data = append(data, val)
		}
	}

	return cooToCSR(rows, cols, data, n, n)
}

// ToEpochsPerSample converts edge weights to epochs per sample for optimization.
// This determines how frequently each edge should be sampled during SGD.
// The formula matches Python UMAP: epochs_per_sample = max_weight / weight
// This means the highest-weighted edge is sampled every epoch (epochs_per_sample=1),
// and lower-weighted edges are sampled less frequently.
func ToEpochsPerSample(graph *CSRMatrix, nEpochs int) []float32 {
	if graph.NNZ == 0 {
		return nil
	}

	// Find maximum weight
	maxWeight := float32(0)
	for _, w := range graph.Data {
		if w > maxWeight {
			maxWeight = w
		}
	}

	if maxWeight == 0 {
		return make([]float32, graph.NNZ)
	}

	// Compute epochs per sample (matching Python UMAP)
	// epochs_per_sample = max_weight / weight
	result := make([]float32, graph.NNZ)
	for i, w := range graph.Data {
		if w > 0 {
			result[i] = maxWeight / w
		} else {
			result[i] = -1 // Skip this edge
		}
	}

	return result
}

// GetEdges returns the edges of the graph as (row, col, weight) triplets.
func (g *CSRMatrix) GetEdges() ([]int32, []int32, []float32) {
	rows := make([]int32, g.NNZ)
	cols := make([]int32, g.NNZ)
	data := make([]float32, g.NNZ)

	idx := 0
	for i := 0; i < g.NRows; i++ {
		start := g.Indptr[i]
		end := g.Indptr[i+1]
		for j := start; j < end; j++ {
			rows[idx] = int32(i)
			cols[idx] = g.Indices[j]
			data[idx] = g.Data[j]
			idx++
		}
	}

	return rows, cols, data
}

// GetRow returns the column indices and values for a given row.
func (g *CSRMatrix) GetRow(row int) ([]int32, []float32) {
	start := g.Indptr[row]
	end := g.Indptr[row+1]
	return g.Indices[start:end], g.Data[start:end]
}
