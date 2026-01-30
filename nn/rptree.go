// Package nn provides nearest neighbor search algorithms for UMAP.
// This file implements Random Projection Trees (RP-trees) for initialization.
package nn

import (
	"sort"

	"github.com/nozzle/umap/distance"
	"github.com/nozzle/umap/internal/rand"
)

// RPTree represents a random projection tree for approximate nearest neighbor search.
type RPTree struct {
	// Hyperplane normal for splits (nil for leaf nodes)
	Hyperplane []float32
	// Offset for the hyperplane decision
	Offset float32
	// Left and Right children (nil for leaves)
	Left  *RPTree
	Right *RPTree
	// Indices of points in this leaf (empty for internal nodes)
	Indices []int32
	// IsLeaf indicates if this is a leaf node
	IsLeaf bool
}

// RPForest is a collection of RP-trees for approximate nearest neighbor search.
type RPForest struct {
	Trees    []*RPTree
	Data     [][]float32
	Metric   string
	LeafSize int
}

// RPForestConfig configures RP-forest construction.
type RPForestConfig struct {
	// NumTrees is the number of trees to build
	NumTrees int
	// LeafSize is the maximum number of points in a leaf
	LeafSize int
	// Angular indicates whether to use angular RP-trees
	Angular bool
	// Seed for random number generation
	Seed int64
}

// DefaultRPForestConfig returns default configuration.
func DefaultRPForestConfig() RPForestConfig {
	return RPForestConfig{
		NumTrees: 10,
		LeafSize: 30,
		Angular:  false,
		Seed:     42,
	}
}

// BuildRPForest builds a random projection forest from data.
func BuildRPForest(data [][]float32, config RPForestConfig) *RPForest {
	n := len(data)
	if n == 0 {
		return &RPForest{}
	}

	rng := rand.New(config.Seed)
	trees := make([]*RPTree, config.NumTrees)

	for t := 0; t < config.NumTrees; t++ {
		// Create index list for all points
		indices := make([]int32, n)
		for i := range indices {
			indices[i] = int32(i)
		}

		trees[t] = buildRPTree(data, indices, config.LeafSize, config.Angular, &rng)
	}

	return &RPForest{
		Trees:    trees,
		Data:     data,
		LeafSize: config.LeafSize,
	}
}

// buildRPTree recursively builds an RP-tree.
func buildRPTree(data [][]float32, indices []int32, leafSize int, angular bool, rng *rand.State) *RPTree {
	if len(indices) <= leafSize {
		// Create leaf node
		leafIndices := make([]int32, len(indices))
		copy(leafIndices, indices)
		return &RPTree{
			Indices: leafIndices,
			IsLeaf:  true,
		}
	}

	// Select two random points to define the split
	dim := len(data[0])
	i := rand.Intn(rng, len(indices))
	j := rand.Intn(rng, len(indices))
	for j == i {
		j = rand.Intn(rng, len(indices))
	}

	p1 := data[indices[i]]
	p2 := data[indices[j]]

	// Compute hyperplane normal (p2 - p1) and offset
	hyperplane := make([]float32, dim)
	var offset float32

	if angular {
		// For angular metrics, normalize the hyperplane
		var normSq float32
		for d := range dim {
			hyperplane[d] = p2[d] - p1[d]
			normSq += hyperplane[d] * hyperplane[d]
		}
		if normSq > 0 {
			invNorm := 1.0 / sqrt32(normSq)
			for d := range dim {
				hyperplane[d] *= invNorm
			}
		}
		// Offset is 0 for angular splits
		offset = 0
	} else {
		// For Euclidean metrics, use the midpoint
		var midpoint float32
		for d := range dim {
			hyperplane[d] = p2[d] - p1[d]
			midpoint += (p1[d] + p2[d]) / 2 * hyperplane[d]
		}
		offset = midpoint
	}

	// Partition points based on side of hyperplane
	leftIndices := make([]int32, 0, len(indices)/2)
	rightIndices := make([]int32, 0, len(indices)/2)

	for _, idx := range indices {
		var side float32
		point := data[idx]
		for d := range dim {
			side += point[d] * hyperplane[d]
		}

		if side < offset {
			leftIndices = append(leftIndices, idx)
		} else {
			rightIndices = append(rightIndices, idx)
		}
	}

	// Handle degenerate splits
	if len(leftIndices) == 0 || len(rightIndices) == 0 {
		// Random split
		rand.Shuffle(rng, indices)
		mid := len(indices) / 2
		leftIndices = indices[:mid]
		rightIndices = indices[mid:]
	}

	return &RPTree{
		Hyperplane: hyperplane,
		Offset:     offset,
		Left:       buildRPTree(data, leftIndices, leafSize, angular, rng),
		Right:      buildRPTree(data, rightIndices, leafSize, angular, rng),
		IsLeaf:     false,
	}
}

// SearchTree finds the leaf containing a query point.
func (t *RPTree) SearchTree(query []float32) []int32 {
	if t.IsLeaf {
		return t.Indices
	}

	var side float32
	for d := range query {
		side += query[d] * t.Hyperplane[d]
	}

	if side < t.Offset {
		return t.Left.SearchTree(query)
	}
	return t.Right.SearchTree(query)
}

// SearchForest finds candidate neighbors from all trees.
func (f *RPForest) SearchForest(query []float32) []int32 {
	candidateSet := make(map[int32]bool)

	for _, tree := range f.Trees {
		candidates := tree.SearchTree(query)
		for _, idx := range candidates {
			candidateSet[idx] = true
		}
	}

	result := make([]int32, 0, len(candidateSet))
	for idx := range candidateSet {
		result = append(result, idx)
	}
	return result
}

// InitializeFromForest initializes k-NN graph using RP-forest.
// This provides better starting neighbors than random initialization.
func InitializeFromForest(data [][]float32, k int, forest *RPForest, metric string) *KNNGraph {
	n := len(data)
	if k >= n {
		k = n - 1
	}

	distFunc, ok := distance.Get(metric)
	if !ok {
		distFunc = distance.Euclidean
	}

	indices := make([][]int32, n)
	dists := make([][]float32, n)

	for i := range n {
		// Get candidates from forest
		candidates := forest.SearchForest(data[i])

		// Initialize heap
		indices[i] = make([]int32, k)
		dists[i] = make([]float32, k)
		for j := 0; j < k; j++ {
			indices[i][j] = -1
			dists[i][j] = 1e30
		}

		// Add candidates to heap
		for _, cand := range candidates {
			if int(cand) == i {
				continue
			}
			d := distFunc(data[i], data[cand])
			simpleHeapPush(indices[i], dists[i], k, cand, d)
		}

		// If we don't have enough neighbors, add random ones
		if countValidNeighbors(indices[i]) < k {
			addRandomNeighbors(data, i, indices[i], dists[i], k, distFunc, int64(i))
		}
	}

	// Sort results
	for i := range n {
		sortNeighbors(indices[i], dists[i])
	}

	return &KNNGraph{
		Indices:   indices,
		Distances: dists,
		N:         n,
		K:         k,
	}
}

// simpleHeapPush adds to max-heap
func simpleHeapPush(indices []int32, dists []float32, k int, idx int32, dist float32) bool {
	if dist >= dists[0] {
		return false
	}

	// Check for duplicates
	for i := range k {
		if indices[i] == idx {
			return false
		}
	}

	dists[0] = dist
	indices[0] = idx

	// Sift down
	i := 0
	for {
		left := 2*i + 1
		right := 2*i + 2

		if left >= k {
			break
		}

		swap := i
		if dists[left] > dists[swap] {
			swap = left
		}
		if right < k && dists[right] > dists[swap] {
			swap = right
		}

		if swap == i {
			break
		}

		dists[i], dists[swap] = dists[swap], dists[i]
		indices[i], indices[swap] = indices[swap], indices[i]
		i = swap
	}

	return true
}

// countValidNeighbors counts neighbors with valid indices
func countValidNeighbors(indices []int32) int {
	count := 0
	for _, idx := range indices {
		if idx >= 0 {
			count++
		}
	}
	return count
}

// addRandomNeighbors adds random neighbors until we have k
func addRandomNeighbors(data [][]float32, point int, indices []int32, dists []float32, k int, distFunc distance.Func, seed int64) {
	n := len(data)
	rng := rand.New(seed)

	for countValidNeighbors(indices) < k {
		j := rand.Intn(&rng, n)
		if j == point {
			continue
		}
		d := distFunc(data[point], data[j])
		simpleHeapPush(indices, dists, k, int32(j), d)
	}
}

// sortNeighbors sorts indices and distances in ascending order
func sortNeighbors(indices []int32, dists []float32) {
	type pair struct {
		idx  int32
		dist float32
	}
	pairs := make([]pair, len(indices))
	for i := range pairs {
		pairs[i] = pair{indices[i], dists[i]}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].dist < pairs[j].dist
	})

	for i := range pairs {
		indices[i] = pairs[i].idx
		dists[i] = pairs[i].dist
	}
}

func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	z := x
	for range 10 {
		z = (z + x/z) / 2
	}
	return z
}
