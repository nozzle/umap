// Package umap implements the UMAP (Uniform Manifold Approximation and Projection)
// dimensionality reduction algorithm.
//
// UMAP is a dimension reduction technique that can be used for visualization
// similarly to t-SNE, but also for general non-linear dimension reduction.
//
// This is a Go port of the original Python implementation by Leland McInnes:
// https://github.com/lmcinnes/umap
//
// Basic usage:
//
//	model := umap.New(umap.DefaultConfig())
//	embedding := model.FitTransform(data)
package umap

import (
	"github.com/nozzle/umap/graph"
	umapinit "github.com/nozzle/umap/init"
	"github.com/nozzle/umap/internal/rand"
	"github.com/nozzle/umap/layout"
	"github.com/nozzle/umap/nn"
)

// Config configures the UMAP algorithm.
type Config struct {
	// NNeighbors is the number of neighbors for k-NN graph construction.
	// Larger values capture more global structure but are slower.
	// Default: 15
	NNeighbors int

	// NComponents is the dimensionality of the target embedding.
	// Default: 2
	NComponents int

	// Metric is the distance metric to use.
	// Options: "euclidean", "manhattan", "cosine", "correlation", etc.
	// Default: "euclidean"
	Metric string

	// MinDist is the effective minimum distance between embedded points.
	// Smaller values create tighter clusters but may lose global structure.
	// Default: 0.1
	MinDist float32

	// Spread is the effective scale of embedded points.
	// In combination with MinDist, this controls the clumpiness of the embedding.
	// Default: 1.0
	Spread float32

	// NEpochs is the number of training epochs.
	// Larger values result in more accurate embeddings but take longer.
	// Default: 200 for large datasets, 500 for small datasets
	NEpochs int

	// LearningRate is the initial learning rate for SGD.
	// Default: 1.0
	LearningRate float32

	// NegativeSampleRate is the number of negative samples per positive sample.
	// Default: 5
	NegativeSampleRate int

	// Init is the initialization method.
	// Options: "spectral" or "random"
	// Default: "spectral"
	Init string

	// LocalConnectivity controls how local the connectivity estimate is.
	// Default: 1.0
	LocalConnectivity float64

	// SetOpMixRatio controls the blend between fuzzy set union and intersection.
	// 0.0 = pure intersection, 1.0 = pure union
	// Default: 1.0
	SetOpMixRatio float64

	// Seed for random number generation.
	// Use a fixed seed for reproducible results.
	// Default: 42
	Seed int64

	// NumWorkers for parallel processing.
	// 0 = auto-detect based on CPU cores.
	// Default: 0
	NumWorkers int

	// Verbose enables progress output.
	// Default: false
	Verbose bool

	// ProgressCallback is called after each epoch with (epoch, totalEpochs).
	// Default: nil
	ProgressCallback func(epoch, total int)
}

// DefaultConfig returns the default UMAP configuration.
func DefaultConfig() Config {
	return Config{
		NNeighbors:         15,
		NComponents:        2,
		Metric:             "euclidean",
		MinDist:            0.1,
		Spread:             1.0,
		NEpochs:            200,
		LearningRate:       1.0,
		NegativeSampleRate: 5,
		Init:               "spectral",
		LocalConnectivity:  1.0,
		SetOpMixRatio:      1.0,
		Seed:               42,
		NumWorkers:         0,
		Verbose:            false,
	}
}

// UMAP is the main UMAP model.
type UMAP struct {
	Config Config

	// Learned state after fitting
	graph     *graph.CSRMatrix
	embedding [][]float32
	knnGraph  *nn.KNNGraph
}

// New creates a new UMAP model with the given configuration.
func New(config Config) *UMAP {
	return &UMAP{Config: config}
}

// FitTransform fits the model to the data and returns the embedding.
func (u *UMAP) FitTransform(data [][]float32) [][]float32 {
	u.Fit(data)
	return u.embedding
}

// Fit fits the model to the training data.
func (u *UMAP) Fit(data [][]float32) {
	n := len(data)
	if n == 0 {
		return
	}

	// Determine number of epochs based on dataset size
	nEpochs := u.Config.NEpochs
	if nEpochs <= 0 {
		if n < 10000 {
			nEpochs = 500
		} else {
			nEpochs = 200
		}
	}

	// Step 1: Build k-NN graph
	u.knnGraph = u.buildKNNGraph(data)

	// Step 2: Construct fuzzy simplicial set
	u.graph = u.buildFuzzySimplicialSet()

	// Step 3: Initialize embedding using MT19937
	// This matches Python's numpy.random.RandomState(seed)
	mt := rand.NewMT19937(uint32(u.Config.Seed))
	u.embedding = u.initializeEmbeddingWithRNG(n, mt)

	// Step 4: Generate rng_state for SGD from the same MT19937
	// This matches Python: random_state.randint(INT32_MIN, INT32_MAX, 3)
	rngState := []int64{
		int64(mt.RandInt32()),
		int64(mt.RandInt32()),
		int64(mt.RandInt32()),
	}

	// Step 5: Optimize layout with the correct rng_state
	u.optimizeLayoutWithRNGState(nEpochs, rngState)
}

// Transform transforms new data using the fitted model.
// Note: This is a simplified implementation that uses nearest neighbor lookup.
func (u *UMAP) Transform(newData [][]float32) [][]float32 {
	if len(u.embedding) == 0 {
		return nil
	}

	// For now, use a simple nearest-neighbor approach
	// A full implementation would use out-of-sample extension
	n := len(newData)
	result := make([][]float32, n)

	for i := range n {
		result[i] = make([]float32, u.Config.NComponents)

		// Find nearest neighbor in training data and copy its embedding
		// This is a placeholder - proper implementation would do interpolation
		bestDist := float32(1e30)
		bestIdx := 0

		for j := 0; j < len(u.knnGraph.Indices); j++ {
			dist := euclideanDist(newData[i], getData(u.knnGraph, j))
			if dist < bestDist {
				bestDist = dist
				bestIdx = j
			}
		}

		copy(result[i], u.embedding[bestIdx])
	}

	return result
}

// Embedding returns the current embedding.
func (u *UMAP) Embedding() [][]float32 {
	return u.embedding
}

// buildKNNGraph constructs the k-nearest neighbor graph.
func (u *UMAP) buildKNNGraph(data [][]float32) *nn.KNNGraph {
	n := len(data)

	// For small datasets, use brute force
	if n < 1000 {
		return nn.BruteForceKNN(data, u.Config.NNeighbors, u.Config.Metric)
	}

	// For larger datasets, use NNDescent
	config := nn.DefaultNNDescentConfig()
	config.K = u.Config.NNeighbors
	config.Metric = u.Config.Metric
	config.Seed = u.Config.Seed
	config.NumWorkers = u.Config.NumWorkers

	return nn.NNDescent(data, config)
}

// buildFuzzySimplicialSet constructs the fuzzy simplicial set from k-NN graph.
func (u *UMAP) buildFuzzySimplicialSet() *graph.CSRMatrix {
	config := graph.DefaultFuzzySimplicialSetConfig()
	config.LocalConnectivity = u.Config.LocalConnectivity
	config.SetOpMixRatio = u.Config.SetOpMixRatio
	config.NumWorkers = u.Config.NumWorkers

	return graph.FuzzySimplicialSet(
		u.knnGraph.Indices,
		u.knnGraph.Distances,
		config,
	)
}

// initializeEmbedding creates the initial embedding.
func (u *UMAP) initializeEmbedding(n int) [][]float32 {
	var method umapinit.InitMethod
	switch u.Config.Init {
	case "spectral":
		method = umapinit.Spectral
	case "random":
		method = umapinit.Random
	default:
		method = umapinit.Spectral
	}

	return umapinit.InitializeEmbedding(
		u.graph,
		n,
		u.Config.NComponents,
		method,
		u.Config.Seed,
	)
}

// initializeEmbeddingWithRNG creates the initial embedding using the provided MT19937 RNG.
// This allows the RNG state to be continued for SGD initialization.
func (u *UMAP) initializeEmbeddingWithRNG(n int, mt *rand.MT19937) [][]float32 {
	switch u.Config.Init {
	case "random":
		// Use the provided MT19937 directly for NumPy compatibility
		return randomEmbeddingWithMT(n, u.Config.NComponents, mt)
	case "spectral":
		// For spectral, use the standard method
		// Note: spectral init doesn't consume from the random state in the same way
		return umapinit.InitializeEmbedding(
			u.graph,
			n,
			u.Config.NComponents,
			umapinit.Spectral,
			u.Config.Seed,
		)
	default:
		return umapinit.InitializeEmbedding(
			u.graph,
			n,
			u.Config.NComponents,
			umapinit.Spectral,
			u.Config.Seed,
		)
	}
}

// randomEmbeddingWithMT generates a random embedding using MT19937.
func randomEmbeddingWithMT(n, dim int, mt *rand.MT19937) [][]float32 {
	result := make([][]float32, n)
	for i := range n {
		result[i] = make([]float32, dim)
		for d := range dim {
			result[i][d] = mt.UniformFloat32(-10.0, 10.0)
		}
	}
	return result
}

// optimizeLayout runs the SGD optimization.
func (u *UMAP) optimizeLayout(nEpochs int) {
	config := layout.DefaultLayoutConfig()
	config.MinDist = u.Config.MinDist
	config.Spread = u.Config.Spread
	config.NEpochs = nEpochs
	config.LearningRate = u.Config.LearningRate
	config.NegativeSampleRate = u.Config.NegativeSampleRate
	config.Seed = u.Config.Seed
	config.NumWorkers = u.Config.NumWorkers
	config.Verbose = u.Config.Verbose
	config.ProgressCallback = u.Config.ProgressCallback

	layout.OptimizeLayout(u.embedding, u.graph, config)
}

// optimizeLayoutWithRNGState runs the SGD optimization with a specific RNG state.
// This allows for exact reproducibility matching Python UMAP.
func (u *UMAP) optimizeLayoutWithRNGState(nEpochs int, rngState []int64) {
	config := layout.DefaultLayoutConfig()
	config.MinDist = u.Config.MinDist
	config.Spread = u.Config.Spread
	config.NEpochs = nEpochs
	config.LearningRate = u.Config.LearningRate
	config.NegativeSampleRate = u.Config.NegativeSampleRate
	config.Seed = u.Config.Seed
	config.NumWorkers = u.Config.NumWorkers
	config.Verbose = u.Config.Verbose
	config.ProgressCallback = u.Config.ProgressCallback
	config.RNGState = rngState

	layout.OptimizeLayout(u.embedding, u.graph, config)
}

// euclideanDist computes Euclidean distance between two vectors.
func euclideanDist(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum // Return squared distance for efficiency
}

// getData is a placeholder - in a real implementation we'd store the original data
func getData(g *nn.KNNGraph, idx int) []float32 {
	// This is a placeholder. For proper Transform support,
	// we would need to store the original training data.
	return nil
}
