// Package layout provides the optimization layout algorithms for UMAP.
// This implements stochastic gradient descent to optimize the low-dimensional embedding.
package layout

import (
	"github.com/nozzle/umap/graph"
	"github.com/nozzle/umap/internal/parallel"
	"github.com/nozzle/umap/internal/rand"
)

// LayoutConfig configures the layout optimization.
type LayoutConfig struct {
	// MinDist is the minimum distance between points in the embedding
	MinDist float32
	// Spread is the effective scale of embedded points
	Spread float32
	// NegativeSampleRate is the ratio of negative samples per positive sample
	NegativeSampleRate int
	// NEpochs is the number of epochs to run
	NEpochs int
	// LearningRate is the initial learning rate (alpha)
	LearningRate float32
	// Seed for random number generation
	Seed int64
	// NumWorkers for parallel processing (0 = auto)
	NumWorkers int
	// Verbose enables progress output
	Verbose bool
	// ProgressCallback is called after each epoch with (epoch, totalEpochs)
	ProgressCallback func(epoch, total int)
}

// DefaultLayoutConfig returns default configuration.
func DefaultLayoutConfig() LayoutConfig {
	return LayoutConfig{
		MinDist:            0.1,
		Spread:             1.0,
		NegativeSampleRate: 5,
		NEpochs:            200,
		LearningRate:       1.0,
		Seed:               42,
		NumWorkers:         0,
		Verbose:            false,
	}
}

// curveParameters stores the a and b parameters for the curve
// that maps distance in embedding space to membership strength.
type curveParameters struct {
	a, b float32
}

// findABParams finds parameters a, b such that:
// 1 / (1 + a * d^(2b)) approximates a smooth curve with:
// - value 1 at d = 0
// - value 0.5 at d = minDist
// - approaching 0 as d -> infinity
func findABParams(minDist, spread float32) curveParameters {
	// Use the approximation from the UMAP paper
	// For typical values, this gives reasonable results
	// A more accurate approach would use scipy.optimize.curve_fit

	// Simple approximation based on minDist and spread
	// These formulas are derived from fitting the original UMAP curve
	if minDist <= 0 {
		minDist = 0.001
	}

	// Target: 1 / (1 + a * minDist^(2*b)) = 0.5
	// => a * minDist^(2*b) = 1
	// => log(a) + 2*b*log(minDist) = 0
	// => log(a) = -2*b*log(minDist)

	// Use b = 1 as a starting point and adjust
	b := float32(1.0)
	if spread > 1 {
		b = 1.0 / spread
	}

	// a = 1 / minDist^(2*b)
	a := 1.0 / pow32(minDist, 2*b)

	return curveParameters{a: a, b: b}
}

// OptimizeLayout runs the SGD optimization loop.
func OptimizeLayout(
	embedding [][]float32,
	g *graph.CSRMatrix,
	config LayoutConfig,
) {
	n := len(embedding)
	if n == 0 {
		return
	}
	dim := len(embedding[0])

	numWorkers := config.NumWorkers
	if numWorkers <= 0 {
		numWorkers = parallel.NumWorkers()
	}

	// Find curve parameters
	params := findABParams(config.MinDist, config.Spread)
	a, b := params.a, params.b

	// Compute epochs per sample for each edge
	epochsPerSample := graph.ToEpochsPerSample(g, config.NEpochs)

	// Get edges
	heads, tails, _ := g.GetEdges()
	nEdges := len(heads)

	// Track when each edge should next be sampled
	epochsPerNegSample := make([]float32, nEdges)
	epochOfNextSample := make([]float32, nEdges)
	epochOfNextNegSample := make([]float32, nEdges)

	for i := 0; i < nEdges; i++ {
		epochsPerNegSample[i] = epochsPerSample[i] / float32(config.NegativeSampleRate)
		epochOfNextSample[i] = epochsPerSample[i]
		epochOfNextNegSample[i] = epochsPerNegSample[i]
	}

	// Initialize random state per worker
	rngs := make([]rand.State, numWorkers)
	for i := 0; i < numWorkers; i++ {
		rngs[i] = rand.New(config.Seed + int64(i))
	}

	// Optimization loop
	alpha := config.LearningRate
	moveOther := float32(1.0) // Move both endpoints of positive edges

	for epoch := 0; epoch < config.NEpochs; epoch++ {
		// Process edges in parallel chunks
		parallel.ParallelFor(0, nEdges, numWorkers, func(edge int) {
			workerID := edge % numWorkers
			rng := &rngs[workerID]

			// Check if this edge should be sampled this epoch
			if float32(epoch) < epochOfNextSample[edge] {
				return
			}

			i := int(heads[edge])
			j := int(tails[edge])

			// Skip self-loops
			if i == j {
				return
			}

			// Attractive force (positive sample)
			applyAttractive(embedding, i, j, dim, a, b, alpha, moveOther)

			// Update next sample time
			epochOfNextSample[edge] += epochsPerSample[edge]

			// Negative sampling
			for {
				if float32(epoch) < epochOfNextNegSample[edge] {
					break
				}

				// Sample random negative
				k := rand.Intn(rng, n)
				if k == i || k == j {
					continue
				}

				// Repulsive force
				applyRepulsive(embedding, i, k, dim, a, b, alpha)

				epochOfNextNegSample[edge] += epochsPerNegSample[edge]

				// Limit negative samples per edge per epoch
				if epochOfNextNegSample[edge] > float32(epoch)+1 {
					break
				}
			}
		})

		// Decay learning rate
		alpha = config.LearningRate * (1.0 - float32(epoch)/float32(config.NEpochs))
		if alpha < 0.0001 {
			alpha = 0.0001
		}

		// Progress callback
		if config.ProgressCallback != nil {
			config.ProgressCallback(epoch+1, config.NEpochs)
		}
	}
}

// applyAttractive applies attractive force between points i and j.
func applyAttractive(embedding [][]float32, i, j, dim int, a, b, alpha, moveOther float32) {
	// Compute squared distance
	var distSq float32
	for d := 0; d < dim; d++ {
		diff := embedding[i][d] - embedding[j][d]
		distSq += diff * diff
	}

	if distSq < 1e-10 {
		distSq = 1e-10
	}

	// Gradient of attractive force
	// grad = -2ab * d^(2b-2) / (1 + a*d^(2b))
	gradCoef := float32(0)
	if distSq > 0 {
		d2b := pow32(distSq, b)
		gradCoef = (-2.0 * a * b * pow32(distSq, b-1)) / (1.0 + a*d2b)
	}

	// Apply gradient with clipping
	for d := 0; d < dim; d++ {
		diff := embedding[i][d] - embedding[j][d]
		grad := clip(gradCoef * diff)

		embedding[i][d] += alpha * grad
		if moveOther > 0 {
			embedding[j][d] -= alpha * grad * moveOther
		}
	}
}

// applyRepulsive applies repulsive force between points i and k.
func applyRepulsive(embedding [][]float32, i, k, dim int, a, b, alpha float32) {
	// Compute squared distance
	var distSq float32
	for d := 0; d < dim; d++ {
		diff := embedding[i][d] - embedding[k][d]
		distSq += diff * diff
	}

	if distSq < 1e-10 {
		distSq = 1e-10
	}

	// Gradient of repulsive force
	// grad = 2b / (d^2 * (1 + a*d^(2b)))
	gradCoef := float32(0)
	if distSq > 0 {
		d2b := pow32(distSq, b)
		gradCoef = (2.0 * b) / ((0.001 + distSq) * (1.0 + a*d2b))
	}

	// Apply gradient with clipping
	for d := 0; d < dim; d++ {
		diff := embedding[i][d] - embedding[k][d]
		grad := clip(gradCoef * diff)

		embedding[i][d] += alpha * grad
	}
}

// clip clamps a value to [-4, 4] range.
func clip(val float32) float32 {
	if val > 4.0 {
		return 4.0
	}
	if val < -4.0 {
		return -4.0
	}
	return val
}

// pow32 computes x^y for float32.
func pow32(x, y float32) float32 {
	// Fast path for common exponents
	if y == 1 {
		return x
	}
	if y == 0 {
		return 1
	}
	if y == 0.5 {
		return sqrt32(x)
	}
	if y == 2 {
		return x * x
	}

	// General case using exp/log
	if x <= 0 {
		return 0
	}
	return exp32(y * log32(x))
}

func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func exp32(x float32) float32 {
	// Limit input range
	if x > 88 {
		return 1e38
	}
	if x < -88 {
		return 0
	}

	// Taylor series approximation
	sum := float32(1.0)
	term := float32(1.0)
	for i := 1; i < 20; i++ {
		term *= x / float32(i)
		sum += term
		if term < 1e-10 && term > -1e-10 {
			break
		}
	}
	return sum
}

func log32(x float32) float32 {
	if x <= 0 {
		return -1e38
	}
	// Use Newton's method for ln
	y := (x - 1) / (x + 1)
	y2 := y * y
	sum := y
	term := y
	for i := 1; i < 50; i++ {
		term *= y2
		sum += term / float32(2*i+1)
	}
	return 2 * sum
}
