package nn

import (
	"testing"
)

func generateTestData(n, dim int, seed int64) [][]float32 {
	data := make([][]float32, n)
	rng := seed
	for i := range n {
		data[i] = make([]float32, dim)
		for j := range dim {
			rng = (rng*6364136223846793005 + 1442695040888963407) & 0x7FFFFFFF
			data[i][j] = float32(rng) / float32(0x7FFFFFFF)
		}
	}
	return data
}

func TestBruteForceKNN(t *testing.T) {
	data := generateTestData(100, 10, 42)

	graph := BruteForceKNN(data, 10, "euclidean")

	if graph.N != 100 {
		t.Errorf("Expected N=100, got %d", graph.N)
	}
	if graph.K != 10 {
		t.Errorf("Expected K=10, got %d", graph.K)
	}

	// Check that each point has k neighbors
	for i := 0; i < graph.N; i++ {
		if len(graph.Indices[i]) != graph.K {
			t.Errorf("Point %d has %d neighbors, expected %d", i, len(graph.Indices[i]), graph.K)
		}

		// Check no self-loops
		for j := 0; j < graph.K; j++ {
			if graph.Indices[i][j] == int32(i) {
				t.Errorf("Self-loop found at point %d", i)
			}
		}

		// Check distances are sorted
		for j := 1; j < graph.K; j++ {
			if graph.Distances[i][j] < graph.Distances[i][j-1] {
				t.Errorf("Distances not sorted for point %d", i)
			}
		}
	}
}

func TestNNDescent(t *testing.T) {
	data := generateTestData(200, 10, 42)

	config := DefaultNNDescentConfig()
	config.K = 10
	config.MaxIterations = 5
	config.Seed = 42

	graph := NNDescent(data, config)

	if graph.N != 200 {
		t.Errorf("Expected N=200, got %d", graph.N)
	}
	if graph.K != 10 {
		t.Errorf("Expected K=10, got %d", graph.K)
	}

	// Check that results are somewhat sensible
	validNeighbors := 0
	for i := 0; i < graph.N; i++ {
		for j := 0; j < graph.K; j++ {
			if graph.Indices[i][j] >= 0 {
				validNeighbors++
			}
		}
	}

	expectedMin := graph.N * graph.K / 2 // At least half should be valid
	if validNeighbors < expectedMin {
		t.Errorf("Too few valid neighbors: %d, expected at least %d", validNeighbors, expectedMin)
	}
}

func TestRPForest(t *testing.T) {
	data := generateTestData(100, 10, 42)

	config := DefaultRPForestConfig()
	config.NumTrees = 5
	config.LeafSize = 10
	config.Seed = 42

	forest := BuildRPForest(data, config)

	if len(forest.Trees) != 5 {
		t.Errorf("Expected 5 trees, got %d", len(forest.Trees))
	}

	// Test search
	for i := range 10 {
		candidates := forest.SearchForest(data[i])
		if len(candidates) == 0 {
			t.Errorf("No candidates found for point %d", i)
		}
	}
}

func BenchmarkBruteForceKNN(b *testing.B) {
	data := generateTestData(500, 50, 42)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BruteForceKNN(data, 15, "euclidean")
	}
}

func BenchmarkNNDescent(b *testing.B) {
	data := generateTestData(1000, 50, 42)

	config := DefaultNNDescentConfig()
	config.K = 15
	config.MaxIterations = 5

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		NNDescent(data, config)
	}
}
