package umap_test

import (
	"fmt"
	"testing"

	umapinit "github.com/nozzle/umap/init"
)

func TestRandomEmbeddingVsNumpy(t *testing.T) {
	// Expected values from Python:
	// np.random.RandomState(42).uniform(-10, 10, (15, 2))
	expected := [][]float32{
		{-2.5091977, 9.014286},
		{4.6398787, 1.9731697},
		{-6.879627, -6.88011},
		{-8.838327, 7.323523},
		{2.0223002, 4.1614513},
		{-9.58831, 9.398197},
		{6.648853, -5.7532177},
		{-6.3635006, -6.3319097},
		{-3.9151552, 0.49512863},
		{-1.3610996, -4.1754174},
		{2.237058, -7.2101226},
		{-4.157107, -2.672763},
		{-0.8786003, 5.7035193},
		{-6.0065246, 0.28468877},
		{1.8482914, -9.0709915},
	}

	embedding := umapinit.RandomEmbedding(15, 2, 42)

	fmt.Println("Comparing Go RandomEmbedding with NumPy:")
	allMatch := true
	for i := range expected {
		fmt.Printf("  [%d]: Go=[%.6f, %.6f], NumPy=[%.6f, %.6f]\n",
			i, embedding[i][0], embedding[i][1], expected[i][0], expected[i][1])

		for d := range 2 {
			diff := embedding[i][d] - expected[i][d]
			if diff < 0 {
				diff = -diff
			}
			if diff > 1e-5 {
				allMatch = false
				t.Errorf("Mismatch at [%d][%d]: got %.6f, expected %.6f", i, d, embedding[i][d], expected[i][d])
			}
		}
	}

	if allMatch {
		fmt.Println("All values match!")
	}
}
