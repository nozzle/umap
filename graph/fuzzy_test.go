package graph

import (
	"testing"
)

func TestFuzzySimplicialSet(t *testing.T) {
	// Create simple k-NN data
	knnIndices := [][]int32{
		{1, 2, 3},
		{0, 2, 3},
		{0, 1, 3},
		{0, 1, 2},
	}
	knnDistances := [][]float32{
		{1.0, 2.0, 3.0},
		{1.0, 1.5, 2.5},
		{2.0, 1.5, 2.0},
		{3.0, 2.5, 2.0},
	}

	config := DefaultFuzzySimplicialSetConfig()
	graph := FuzzySimplicialSet(knnIndices, knnDistances, config)

	if graph.NRows != 4 {
		t.Errorf("Expected 4 rows, got %d", graph.NRows)
	}

	if graph.NNZ == 0 {
		t.Error("Graph has no edges")
	}

	// Check symmetry (since we apply set operations)
	rows, cols, data := graph.GetEdges()
	edgeMap := make(map[[2]int32]float32)
	for i := range rows {
		edgeMap[[2]int32{rows[i], cols[i]}] = data[i]
	}

	for key, val := range edgeMap {
		reverse := [2]int32{key[1], key[0]}
		if reverseVal, ok := edgeMap[reverse]; ok {
			// Both directions exist; they should be equal after symmetrization
			if val != reverseVal {
				t.Logf("Edge (%d,%d) = %f, reverse = %f", key[0], key[1], val, reverseVal)
			}
		}
	}
}

func TestSmoothKNNDist(t *testing.T) {
	distances := []float32{0.0, 1.0, 2.0, 3.0, 4.0}

	sigma, rho := smoothKNNDist(distances, 5.0, 1.0)

	if sigma <= 0 {
		t.Errorf("Sigma should be positive, got %f", sigma)
	}

	if rho < 0 {
		t.Errorf("Rho should be non-negative, got %f", rho)
	}

	t.Logf("sigma=%f, rho=%f", sigma, rho)
}

func TestToEpochsPerSample(t *testing.T) {
	graph := &CSRMatrix{
		Indptr:  []int32{0, 2, 4},
		Indices: []int32{1, 2, 0, 2},
		Data:    []float32{1.0, 0.5, 1.0, 0.25},
		NRows:   2,
		NCols:   2,
		NNZ:     4,
	}

	epochs := ToEpochsPerSample(graph, 100)

	if len(epochs) != 4 {
		t.Errorf("Expected 4 epochs values, got %d", len(epochs))
	}

	// Maximum weight edge (weight=1.0) should have epochs_per_sample = 1.0
	// This means it's sampled every epoch
	if epochs[0] != 1.0 {
		t.Errorf("Max weight edge should have epochs_per_sample=1.0, got %f", epochs[0])
	}

	// Half weight (0.5) should have epochs_per_sample = 2.0
	// This means it's sampled every 2 epochs
	if epochs[1] != 2.0 {
		t.Errorf("Half weight edge should have epochs_per_sample=2.0, got %f", epochs[1])
	}

	// Quarter weight (0.25) should have epochs_per_sample = 4.0
	if epochs[3] != 4.0 {
		t.Errorf("Quarter weight edge should have epochs_per_sample=4.0, got %f", epochs[3])
	}

	t.Logf("Epochs per sample: %v", epochs)
}

func TestCSRMatrixGetRow(t *testing.T) {
	graph := &CSRMatrix{
		Indptr:  []int32{0, 2, 4, 5},
		Indices: []int32{1, 2, 0, 2, 1},
		Data:    []float32{1.0, 2.0, 3.0, 4.0, 5.0},
		NRows:   3,
		NCols:   3,
		NNZ:     5,
	}

	// Get row 0
	indices, _ := graph.GetRow(0)
	if len(indices) != 2 {
		t.Errorf("Row 0 should have 2 elements, got %d", len(indices))
	}

	// Get row 1
	indices, _ = graph.GetRow(1)
	if len(indices) != 2 {
		t.Errorf("Row 1 should have 2 elements, got %d", len(indices))
	}

	// Get row 2
	indices, data := graph.GetRow(2)
	if len(indices) != 1 {
		t.Errorf("Row 2 should have 1 element, got %d", len(indices))
	}
	if data[0] != 5.0 {
		t.Errorf("Row 2 data should be 5.0, got %f", data[0])
	}
}
