package distance

import (
	"math"
	"testing"
)

func TestEuclidean(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{3, 4, 0}

	dist := Euclidean(a, b)
	expected := float32(5.0)

	if math.Abs(float64(dist-expected)) > 1e-5 {
		t.Errorf("Expected %f, got %f", expected, dist)
	}
}

func TestManhattan(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{3, 4, 5}

	dist := Manhattan(a, b)
	expected := float32(12.0)

	if math.Abs(float64(dist-expected)) > 1e-5 {
		t.Errorf("Expected %f, got %f", expected, dist)
	}
}

func TestCosine(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}

	dist := Cosine(a, b)
	// Orthogonal vectors have cosine similarity 0, distance 1
	expected := float32(1.0)

	if math.Abs(float64(dist-expected)) > 1e-5 {
		t.Errorf("Expected %f, got %f", expected, dist)
	}

	// Same direction should have distance 0
	c := []float32{1, 0, 0}
	dist2 := Cosine(a, c)
	if math.Abs(float64(dist2)) > 1e-5 {
		t.Errorf("Same direction should have distance 0, got %f", dist2)
	}
}

func TestHamming(t *testing.T) {
	a := []float32{1, 0, 1, 0}
	b := []float32{1, 1, 0, 0}

	dist := Hamming(a, b)
	expected := float32(0.5) // 2 out of 4 differ

	if math.Abs(float64(dist-expected)) > 1e-5 {
		t.Errorf("Expected %f, got %f", expected, dist)
	}
}

func TestRegistry(t *testing.T) {
	metrics := []string{"euclidean", "l2", "manhattan", "l1", "cosine", "chebyshev"}

	for _, name := range metrics {
		fn, ok := Get(name)
		if !ok {
			t.Errorf("Metric %s not found in registry", name)
			continue
		}

		a := []float32{1, 2, 3}
		b := []float32{4, 5, 6}
		dist := fn(a, b)

		if math.IsNaN(float64(dist)) || math.IsInf(float64(dist), 0) {
			t.Errorf("Metric %s returned non-finite value: %f", name, dist)
		}
	}
}

func TestGradRegistry(t *testing.T) {
	metrics := []string{"euclidean", "manhattan", "cosine", "canberra"}

	for _, name := range metrics {
		fn, ok := GetGrad(name)
		if !ok {
			t.Errorf("Gradient metric %s not found in registry", name)
			continue
		}

		a := []float32{1, 2, 3}
		b := []float32{4, 5, 6}
		dist, grad := fn(a, b)

		if math.IsNaN(float64(dist)) || math.IsInf(float64(dist), 0) {
			t.Errorf("Metric %s returned non-finite distance: %f", name, dist)
		}

		if len(grad) != len(a) {
			t.Errorf("Gradient for %s has wrong length: %d, expected %d", name, len(grad), len(a))
		}

		for i, g := range grad {
			if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
				t.Errorf("Metric %s returned non-finite gradient at %d: %f", name, i, g)
			}
		}
	}
}

func TestIsAngular(t *testing.T) {
	if !IsAngular("cosine") {
		t.Error("cosine should be angular")
	}
	if !IsAngular("correlation") {
		t.Error("correlation should be angular")
	}
	if IsAngular("euclidean") {
		t.Error("euclidean should not be angular")
	}
}

func BenchmarkEuclidean(b *testing.B) {
	x := make([]float32, 100)
	y := make([]float32, 100)
	for i := range x {
		x[i] = float32(i)
		y[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Euclidean(x, y)
	}
}

func BenchmarkCosine(b *testing.B) {
	x := make([]float32, 100)
	y := make([]float32, 100)
	for i := range x {
		x[i] = float32(i)
		y[i] = float32(i + 1)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Cosine(x, y)
	}
}
