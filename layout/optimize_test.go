package layout

import (
	"fmt"
	"math"
	"testing"
)

func TestFindABParams(t *testing.T) {
	// Test with default UMAP parameters
	minDist := float32(0.1)
	spread := float32(1.0)

	params := findABParams(minDist, spread)

	// Expected values from Python's scipy.optimize.curve_fit
	expectedA := float32(1.5769434605754993)
	expectedB := float32(0.8950608781680347)

	fmt.Printf("Go:     a = %.15f, b = %.15f\n", params.a, params.b)
	fmt.Printf("Python: a = %.15f, b = %.15f\n", expectedA, expectedB)

	// Check relative error
	relErrA := math.Abs(float64(params.a-expectedA)) / float64(expectedA)
	relErrB := math.Abs(float64(params.b-expectedB)) / float64(expectedB)

	fmt.Printf("Relative error: a = %.2e, b = %.2e\n", relErrA, relErrB)

	// Should be within 1%
	if relErrA > 0.01 {
		t.Errorf("Parameter 'a' too far from expected: got %.6f, expected %.6f", params.a, expectedA)
	}
	if relErrB > 0.01 {
		t.Errorf("Parameter 'b' too far from expected: got %.6f, expected %.6f", params.b, expectedB)
	}
}
