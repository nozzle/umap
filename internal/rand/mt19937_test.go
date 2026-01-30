package rand_test

import (
	"fmt"
	"testing"

	"github.com/nozzle/umap/internal/rand"
)

func TestMT19937VsNumpy(t *testing.T) {
	mt := rand.NewMT19937(42)

	// Expected values from Python: numpy.random.RandomState(42).uniform(-10, 10, 20)
	expected := []float64{
		-2.509197623052750,
		9.014286128198323,
		4.639878836228101,
		1.973169683940732,
		-6.879627191151270,
		-6.880109593275947,
		-8.838327756636010,
		7.323522915498703,
		2.022300234864176,
		4.161451555920910,
		-9.588310114083951,
		9.398197043239886,
		6.648852816008435,
		-5.753217786434477,
		-6.363500655857988,
		-6.331909802931324,
		-3.915155140809246,
		0.495128632644757,
		-1.361099627157685,
		-4.175417196039161,
	}

	fmt.Println("Comparing Go MT19937 with NumPy RandomState(42):")
	for i, exp := range expected {
		got := mt.Uniform(-10.0, 10.0)
		diff := got - exp
		if diff < 0 {
			diff = -diff
		}
		fmt.Printf("  %2d: Go=%.15f  NumPy=%.15f  diff=%.2e\n", i, got, exp, diff)
		if diff > 1e-6 {
			t.Errorf("Value %d: got %.15f, expected %.15f, diff %.2e", i, got, exp, diff)
		}
	}
}

func TestMT19937RNGState(t *testing.T) {
	// Test that after 30 uniform calls (for 15x2 init embedding),
	// the next 3 randint values match what Python generates for rng_state
	mt := rand.NewMT19937(42)

	// Consume 30 uniform values (simulating init embedding generation)
	for range 30 {
		_ = mt.Uniform(-10.0, 10.0)
	}

	// Expected rng_state values from Python:
	// random_state.randint(INT32_MIN, INT32_MAX+1, 3) after 30 uniform calls
	expectedRNGState := []int32{461901618, 774414982, -1415088108}

	fmt.Println("Comparing rng_state values:")
	for i, exp := range expectedRNGState {
		got := mt.RandInt32()
		fmt.Printf("  %d: Go=%d, NumPy=%d\n", i, got, exp)
		if got != exp {
			t.Errorf("RNG state %d: got %d, expected %d", i, got, exp)
		}
	}
}
