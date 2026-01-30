// Package rand provides fast pseudo-random number generation
// using the Tausworthe algorithm, matching the Python UMAP implementation.
package rand

// State holds the internal state of the Tausworthe PRNG.
// It requires 3 int64 values for the combined generator.
type State [3]int64

// New creates a new random state from a seed.
func New(seed int64) State {
	// Initialize state from seed using simple LCG
	s := State{}
	s[0] = seed
	if s[0] == 0 {
		s[0] = 1
	}
	s[1] = s[0]*6364136223846793005 + 1442695040888963407
	s[2] = s[1]*6364136223846793005 + 1442695040888963407
	// Warm up
	for i := 0; i < 10; i++ {
		Int(&s)
	}
	return s
}

// Int generates a pseudo-random int32 using the Tausworthe algorithm.
// This matches the tau_rand_int function in the Python implementation.
func Int(state *State) int32 {
	state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^
		((((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19)
	state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^
		((((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25)
	state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^
		((((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11)
	return int32(state[0] ^ state[1] ^ state[2])
}

// Float32 generates a pseudo-random float32 in [0, 1).
func Float32(state *State) float32 {
	i := Int(state)
	if i < 0 {
		i = -i
	}
	return float32(i) / float32(0x7FFFFFFF)
}

// Float64 generates a pseudo-random float64 in [0, 1).
func Float64(state *State) float64 {
	i := Int(state)
	if i < 0 {
		i = -i
	}
	return float64(i) / float64(0x7FFFFFFF)
}

// Intn returns a non-negative pseudo-random int in [0, n).
func Intn(state *State, n int) int {
	if n <= 0 {
		return 0
	}
	i := Int(state)
	if i < 0 {
		i = -i
	}
	return int(i) % n
}

// Shuffle randomly shuffles a slice of int32.
func Shuffle(state *State, arr []int32) {
	n := len(arr)
	for i := n - 1; i > 0; i-- {
		j := Intn(state, i+1)
		arr[i], arr[j] = arr[j], arr[i]
	}
}

// NormFloat64 returns a normally distributed float64 with mean 0 and stddev 1.
// Uses the Box-Muller transform.
func NormFloat64(state *State) float64 {
	for {
		u1 := Float64(state)
		u2 := Float64(state)
		if u1 > 1e-10 {
			// Box-Muller transform
			r := sqrt(-2.0 * log(u1))
			theta := 2.0 * 3.141592653589793 * u2
			return r * cos(theta)
		}
	}
}

// Simple math functions to avoid import
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

func log(x float64) float64 {
	if x <= 0 {
		return -1e308
	}
	// Use Newton's method for ln
	y := (x - 1) / (x + 1)
	y2 := y * y
	sum := y
	term := y
	for i := 1; i < 50; i++ {
		term *= y2
		sum += term / float64(2*i+1)
	}
	return 2 * sum
}

func cos(x float64) float64 {
	// Reduce to [0, 2*pi]
	pi := 3.141592653589793
	for x < 0 {
		x += 2 * pi
	}
	for x > 2*pi {
		x -= 2 * pi
	}
	// Taylor series
	sum := 1.0
	term := 1.0
	x2 := x * x
	for i := 1; i < 20; i++ {
		term *= -x2 / float64((2*i-1)*(2*i))
		sum += term
	}
	return sum
}
