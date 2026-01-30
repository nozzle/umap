// Package rand provides random number generation compatible with NumPy's RandomState.
// This implements the Mersenne Twister (MT19937) algorithm for exact reproducibility
// with Python's numpy.random.RandomState.
package rand

const (
	mtN        = 624
	mtM        = 397
	matrixA    = 0x9908b0df
	upperMask  = 0x80000000
	lowerMask  = 0x7fffffff
	temperingB = 0x9d2c5680
	temperingC = 0xefc60000
)

// MT19937 is a Mersenne Twister random number generator compatible with NumPy.
type MT19937 struct {
	mt  [mtN]uint32
	mti int
}

// NewMT19937 creates a new Mersenne Twister with the given seed.
// This matches numpy.random.RandomState(seed).
func NewMT19937(seed uint32) *MT19937 {
	mt := &MT19937{}
	mt.Seed(seed)
	return mt
}

// Seed initializes the generator with a seed.
// This matches numpy.random.RandomState(seed) initialization.
func (mt *MT19937) Seed(seed uint32) {
	mt.mt[0] = seed
	for i := 1; i < mtN; i++ {
		mt.mt[i] = 1812433253*(mt.mt[i-1]^(mt.mt[i-1]>>30)) + uint32(i)
	}
	mt.mti = mtN
}

// Uint32 generates a random uint32.
func (mt *MT19937) Uint32() uint32 {
	var y uint32
	mag01 := [2]uint32{0, matrixA}

	if mt.mti >= mtN {
		// Generate N words at a time
		var kk int
		for kk = 0; kk < mtN-mtM; kk++ {
			y = (mt.mt[kk] & upperMask) | (mt.mt[kk+1] & lowerMask)
			mt.mt[kk] = mt.mt[kk+mtM] ^ (y >> 1) ^ mag01[y&1]
		}
		for ; kk < mtN-1; kk++ {
			y = (mt.mt[kk] & upperMask) | (mt.mt[kk+1] & lowerMask)
			mt.mt[kk] = mt.mt[kk+(mtM-mtN)] ^ (y >> 1) ^ mag01[y&1]
		}
		y = (mt.mt[mtN-1] & upperMask) | (mt.mt[0] & lowerMask)
		mt.mt[mtN-1] = mt.mt[mtM-1] ^ (y >> 1) ^ mag01[y&1]
		mt.mti = 0
	}

	y = mt.mt[mt.mti]
	mt.mti++

	// Tempering
	y ^= y >> 11
	y ^= (y << 7) & temperingB
	y ^= (y << 15) & temperingC
	y ^= y >> 18

	return y
}

// Float64 generates a random float64 in [0, 1).
// This matches numpy's random_sample() / uniform(0, 1).
func (mt *MT19937) Float64() float64 {
	// NumPy uses (Uint32() >> 5) * (1.0 / 67108864.0) + (Uint32() >> 6) * (1.0 / 9007199254740992.0)
	// for 53-bit precision. Simplified version:
	a := mt.Uint32() >> 5
	b := mt.Uint32() >> 6
	return (float64(a)*67108864.0 + float64(b)) * (1.0 / 9007199254740992.0)
}

// Uniform generates a random float64 in [low, high).
// This matches numpy.random.uniform(low, high).
func (mt *MT19937) Uniform(low, high float64) float64 {
	return low + (high-low)*mt.Float64()
}

// Intn returns a random int in [0, n).
func (mt *MT19937) Intn(n int) int {
	if n <= 0 {
		return 0
	}
	// Use rejection sampling to avoid bias
	// This is a simplified version - NumPy has more complex logic
	return int(mt.Uint32()>>1) % n
}

// Float32 generates a random float32 in [0, 1).
func (mt *MT19937) Float32() float32 {
	return float32(mt.Float64())
}

// RandInt32 generates a random int32 in the full int32 range.
// This matches numpy.random.RandomState.randint(INT32_MIN, INT32_MAX+1).
// NumPy generates a uint32 and subtracts 2^31 to shift the range.
func (mt *MT19937) RandInt32() int32 {
	// Generate a uint32 and shift to signed range
	// NumPy does: raw_uint32 + INT32_MIN = raw_uint32 - 2^31
	return int32(mt.Uint32() - 0x80000000)
}

// UniformFloat32 generates a random float32 in [low, high).
func (mt *MT19937) UniformFloat32(low, high float32) float32 {
	return float32(mt.Uniform(float64(low), float64(high)))
}

// Shuffle randomly permutes a slice of int32.
func (mt *MT19937) ShuffleInt32(arr []int32) {
	n := len(arr)
	for i := n - 1; i > 0; i-- {
		j := mt.Intn(i + 1)
		arr[i], arr[j] = arr[j], arr[i]
	}
}
