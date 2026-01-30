// Package math provides float32 math utilities for UMAP.
package math

import "math"

// Clip clamps a value to the range [-4.0, 4.0].
// This is used for gradient clipping in the optimization.
func Clip(val float32) float32 {
	if val > 4.0 {
		return 4.0
	}
	if val < -4.0 {
		return -4.0
	}
	return val
}

// ClipFloat64 clamps a float64 value to the range [-4.0, 4.0].
func ClipFloat64(val float64) float64 {
	if val > 4.0 {
		return 4.0
	}
	if val < -4.0 {
		return -4.0
	}
	return val
}

// Sqrt32 computes the square root of a float32.
func Sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

// Pow32 computes x^y for float32.
func Pow32(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

// Exp32 computes e^x for float32.
func Exp32(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

// Log32 computes natural log of x for float32.
func Log32(x float32) float32 {
	return float32(math.Log(float64(x)))
}

// Log2_32 computes log base 2 of x for float32.
func Log2_32(x float32) float32 {
	return float32(math.Log2(float64(x)))
}

// Abs32 returns the absolute value of a float32.
func Abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

// Max32 returns the maximum of two float32 values.
func Max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// Min32 returns the minimum of two float32 values.
func Min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// MaxInt returns the maximum of two int values.
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// MinInt returns the minimum of two int values.
func MinInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Sign returns -1 for negative, 1 for positive, 0 for zero.
func Sign(x float32) float32 {
	if x < 0 {
		return -1
	}
	if x > 0 {
		return 1
	}
	return 0
}

// Norm computes the L2 norm of a vector.
func Norm(vec []float32) float32 {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	return Sqrt32(sum)
}

// NormSquared computes the squared L2 norm of a vector.
func NormSquared(vec []float32) float32 {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	return sum
}

// Dot computes the dot product of two vectors.
func Dot(a, b []float32) float32 {
	var sum float32
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// Floor32 returns the greatest integer value less than or equal to x.
func Floor32(x float32) float32 {
	return float32(math.Floor(float64(x)))
}

// Ceil32 returns the least integer value greater than or equal to x.
func Ceil32(x float32) float32 {
	return float32(math.Ceil(float64(x)))
}

// Cos32 computes cosine for float32.
func Cos32(x float32) float32 {
	return float32(math.Cos(float64(x)))
}

// Sin32 computes sine for float32.
func Sin32(x float32) float32 {
	return float32(math.Sin(float64(x)))
}

// Arccos32 computes arccosine for float32.
func Arccos32(x float32) float32 {
	return float32(math.Acos(float64(x)))
}

// Arcsin32 computes arcsine for float32.
func Arcsin32(x float32) float32 {
	return float32(math.Asin(float64(x)))
}

// Arccosh32 computes inverse hyperbolic cosine for float32.
func Arccosh32(x float32) float32 {
	return float32(math.Acosh(float64(x)))
}

// Inf32 returns positive infinity as float32.
func Inf32() float32 {
	return float32(math.Inf(1))
}

// IsInf32 checks if a float32 is infinite.
func IsInf32(x float32) bool {
	return math.IsInf(float64(x), 0)
}

// IsNaN32 checks if a float32 is NaN.
func IsNaN32(x float32) bool {
	return math.IsNaN(float64(x))
}
