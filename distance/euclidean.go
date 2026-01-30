package distance

// Euclidean computes the standard Euclidean (L2) distance.
// D(x, y) = sqrt(sum((x_i - y_i)^2))
func Euclidean(x, y []float32) float32 {
	var sum float32
	for i := range x {
		d := x[i] - y[i]
		sum += d * d
	}
	return sqrt32(sum)
}

// EuclideanGrad computes Euclidean distance and its gradient.
func EuclideanGrad(x, y []float32) (float32, []float32) {
	var sum float32
	for i := range x {
		d := x[i] - y[i]
		sum += d * d
	}
	dist := sqrt32(sum)
	grad := make([]float32, len(x))
	if dist > 1e-6 {
		for i := range x {
			grad[i] = (x[i] - y[i]) / dist
		}
	}
	return dist, grad
}

// SquaredEuclidean computes the squared Euclidean distance (faster, no sqrt).
// D(x, y) = sum((x_i - y_i)^2)
func SquaredEuclidean(x, y []float32) float32 {
	var sum float32
	for i := range x {
		d := x[i] - y[i]
		sum += d * d
	}
	return sum
}

// ReducedEuclidean is an alias for SquaredEuclidean (used in optimization).
func ReducedEuclidean(x, y []float32) float32 {
	return SquaredEuclidean(x, y)
}

// Manhattan computes the Manhattan (L1/taxicab) distance.
// D(x, y) = sum(|x_i - y_i|)
func Manhattan(x, y []float32) float32 {
	var sum float32
	for i := range x {
		sum += abs32(x[i] - y[i])
	}
	return sum
}

// ManhattanGrad computes Manhattan distance and its gradient.
func ManhattanGrad(x, y []float32) (float32, []float32) {
	var sum float32
	grad := make([]float32, len(x))
	for i := range x {
		d := x[i] - y[i]
		sum += abs32(d)
		grad[i] = sign32(d)
	}
	return sum, grad
}

// Chebyshev computes the Chebyshev (L-infinity) distance.
// D(x, y) = max(|x_i - y_i|)
func Chebyshev(x, y []float32) float32 {
	var maxVal float32
	for i := range x {
		d := abs32(x[i] - y[i])
		if d > maxVal {
			maxVal = d
		}
	}
	return maxVal
}

// ChebyshevGrad computes Chebyshev distance and its gradient.
func ChebyshevGrad(x, y []float32) (float32, []float32) {
	var maxVal float32
	maxIdx := 0
	for i := range x {
		d := abs32(x[i] - y[i])
		if d > maxVal {
			maxVal = d
			maxIdx = i
		}
	}
	grad := make([]float32, len(x))
	grad[maxIdx] = sign32(x[maxIdx] - y[maxIdx])
	return maxVal, grad
}

// Minkowski computes the Minkowski distance with given p.
// D(x, y) = (sum(|x_i - y_i|^p))^(1/p)
func Minkowski(x, y []float32, p float32) float32 {
	var sum float32
	for i := range x {
		sum += pow32(abs32(x[i]-y[i]), p)
	}
	return pow32(sum, 1.0/p)
}

// Minkowski2 is Minkowski with p=2 (equivalent to Euclidean).
func Minkowski2(x, y []float32) float32 {
	return Euclidean(x, y)
}

// MinkowskiGrad computes Minkowski distance and its gradient.
func MinkowskiGrad(x, y []float32, p float32) (float32, []float32) {
	var sum float32
	for i := range x {
		sum += pow32(abs32(x[i]-y[i]), p)
	}
	dist := pow32(sum, 1.0/p)
	grad := make([]float32, len(x))
	for i := range x {
		grad[i] = pow32(abs32(x[i]-y[i]), p-1.0) * sign32(x[i]-y[i]) * pow32(sum, 1.0/(p-1.0))
	}
	return dist, grad
}

// StandardizedEuclidean computes Euclidean distance standardized by std dev.
// D(x, y) = sqrt(sum((x_i - y_i)^2 / sigma_i))
func StandardizedEuclidean(x, y, sigma []float32) float32 {
	var sum float32
	for i := range x {
		d := x[i] - y[i]
		sum += (d * d) / sigma[i]
	}
	return sqrt32(sum)
}

// StandardizedEuclideanGrad computes standardized Euclidean and gradient.
func StandardizedEuclideanGrad(x, y, sigma []float32) (float32, []float32) {
	var sum float32
	for i := range x {
		d := x[i] - y[i]
		sum += (d * d) / sigma[i]
	}
	dist := sqrt32(sum)
	grad := make([]float32, len(x))
	if dist > 1e-6 {
		for i := range x {
			grad[i] = (x[i] - y[i]) / (dist * sigma[i])
		}
	}
	return dist, grad
}

// WeightedMinkowski computes weighted Minkowski distance.
// D(x, y) = (sum(w_i * |x_i - y_i|^p))^(1/p)
func WeightedMinkowski(x, y, w []float32, p float32) float32 {
	var sum float32
	for i := range x {
		sum += w[i] * pow32(abs32(x[i]-y[i]), p)
	}
	return pow32(sum, 1.0/p)
}

// Poincare computes Poincare distance in the Poincare ball model.
func Poincare(x, y []float32) float32 {
	var sqUNorm, sqVNorm, sqDist float32
	for i := range x {
		sqUNorm += x[i] * x[i]
		sqVNorm += y[i] * y[i]
		d := x[i] - y[i]
		sqDist += d * d
	}
	delta := 2.0 * sqDist / ((1.0 - sqUNorm) * (1.0 - sqVNorm))
	return acosh32(1.0 + delta)
}
