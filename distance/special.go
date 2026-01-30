package distance

import "math"

// Canberra computes the Canberra distance.
// D(x, y) = sum(|x_i - y_i| / (|x_i| + |y_i|))
func Canberra(x, y []float32) float32 {
	var sum float32
	for i := range x {
		num := abs32(x[i] - y[i])
		denom := abs32(x[i]) + abs32(y[i])
		if denom > 0 {
			sum += num / denom
		}
	}
	return sum
}

// CanberraGrad computes Canberra distance and its gradient.
func CanberraGrad(x, y []float32) (float32, []float32) {
	var sum float32
	grad := make([]float32, len(x))
	for i := range x {
		num := abs32(x[i] - y[i])
		absX := abs32(x[i])
		absY := abs32(y[i])
		denom := absX + absY
		if denom > 0 {
			sum += num / denom
			// Gradient: d/dx_i = sign(x_i - y_i)/denom - |x_i - y_i| * sign(x_i) / denom^2
			signDiff := sign32(x[i] - y[i])
			signX := sign32(x[i])
			grad[i] = signDiff/denom - num*signX/(denom*denom)
		}
	}
	return sum, grad
}

// BrayCurtis computes the Bray-Curtis distance.
// D(x, y) = sum(|x_i - y_i|) / sum(|x_i + y_i|)
func BrayCurtis(x, y []float32) float32 {
	var numerator, denominator float32
	for i := range x {
		numerator += abs32(x[i] - y[i])
		denominator += abs32(x[i] + y[i])
	}
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

// BrayCurtisGrad computes Bray-Curtis distance and its gradient.
func BrayCurtisGrad(x, y []float32) (float32, []float32) {
	var numerator, denominator float32
	for i := range x {
		numerator += abs32(x[i] - y[i])
		denominator += abs32(x[i] + y[i])
	}
	grad := make([]float32, len(x))
	if denominator == 0 {
		return 0, grad
	}
	dist := numerator / denominator
	// Gradient: d/dx_i = sign(x_i - y_i)/denom - (num/denom^2) * sign(x_i + y_i)
	for i := range x {
		signDiff := sign32(x[i] - y[i])
		signSum := sign32(x[i] + y[i])
		grad[i] = signDiff/denominator - dist*signSum/denominator
	}
	return dist, grad
}

// Haversine computes the Haversine distance for lat/lon coordinates.
// Assumes x and y are [lat, lon] in radians.
// D(x, y) = 2 * arcsin(sqrt(sin^2((lat1-lat2)/2) + cos(lat1)*cos(lat2)*sin^2((lon1-lon2)/2)))
func Haversine(x, y []float32) float32 {
	if len(x) != 2 || len(y) != 2 {
		return 0
	}
	sinLat := sin32((x[0] - y[0]) / 2)
	sinLon := sin32((x[1] - y[1]) / 2)
	a := sinLat*sinLat + cos32(x[0])*cos32(y[0])*sinLon*sinLon
	if a < 0 {
		a = 0
	}
	if a > 1 {
		a = 1
	}
	return 2 * asin32(sqrt32(a))
}

// HaversineGrad computes Haversine distance and its gradient.
func HaversineGrad(x, y []float32) (float32, []float32) {
	grad := make([]float32, len(x))
	if len(x) != 2 || len(y) != 2 {
		return 0, grad
	}

	sinLatHalf := sin32((x[0] - y[0]) / 2)
	cosLatHalf := cos32((x[0] - y[0]) / 2)
	sinLonHalf := sin32((x[1] - y[1]) / 2)
	cosLonHalf := cos32((x[1] - y[1]) / 2)

	cosLat1 := cos32(x[0])
	cosLat2 := cos32(y[0])
	sinLat1 := sin32(x[0])

	a := sinLatHalf*sinLatHalf + cosLat1*cosLat2*sinLonHalf*sinLonHalf
	if a < 0 {
		a = 0
	}
	if a > 1 {
		a = 1
	}

	sqrtA := sqrt32(a)
	dist := 2 * asin32(sqrtA)

	if sqrtA > 1e-6 && a < 1-1e-6 {
		// d(dist)/da = 1 / sqrt(a * (1 - a))
		dadLat := sinLatHalf*cosLatHalf - sinLat1*cosLat2*sinLonHalf*sinLonHalf
		dadLon := cosLat1 * cosLat2 * sinLonHalf * cosLonHalf

		factor := 1.0 / sqrt32(a*(1-a))
		grad[0] = factor * dadLat
		grad[1] = factor * dadLon
	}

	return dist, grad
}

// Hellinger computes the Hellinger distance.
// D(x, y) = sqrt(sum((sqrt(x_i) - sqrt(y_i))^2)) / sqrt(2)
// Assumes non-negative values (probability distributions).
func Hellinger(x, y []float32) float32 {
	var sum float32
	for i := range x {
		xi := x[i]
		yi := y[i]
		if xi < 0 {
			xi = 0
		}
		if yi < 0 {
			yi = 0
		}
		d := sqrt32(xi) - sqrt32(yi)
		sum += d * d
	}
	return sqrt32(sum) / 1.4142135623730951 // sqrt(2)
}

// HellingerGrad computes Hellinger distance and its gradient.
func HellingerGrad(x, y []float32) (float32, []float32) {
	var sum float32
	diffs := make([]float32, len(x))
	sqrtX := make([]float32, len(x))

	for i := range x {
		xi := x[i]
		yi := y[i]
		if xi < 0 {
			xi = 0
		}
		if yi < 0 {
			yi = 0
		}
		sqrtX[i] = sqrt32(xi)
		sqrtY := sqrt32(yi)
		diffs[i] = sqrtX[i] - sqrtY
		sum += diffs[i] * diffs[i]
	}

	dist := sqrt32(sum) / 1.4142135623730951
	grad := make([]float32, len(x))

	if sum > 1e-12 {
		sqrtSum := sqrt32(sum)
		for i := range x {
			if x[i] > 1e-12 {
				// d/dx_i = (sqrt(x_i) - sqrt(y_i)) / (sqrt(sum) * sqrt(2) * sqrt(x_i))
				grad[i] = diffs[i] / (sqrtSum * 1.4142135623730951 * sqrtX[i] * 2)
			}
		}
	}

	return dist, grad
}

// Mahalanobis computes the Mahalanobis distance.
// D(x, y) = sqrt((x-y)^T * V^{-1} * (x-y))
// where vinv is the flattened inverse covariance matrix.
func Mahalanobis(x, y []float32, vinv []float32) float32 {
	n := len(x)
	diff := make([]float32, n)
	for i := range x {
		diff[i] = x[i] - y[i]
	}

	var sum float32
	for i := 0; i < n; i++ {
		var inner float32
		for j := 0; j < n; j++ {
			inner += vinv[i*n+j] * diff[j]
		}
		sum += diff[i] * inner
	}

	if sum < 0 {
		sum = 0
	}
	return sqrt32(sum)
}

// SEuclidean computes standardized Euclidean distance.
// D(x, y) = sqrt(sum((x_i - y_i)^2 / v_i))
// where v is the variance vector.
func SEuclidean(x, y, v []float32) float32 {
	var sum float32
	for i := range x {
		d := x[i] - y[i]
		if v[i] > 0 {
			sum += (d * d) / v[i]
		}
	}
	return sqrt32(sum)
}

// WMinkowski computes weighted Minkowski distance.
// D(x, y) = (sum(w_i * |x_i - y_i|^p))^(1/p)
func WMinkowski(x, y, w []float32, p float32) float32 {
	var sum float32
	for i := range x {
		sum += w[i] * pow32(abs32(x[i]-y[i]), p)
	}
	return pow32(sum, 1.0/p)
}

// SphericalArcDistance computes arc distance on a unit sphere.
func SphericalArcDistance(x, y []float32) float32 {
	// Normalize vectors
	var normX, normY float32
	for i := range x {
		normX += x[i] * x[i]
		normY += y[i] * y[i]
	}
	normX = sqrt32(normX)
	normY = sqrt32(normY)

	if normX == 0 || normY == 0 {
		return 0
	}

	// Compute dot product of normalized vectors
	var dot float32
	for i := range x {
		dot += (x[i] / normX) * (y[i] / normY)
	}

	// Clamp to [-1, 1]
	if dot > 1 {
		dot = 1
	}
	if dot < -1 {
		dot = -1
	}

	return float32(math.Acos(float64(dot)))
}
