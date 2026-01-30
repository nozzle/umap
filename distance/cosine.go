package distance

// Cosine computes the cosine distance.
// D(x, y) = 1 - (x . y) / (||x|| * ||y||)
func Cosine(x, y []float32) float32 {
	var dotProduct, normX, normY float32
	for i := range x {
		dotProduct += x[i] * y[i]
		normX += x[i] * x[i]
		normY += y[i] * y[i]
	}
	if normX == 0 || normY == 0 {
		return 1.0
	}
	similarity := dotProduct / (sqrt32(normX) * sqrt32(normY))
	// Clamp to [-1, 1] to handle floating point errors
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}
	return 1.0 - similarity
}

// CosineGrad computes cosine distance and its gradient.
func CosineGrad(x, y []float32) (float32, []float32) {
	var dotProduct, normX, normY float32
	for i := range x {
		dotProduct += x[i] * y[i]
		normX += x[i] * x[i]
		normY += y[i] * y[i]
	}

	grad := make([]float32, len(x))
	if normX == 0 || normY == 0 {
		return 1.0, grad
	}

	magX := sqrt32(normX)
	magY := sqrt32(normY)
	similarity := dotProduct / (magX * magY)

	// Clamp similarity
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}

	// Gradient: d/dx (1 - cos) = -d/dx cos
	// d/dx_i cos(x,y) = y_i/(|x||y|) - x_i * (x.y)/(|x|^3 * |y|)
	for i := range x {
		grad[i] = -(y[i]/(magX*magY) - x[i]*dotProduct/(normX*magX*magY))
	}

	return 1.0 - similarity, grad
}

// AlternativeCosine computes angular distance in radians.
// D(x, y) = arccos((x . y) / (||x|| * ||y||)) / pi
func AlternativeCosine(x, y []float32) float32 {
	var dotProduct, normX, normY float32
	for i := range x {
		dotProduct += x[i] * y[i]
		normX += x[i] * x[i]
		normY += y[i] * y[i]
	}
	if normX == 0 || normY == 0 {
		return 1.0
	}
	similarity := dotProduct / (sqrt32(normX) * sqrt32(normY))
	// Clamp to valid range for arccos
	if similarity > 1.0 {
		similarity = 1.0
	} else if similarity < -1.0 {
		similarity = -1.0
	}
	return acos32(similarity) / 3.14159265358979323846
}

// TrueAngular computes the true angular distance (in radians/pi).
func TrueAngular(x, y []float32) float32 {
	return AlternativeCosine(x, y)
}

// Correlation computes the correlation distance.
// D(x, y) = 1 - correlation(x, y)
// correlation = (x-mean(x)).(y-mean(y)) / (||x-mean(x)|| * ||y-mean(y)||)
func Correlation(x, y []float32) float32 {
	n := float32(len(x))
	if n == 0 {
		return 0
	}

	// Compute means
	var meanX, meanY float32
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= n
	meanY /= n

	// Compute correlation using centered vectors
	var dotProduct, normX, normY float32
	for i := range x {
		dx := x[i] - meanX
		dy := y[i] - meanY
		dotProduct += dx * dy
		normX += dx * dx
		normY += dy * dy
	}

	if normX == 0 || normY == 0 {
		return 1.0
	}

	correlation := dotProduct / (sqrt32(normX) * sqrt32(normY))
	// Clamp to [-1, 1]
	if correlation > 1.0 {
		correlation = 1.0
	} else if correlation < -1.0 {
		correlation = -1.0
	}
	return 1.0 - correlation
}

// CorrelationGrad computes correlation distance and its gradient.
func CorrelationGrad(x, y []float32) (float32, []float32) {
	n := float32(len(x))
	grad := make([]float32, len(x))
	if n == 0 {
		return 0, grad
	}

	// Compute means
	var meanX, meanY float32
	for i := range x {
		meanX += x[i]
		meanY += y[i]
	}
	meanX /= n
	meanY /= n

	// Compute centered values and norms
	var dotProduct, normX, normY float32
	dx := make([]float32, len(x))
	dy := make([]float32, len(x))
	for i := range x {
		dx[i] = x[i] - meanX
		dy[i] = y[i] - meanY
		dotProduct += dx[i] * dy[i]
		normX += dx[i] * dx[i]
		normY += dy[i] * dy[i]
	}

	if normX == 0 || normY == 0 {
		return 1.0, grad
	}

	magX := sqrt32(normX)
	magY := sqrt32(normY)
	correlation := dotProduct / (magX * magY)

	// Clamp
	if correlation > 1.0 {
		correlation = 1.0
	} else if correlation < -1.0 {
		correlation = -1.0
	}

	// Gradient of correlation distance
	// Similar to cosine but with centered vectors
	for i := range x {
		grad[i] = -(dy[i]/(magX*magY) - dx[i]*dotProduct/(normX*magX*magY))
	}

	return 1.0 - correlation, grad
}

// Helper for arccos
func acos32(x float32) float32 {
	// Use identity: acos(x) = pi/2 - asin(x)
	return 1.5707963267948966 - asin32(x)
}
