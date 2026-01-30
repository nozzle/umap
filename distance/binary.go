package distance

// Binary distance metrics for binary/boolean vectors.
// These treat non-zero values as true (1) and zero values as false (0).

// countBinary counts true-true, true-false, false-true, false-false pairs.
func countBinary(x, y []float32) (ntt, ntf, nft, nff int) {
	for i := range x {
		xTrue := x[i] != 0
		yTrue := y[i] != 0
		switch {
		case xTrue && yTrue:
			ntt++
		case xTrue && !yTrue:
			ntf++
		case !xTrue && yTrue:
			nft++
		default:
			nff++
		}
	}
	return
}

// Hamming computes the Hamming distance (proportion of disagreeing components).
// D(x, y) = (number of x_i != y_i) / n
func Hamming(x, y []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	var count int
	for i := range x {
		if (x[i] != 0) != (y[i] != 0) {
			count++
		}
	}
	return float32(count) / float32(len(x))
}

// Jaccard computes the Jaccard distance.
// D(x, y) = 1 - |intersection| / |union|
// D(x, y) = (ntf + nft) / (ntt + ntf + nft)
func Jaccard(x, y []float32) float32 {
	ntt, ntf, nft, _ := countBinary(x, y)
	denom := ntt + ntf + nft
	if denom == 0 {
		return 0
	}
	return float32(ntf+nft) / float32(denom)
}

// Dice computes the Dice distance (Sørensen-Dice).
// D(x, y) = (ntf + nft) / (2*ntt + ntf + nft)
func Dice(x, y []float32) float32 {
	ntt, ntf, nft, _ := countBinary(x, y)
	denom := 2*ntt + ntf + nft
	if denom == 0 {
		return 0
	}
	return float32(ntf+nft) / float32(denom)
}

// Matching computes the matching distance (same as Hamming for binary).
// D(x, y) = (ntf + nft) / n
func Matching(x, y []float32) float32 {
	return Hamming(x, y)
}

// Kulsinski computes the Kulsinski distance.
// D(x, y) = (ntf + nft - ntt + n) / (ntf + nft + n)
func Kulsinski(x, y []float32) float32 {
	ntt, ntf, nft, nff := countBinary(x, y)
	n := ntt + ntf + nft + nff
	denom := ntf + nft + n
	if denom == 0 {
		return 0
	}
	return float32(ntf+nft-ntt+n) / float32(denom)
}

// RogersTanimoto computes the Rogers-Tanimoto distance.
// D(x, y) = 2*(ntf + nft) / (n + ntf + nft)
func RogersTanimoto(x, y []float32) float32 {
	ntt, ntf, nft, nff := countBinary(x, y)
	n := ntt + ntf + nft + nff
	denom := n + ntf + nft
	if denom == 0 {
		return 0
	}
	return float32(2*(ntf+nft)) / float32(denom)
}

// RussellRao computes the Russell-Rao distance.
// D(x, y) = (n - ntt) / n
func RussellRao(x, y []float32) float32 {
	ntt, ntf, nft, nff := countBinary(x, y)
	n := ntt + ntf + nft + nff
	if n == 0 {
		return 0
	}
	return float32(n-ntt) / float32(n)
}

// SokalMichener computes the Sokal-Michener distance.
// D(x, y) = 2*(ntf + nft) / (n + ntf + nft)
// (Same as Rogers-Tanimoto)
func SokalMichener(x, y []float32) float32 {
	return RogersTanimoto(x, y)
}

// SokalSneath computes the Sokal-Sneath distance.
// D(x, y) = 2*(ntf + nft) / (ntt + 2*(ntf + nft))
func SokalSneath(x, y []float32) float32 {
	ntt, ntf, nft, _ := countBinary(x, y)
	denom := ntt + 2*(ntf+nft)
	if denom == 0 {
		return 0
	}
	return float32(2*(ntf+nft)) / float32(denom)
}

// Yule computes the Yule distance.
// D(x, y) = 2*ntf*nft / (ntt*nff + ntf*nft)
func Yule(x, y []float32) float32 {
	ntt, ntf, nft, nff := countBinary(x, y)
	denom := ntt*nff + ntf*nft
	if denom == 0 {
		return 0
	}
	return float32(2*ntf*nft) / float32(denom)
}

// Categorical/ordinal metrics

// SimpleCategorical computes categorical distance.
// D(x, y) = 1 if x != y, 0 otherwise (for single values)
// For vectors: proportion of differing elements
func SimpleCategorical(x, y []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	var diff int
	for i := range x {
		if x[i] != y[i] {
			diff++
		}
	}
	return float32(diff) / float32(len(x))
}

// Ordinal computes ordinal distance (normalized absolute difference).
func Ordinal(x, y []float32) float32 {
	if len(x) == 0 {
		return 0
	}
	var sum float32
	for i := range x {
		sum += abs32(x[i] - y[i])
	}
	return sum / float32(len(x))
}
