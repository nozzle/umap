// Package distance provides distance metrics for UMAP.
package distance

import (
	"math"
)

// Func is a distance function between two vectors.
type Func func(x, y []float32) float32

// GradFunc computes distance and gradient with respect to x.
type GradFunc func(x, y []float32) (dist float32, grad []float32)

// SparseFunc computes distance between two sparse vectors.
type SparseFunc func(ind1 []int32, data1 []float32, ind2 []int32, data2 []float32) float32

// Metric represents a named distance metric.
type Metric struct {
	Name         string
	Func         Func
	GradFunc     GradFunc
	SparseFunc   SparseFunc
	RequiresGrad bool
	Angular      bool // True for angular/cosine-based metrics
}

// Registry maps metric names to their implementations.
var Registry = map[string]Func{
	// Minkowski family
	"euclidean":   Euclidean,
	"l2":          Euclidean,
	"sqeuclidean": SquaredEuclidean,
	"manhattan":   Manhattan,
	"l1":          Manhattan,
	"taxicab":     Manhattan,
	"chebyshev":   Chebyshev,
	"linfinity":   Chebyshev,
	"linf":        Chebyshev,
	"minkowski":   Minkowski2, // Default p=2

	// Angular metrics
	"cosine":      Cosine,
	"correlation": Correlation,

	// Other metrics
	"canberra":   Canberra,
	"braycurtis": BrayCurtis,
	"haversine":  Haversine,
	"hellinger":  Hellinger,

	// Binary metrics
	"hamming":        Hamming,
	"jaccard":        Jaccard,
	"dice":           Dice,
	"matching":       Matching,
	"kulsinski":      Kulsinski,
	"rogerstanimoto": RogersTanimoto,
	"russellrao":     RussellRao,
	"sokalmichener":  SokalMichener,
	"sokalsneath":    SokalSneath,
	"yule":           Yule,
}

// GradRegistry maps metric names to gradient-enabled implementations.
var GradRegistry = map[string]GradFunc{
	"euclidean":   EuclideanGrad,
	"l2":          EuclideanGrad,
	"manhattan":   ManhattanGrad,
	"l1":          ManhattanGrad,
	"chebyshev":   ChebyshevGrad,
	"cosine":      CosineGrad,
	"correlation": CorrelationGrad,
	"canberra":    CanberraGrad,
	"braycurtis":  BrayCurtisGrad,
	"haversine":   HaversineGrad,
	"hellinger":   HellingerGrad,
}

// AngularMetrics are metrics where angular RP-trees work better.
var AngularMetrics = map[string]bool{
	"cosine":      true,
	"correlation": true,
}

// Get returns the distance function for the given metric name.
func Get(name string) (Func, bool) {
	f, ok := Registry[name]
	return f, ok
}

// GetGrad returns the gradient function for the given metric name.
func GetGrad(name string) (GradFunc, bool) {
	f, ok := GradRegistry[name]
	return f, ok
}

// IsAngular returns true if the metric is angular/cosine-based.
func IsAngular(name string) bool {
	return AngularMetrics[name]
}

// Helper functions
func sqrt32(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func pow32(x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func exp32(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func log32(x float32) float32 {
	return float32(math.Log(float64(x)))
}

func sign32(x float32) float32 {
	if x < 0 {
		return -1
	}
	if x > 0 {
		return 1
	}
	return 0
}

func sin32(x float32) float32 {
	return float32(math.Sin(float64(x)))
}

func cos32(x float32) float32 {
	return float32(math.Cos(float64(x)))
}

func asin32(x float32) float32 {
	return float32(math.Asin(float64(x)))
}

func acosh32(x float32) float32 {
	return float32(math.Acosh(float64(x)))
}
