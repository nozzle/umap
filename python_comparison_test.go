package umap_test

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/nozzle/umap"
)

// TestPythonComparison compares the Go UMAP implementation against the
// reference Python umap-learn library on the same input data.
//
// This test requires `uv` to be installed and available in PATH.
// The test will automatically sync the Python environment if needed.
func TestPythonComparison(t *testing.T) {
	// Skip if uv is not available
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv not found in PATH, skipping Python comparison test")
	}

	// Get paths
	testdataDir := "testdata"
	pythonDir := filepath.Join(testdataDir, "python")
	inputFile := filepath.Join(testdataDir, "test_data.csv")

	// Ensure the Python environment is synced
	syncCmd := exec.Command("uv", "sync")
	syncCmd.Dir = pythonDir
	syncCmd.Stdout = os.Stdout
	syncCmd.Stderr = os.Stderr
	if err := syncCmd.Run(); err != nil {
		t.Fatalf("Failed to sync Python environment: %v", err)
	}

	// Create temp file for Python output
	pythonOutput, err := os.CreateTemp("", "python_umap_*.csv")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	pythonOutputPath := pythonOutput.Name()
	pythonOutput.Close()
	defer os.Remove(pythonOutputPath)

	// Test parameters - use same seed and parameters for both implementations
	params := struct {
		neighbors  int
		components int
		metric     string
		minDist    float32
		spread     float32
		epochs     int
		seed       int64
	}{
		neighbors:  5,
		components: 2,
		metric:     "euclidean",
		minDist:    0.1,
		spread:     1.0,
		epochs:     200,
		seed:       42,
	}

	// Get absolute path to input file for Python
	absInputPath, err := filepath.Abs(inputFile)
	if err != nil {
		t.Fatalf("Failed to get absolute path: %v", err)
	}

	// Run Python UMAP
	t.Log("Running Python UMAP...")
	pythonCmd := exec.Command("uv", "run", "python", "run_umap.py",
		"--input", absInputPath,
		"--output", pythonOutputPath,
		"--neighbors", strconv.Itoa(params.neighbors),
		"--components", strconv.Itoa(params.components),
		"--metric", params.metric,
		"--min-dist", fmt.Sprintf("%f", params.minDist),
		"--spread", fmt.Sprintf("%f", params.spread),
		"--epochs", strconv.Itoa(params.epochs),
		"--seed", strconv.FormatInt(params.seed, 10),
	)
	pythonCmd.Dir = pythonDir
	pythonCmd.Stdout = os.Stdout
	pythonCmd.Stderr = os.Stderr
	if err := pythonCmd.Run(); err != nil {
		t.Fatalf("Failed to run Python UMAP: %v", err)
	}

	// Load input data for Go UMAP
	inputData, err := loadCSV(inputFile)
	if err != nil {
		t.Fatalf("Failed to load input data: %v", err)
	}

	// Run Go UMAP
	t.Log("Running Go UMAP...")
	config := umap.DefaultConfig()
	config.NNeighbors = params.neighbors
	config.NComponents = params.components
	config.Metric = params.metric
	config.MinDist = params.minDist
	config.Spread = params.spread
	config.NEpochs = params.epochs
	config.Seed = params.seed
	config.Init = "random" // Use random init for reproducibility

	model := umap.New(config)
	goEmbedding := model.FitTransform(inputData)

	// Load Python embedding
	pythonEmbedding, err := loadCSV(pythonOutputPath)
	if err != nil {
		t.Fatalf("Failed to load Python embedding: %v", err)
	}

	// Compare dimensions
	if len(goEmbedding) != len(pythonEmbedding) {
		t.Errorf("Dimension mismatch: Go has %d points, Python has %d points",
			len(goEmbedding), len(pythonEmbedding))
		return
	}

	if len(goEmbedding) > 0 && len(goEmbedding[0]) != len(pythonEmbedding[0]) {
		t.Errorf("Component mismatch: Go has %d components, Python has %d components",
			len(goEmbedding[0]), len(pythonEmbedding[0]))
		return
	}

	// Log the embeddings for comparison
	t.Log("Go embedding:")
	for i, row := range goEmbedding {
		t.Logf("  [%d]: %v", i, row)
	}

	t.Log("Python embedding:")
	for i, row := range pythonEmbedding {
		t.Logf("  [%d]: %v", i, row)
	}

	// Compare structural similarity using pairwise distances
	// Since UMAP is stochastic and implementations may differ,
	// we compare the relative structure rather than exact values
	goDistances := computePairwiseDistances(goEmbedding)
	pythonDistances := computePairwiseDistances(pythonEmbedding)

	// Compute correlation between pairwise distances
	correlation := computeCorrelation(goDistances, pythonDistances)
	t.Logf("Pairwise distance correlation: %.4f", correlation)

	// The correlation should be reasonably high if both implementations
	// preserve the same relative structure
	if correlation < 0.5 {
		t.Errorf("Low structural similarity between Go and Python embeddings: correlation = %.4f", correlation)
	}

	// Also check that both embeddings have reasonable spread (not collapsed)
	goSpread := computeSpread(goEmbedding)
	pythonSpread := computeSpread(pythonEmbedding)
	t.Logf("Go embedding spread: %.4f", goSpread)
	t.Logf("Python embedding spread: %.4f", pythonSpread)

	if goSpread < 0.01 {
		t.Error("Go embedding appears to be collapsed (very low spread)")
	}
	if pythonSpread < 0.01 {
		t.Error("Python embedding appears to be collapsed (very low spread)")
	}
}

// loadCSV loads data from a CSV file (no header, numeric values only).
func loadCSV(filename string) ([][]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	if len(records) == 0 {
		return nil, fmt.Errorf("empty file")
	}

	data := make([][]float32, len(records))
	for i, record := range records {
		data[i] = make([]float32, len(record))
		for j, val := range record {
			f, err := strconv.ParseFloat(val, 32)
			if err != nil {
				return nil, fmt.Errorf("row %d, col %d: %v", i, j, err)
			}
			data[i][j] = float32(f)
		}
	}

	return data, nil
}

// computePairwiseDistances computes all pairwise Euclidean distances.
func computePairwiseDistances(embedding [][]float32) []float64 {
	n := len(embedding)
	distances := make([]float64, 0, n*(n-1)/2)

	for i := range n {
		for j := i + 1; j < n; j++ {
			dist := euclideanDistance(embedding[i], embedding[j])
			distances = append(distances, dist)
		}
	}

	return distances
}

// euclideanDistance computes the Euclidean distance between two points.
func euclideanDistance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		diff := float64(a[i] - b[i])
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// computeCorrelation computes the Pearson correlation coefficient.
func computeCorrelation(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	n := float64(len(a))

	// Compute means
	var sumA, sumB float64
	for i := range a {
		sumA += a[i]
		sumB += b[i]
	}
	meanA := sumA / n
	meanB := sumB / n

	// Compute correlation
	var num, denomA, denomB float64
	for i := range a {
		diffA := a[i] - meanA
		diffB := b[i] - meanB
		num += diffA * diffB
		denomA += diffA * diffA
		denomB += diffB * diffB
	}

	if denomA == 0 || denomB == 0 {
		return 0
	}

	return num / (math.Sqrt(denomA) * math.Sqrt(denomB))
}

// computeSpread computes the standard deviation of embedding coordinates.
func computeSpread(embedding [][]float32) float64 {
	if len(embedding) == 0 || len(embedding[0]) == 0 {
		return 0
	}

	var sum, sumSq float64
	count := 0

	for _, row := range embedding {
		for _, val := range row {
			v := float64(val)
			sum += v
			sumSq += v * v
			count++
		}
	}

	if count == 0 {
		return 0
	}

	mean := sum / float64(count)
	variance := sumSq/float64(count) - mean*mean
	if variance < 0 {
		variance = 0
	}
	return math.Sqrt(variance)
}
