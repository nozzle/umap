// Command umap provides a CLI for running UMAP on data files.
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"

	"github.com/nozzle/umap"
)

func main() {
	// Parse command-line flags
	inputFile := flag.String("input", "", "Input CSV file (required)")
	outputFile := flag.String("output", "embedding.csv", "Output CSV file")
	nNeighbors := flag.Int("neighbors", 15, "Number of neighbors for k-NN")
	nComponents := flag.Int("components", 2, "Number of output dimensions")
	metric := flag.String("metric", "euclidean", "Distance metric")
	minDist := flag.Float64("min-dist", 0.1, "Minimum distance between points")
	spread := flag.Float64("spread", 1.0, "Spread of embedded points")
	nEpochs := flag.Int("epochs", 200, "Number of training epochs")
	seed := flag.Int64("seed", 42, "Random seed")
	verbose := flag.Bool("verbose", false, "Verbose output")
	flag.Parse()

	if *inputFile == "" {
		fmt.Fprintln(os.Stderr, "Error: -input flag is required")
		flag.Usage()
		os.Exit(1)
	}

	// Load data
	data, err := loadCSV(*inputFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading data: %v\n", err)
		os.Exit(1)
	}

	if *verbose {
		fmt.Printf("Loaded %d samples with %d features\n", len(data), len(data[0]))
	}

	// Configure UMAP
	config := umap.DefaultConfig()
	config.NNeighbors = *nNeighbors
	config.NComponents = *nComponents
	config.Metric = *metric
	config.MinDist = float32(*minDist)
	config.Spread = float32(*spread)
	config.NEpochs = *nEpochs
	config.Seed = *seed
	config.Verbose = *verbose

	if *verbose {
		config.ProgressCallback = func(epoch, total int) {
			if epoch%10 == 0 || epoch == total {
				fmt.Printf("Epoch %d/%d\n", epoch, total)
			}
		}
	}

	// Run UMAP
	model := umap.New(config)
	embedding := model.FitTransform(data)

	// Save output
	if err := saveCSV(*outputFile, embedding); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving output: %v\n", err)
		os.Exit(1)
	}

	if *verbose {
		fmt.Printf("Saved embedding to %s\n", *outputFile)
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

// saveCSV saves an embedding to a CSV file.
func saveCSV(filename string, embedding [][]float32) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, row := range embedding {
		record := make([]string, len(row))
		for j, val := range row {
			record[j] = strconv.FormatFloat(float64(val), 'f', 6, 32)
		}
		if err := writer.Write(record); err != nil {
			return err
		}
	}

	return nil
}
