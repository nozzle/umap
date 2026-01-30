// Package parallel provides parallel execution helpers.
package parallel

import (
	"runtime"
	"sync"
)

// NumWorkers returns the default number of workers for parallel operations.
func NumWorkers() int {
	return runtime.GOMAXPROCS(0)
}

// ParallelFor executes fn for indices [start, end) using n workers.
func ParallelFor(start, end, n int, fn func(i int)) {
	if n <= 1 {
		for i := start; i < end; i++ {
			fn(i)
		}
		return
	}

	total := end - start
	if total <= 0 {
		return
	}

	var wg sync.WaitGroup
	chunkSize := (total + n - 1) / n

	for w := 0; w < n; w++ {
		chunkStart := start + w*chunkSize
		chunkEnd := chunkStart + chunkSize
		if chunkEnd > end {
			chunkEnd = end
		}
		if chunkStart >= chunkEnd {
			break
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				fn(i)
			}
		}(chunkStart, chunkEnd)
	}

	wg.Wait()
}

// ParallelForChunked executes fn for chunks of indices.
// fn receives (chunkStart, chunkEnd) for each chunk.
func ParallelForChunked(start, end, chunkSize, n int, fn func(chunkStart, chunkEnd int)) {
	if n <= 1 {
		for s := start; s < end; s += chunkSize {
			e := s + chunkSize
			if e > end {
				e = end
			}
			fn(s, e)
		}
		return
	}

	var wg sync.WaitGroup
	chunks := make(chan [2]int, n)

	// Start workers
	for w := 0; w < n; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for chunk := range chunks {
				fn(chunk[0], chunk[1])
			}
		}()
	}

	// Send chunks
	for s := start; s < end; s += chunkSize {
		e := s + chunkSize
		if e > end {
			e = end
		}
		chunks <- [2]int{s, e}
	}
	close(chunks)

	wg.Wait()
}

// ParallelMap applies fn to each index and collects results.
func ParallelMap[T any](start, end, n int, fn func(i int) T) []T {
	results := make([]T, end-start)

	if n <= 1 {
		for i := start; i < end; i++ {
			results[i-start] = fn(i)
		}
		return results
	}

	var wg sync.WaitGroup
	chunkSize := (end - start + n - 1) / n

	for w := 0; w < n; w++ {
		chunkStart := start + w*chunkSize
		chunkEnd := chunkStart + chunkSize
		if chunkEnd > end {
			chunkEnd = end
		}
		if chunkStart >= chunkEnd {
			break
		}

		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				results[i-start] = fn(i)
			}
		}(chunkStart, chunkEnd)
	}

	wg.Wait()
	return results
}

// Do executes multiple functions in parallel.
func Do(fns ...func()) {
	if len(fns) == 0 {
		return
	}
	if len(fns) == 1 {
		fns[0]()
		return
	}

	var wg sync.WaitGroup
	wg.Add(len(fns))
	for _, fn := range fns {
		go func(f func()) {
			defer wg.Done()
			f()
		}(fn)
	}
	wg.Wait()
}
