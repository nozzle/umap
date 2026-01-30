// Package heap provides max-heap implementations for k-NN tracking.
package heap

// MaxHeap is a max-heap for tracking k-nearest neighbors.
// The largest distance is always at the root (index 0).
type MaxHeap struct {
	Indices   []int32
	Distances []float32
	Flags     []uint8 // 0 = old, 1 = new (for NNDescent)
	Size      int
	K         int
}

// New creates a new max-heap with capacity k.
func New(k int) *MaxHeap {
	h := &MaxHeap{
		Indices:   make([]int32, k),
		Distances: make([]float32, k),
		Flags:     make([]uint8, k),
		Size:      0,
		K:         k,
	}
	// Initialize with sentinel values
	for i := 0; i < k; i++ {
		h.Indices[i] = -1
		h.Distances[i] = 1e30 // Large value
		h.Flags[i] = 0
	}
	return h
}

// NewFromSlices creates a heap from existing slices.
func NewFromSlices(indices []int32, distances []float32, flags []uint8) *MaxHeap {
	k := len(indices)
	h := &MaxHeap{
		Indices:   indices,
		Distances: distances,
		Flags:     flags,
		Size:      k,
		K:         k,
	}
	// Heapify
	for i := k/2 - 1; i >= 0; i-- {
		h.siftDown(i)
	}
	return h
}

// MaxDist returns the maximum distance in the heap (root).
func (h *MaxHeap) MaxDist() float32 {
	if h.Size == 0 {
		return 1e30
	}
	return h.Distances[0]
}

// Push attempts to add a new neighbor to the heap.
// Returns true if the neighbor was added (was closer than max).
func (h *MaxHeap) Push(idx int32, dist float32, flag uint8) bool {
	// Quick reject if worse than current worst
	if dist >= h.Distances[0] {
		return false
	}

	// Check for duplicates
	for i := 0; i < h.K; i++ {
		if h.Indices[i] == idx {
			return false
		}
	}

	// Replace root and sift down
	h.Distances[0] = dist
	h.Indices[0] = idx
	h.Flags[0] = flag
	h.siftDown(0)

	if h.Size < h.K {
		h.Size++
	}

	return true
}

// PushWithoutDuplicateCheck adds a neighbor without checking duplicates.
// Use when you know there are no duplicates.
func (h *MaxHeap) PushWithoutDuplicateCheck(idx int32, dist float32, flag uint8) bool {
	if dist >= h.Distances[0] {
		return false
	}

	h.Distances[0] = dist
	h.Indices[0] = idx
	h.Flags[0] = flag
	h.siftDown(0)

	if h.Size < h.K {
		h.Size++
	}

	return true
}

// siftDown restores heap property after replacing root.
func (h *MaxHeap) siftDown(i int) {
	for {
		left := 2*i + 1
		right := 2*i + 2

		if left >= h.K {
			break
		}

		swap := i
		if h.Distances[left] > h.Distances[swap] {
			swap = left
		}
		if right < h.K && h.Distances[right] > h.Distances[swap] {
			swap = right
		}

		if swap == i {
			break
		}

		// Swap all arrays
		h.Distances[i], h.Distances[swap] = h.Distances[swap], h.Distances[i]
		h.Indices[i], h.Indices[swap] = h.Indices[swap], h.Indices[i]
		h.Flags[i], h.Flags[swap] = h.Flags[swap], h.Flags[i]
		i = swap
	}
}

// Sort converts the heap to sorted order (ascending by distance).
// After sorting, the heap property is no longer maintained.
func (h *MaxHeap) Sort() {
	// Heap sort
	for i := h.K - 1; i > 0; i-- {
		// Swap root with last
		h.Distances[0], h.Distances[i] = h.Distances[i], h.Distances[0]
		h.Indices[0], h.Indices[i] = h.Indices[i], h.Indices[0]
		h.Flags[0], h.Flags[i] = h.Flags[i], h.Flags[0]

		// Sift down on reduced heap
		h.siftDownN(0, i)
	}
}

// siftDownN is siftDown limited to first n elements.
func (h *MaxHeap) siftDownN(i, n int) {
	for {
		left := 2*i + 1
		right := 2*i + 2

		if left >= n {
			break
		}

		swap := i
		if h.Distances[left] > h.Distances[swap] {
			swap = left
		}
		if right < n && h.Distances[right] > h.Distances[swap] {
			swap = right
		}

		if swap == i {
			break
		}

		h.Distances[i], h.Distances[swap] = h.Distances[swap], h.Distances[i]
		h.Indices[i], h.Indices[swap] = h.Indices[swap], h.Indices[i]
		h.Flags[i], h.Flags[swap] = h.Flags[swap], h.Flags[i]
		i = swap
	}
}

// Reset clears the heap.
func (h *MaxHeap) Reset() {
	for i := 0; i < h.K; i++ {
		h.Indices[i] = -1
		h.Distances[i] = 1e30
		h.Flags[i] = 0
	}
	h.Size = 0
}

// SimpleHeapPush pushes to array-based heap without the MaxHeap struct.
// This is for use in hot loops where we want to avoid indirection.
func SimpleHeapPush(
	indices []int32,
	distances []float32,
	k int,
	idx int32,
	dist float32,
) bool {
	// Quick reject
	if dist >= distances[0] {
		return false
	}

	// Check duplicates
	for i := 0; i < k; i++ {
		if indices[i] == idx {
			return false
		}
	}

	// Replace root
	distances[0] = dist
	indices[0] = idx

	// Sift down
	i := 0
	for {
		left := 2*i + 1
		right := 2*i + 2

		if left >= k {
			break
		}

		swap := i
		if distances[left] > distances[swap] {
			swap = left
		}
		if right < k && distances[right] > distances[swap] {
			swap = right
		}

		if swap == i {
			break
		}

		distances[i], distances[swap] = distances[swap], distances[i]
		indices[i], indices[swap] = indices[swap], indices[i]
		i = swap
	}

	return true
}

// FlaggedHeapPush pushes to array-based heap with flags.
func FlaggedHeapPush(
	indices []int32,
	distances []float32,
	flags []uint8,
	k int,
	idx int32,
	dist float32,
	flag uint8,
) bool {
	if dist >= distances[0] {
		return false
	}

	for i := 0; i < k; i++ {
		if indices[i] == idx {
			return false
		}
	}

	distances[0] = dist
	indices[0] = idx
	flags[0] = flag

	i := 0
	for {
		left := 2*i + 1
		right := 2*i + 2

		if left >= k {
			break
		}

		swap := i
		if distances[left] > distances[swap] {
			swap = left
		}
		if right < k && distances[right] > distances[swap] {
			swap = right
		}

		if swap == i {
			break
		}

		distances[i], distances[swap] = distances[swap], distances[i]
		indices[i], indices[swap] = indices[swap], indices[i]
		flags[i], flags[swap] = flags[swap], flags[i]
		i = swap
	}

	return true
}

// DeheapSort sorts heap arrays in place (ascending).
func DeheapSort(indices []int32, distances []float32, k int) {
	for i := k - 1; i > 0; i-- {
		distances[0], distances[i] = distances[i], distances[0]
		indices[0], indices[i] = indices[i], indices[0]

		// Sift down
		j := 0
		for {
			left := 2*j + 1
			right := 2*j + 2

			if left >= i {
				break
			}

			swap := j
			if distances[left] > distances[swap] {
				swap = left
			}
			if right < i && distances[right] > distances[swap] {
				swap = right
			}

			if swap == j {
				break
			}

			distances[j], distances[swap] = distances[swap], distances[j]
			indices[j], indices[swap] = indices[swap], indices[j]
			j = swap
		}
	}
}
