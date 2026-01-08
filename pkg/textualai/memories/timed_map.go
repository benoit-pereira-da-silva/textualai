// Package memories provide time-indexed data structures.
//
// TimedMap is a generic map keyed by time.Time, with helper methods to
// retrieve values in a deterministic chronological order.
package memories

import (
	"sort"
	"time"
)

// TimedMap is a map indexed by time.Time.
//
// The generic type parameter "I" represents the value stored at a given
// timestamp. Since Go maps are unordered by definition, TimedMap exposes
// utility methods (such as Sorted) to get values in a predictable order.
type TimedMap[I any] map[time.Time]I

// Sorted returns all values stored in the TimedMap ordered by their timestamp.
//
// The values are returned in ascending chronological order (from the earliest
// time.Time key to the latest). The original map is not modified.
//
// If the map is empty, Sorted returns an empty slice.
func (t TimedMap[I]) Sorted() []I {
	if len(t) == 0 {
		return []I{}
	}

	keys := make([]time.Time, 0, len(t))
	for k := range t {
		keys = append(keys, k)
	}

	sort.Slice(keys, func(i, j int) bool {
		return keys[i].Before(keys[j])
	})

	result := make([]I, 0, len(t))
	for _, k := range keys {
		result = append(result, t[k])
	}
	return result
}
