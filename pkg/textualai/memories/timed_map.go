// Package memories provide time-indexed data structures.
//
// TimedMap is a generic map keyed by time.Time, with helper methods to
// retrieve values in a deterministic chronological order.
package memories

import (
	"sort"
	"sync"
	"time"
)

// TimedKey is a collision-proof, totally ordered key.
//
// t is the wall-clock timestamp in Unix nanoseconds.
// n is a per-timestamp sequence number used to break ties when multiple
// inserts happen with the same UnixNano value.
type TimedKey struct {
	t int64  // UnixNano
	n uint64 // tie-breaker for identical t
}

// Before reports whether k occurs before other in total order.
func (k TimedKey) Before(other TimedKey) bool {
	if k.t != other.t {
		return k.t < other.t
	}
	return k.n < other.n
}

// Time returns the wall-clock time represented by this key (the monotonic component is not preserved).
func (k TimedKey) Time() time.Time {
	return time.Unix(0, k.t)
}

// TimedMap is a map indexed by a collision-proof time-like key.
//
// The generic type parameter "I" represents the value stored at a given
// timestamp. Since Go maps are unordered by definition, TimedMap exposes
// utility methods (such as Sorted) to get values in a predictable order.
//
// Keys are ordered first by time (t), then by tie-breaker (n).
type TimedMap[I any] map[TimedKey]I

// Sorted returns all values stored in the TimedMap ordered by their key.
//
// The values are returned in ascending chronological order (from the earliest
// key to the latest). The original map is not modified.
//
// If the map is empty, Sorted returns an empty slice.
func (t TimedMap[I]) Sorted() []I {
	if len(t) == 0 {
		return []I{}
	}

	keys := make([]TimedKey, 0, len(t))
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

// KeyFactory generates strictly increasing TimedKeys.
//
// It guarantees uniqueness even if time.Now().UnixNano() returns the same
// value multiple times in a row.
type KeyFactory struct {
	mu    sync.Mutex
	lastT int64
	seq   uint64
}

// NowKey returns a new unique TimedKey based on the current time.
func (kf *KeyFactory) NowKey() TimedKey {
	now := time.Now().UnixNano()

	kf.mu.Lock()
	defer kf.mu.Unlock()

	if now == kf.lastT {
		kf.seq++
	} else if now > kf.lastT {
		kf.lastT = now
		kf.seq = 0
	} else {
		// System clock went backwards (possible with wall time).
		// Clamp to lastT to keep keys monotonic, then bump seq.
		now = kf.lastT
		kf.seq++
	}

	return TimedKey{t: now, n: kf.seq}
}
