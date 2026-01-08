package memories

import (
	"sync"
	"time"
)

// Memory is a generic, time-indexed in-memory storage structure.
// Big brother is watching you!
//
// Items are stored with their insertion time as the key. This allows
// deterministic eviction based on age (oldest-first) and automatic
// expiration using a timeout.
//
// Memory is safe for concurrent use.
type Memory[I any] struct {
	// UUID uniquely identifies this memory instance.
	UUID UUID `json:"UUID"`

	// limit defines the maximum number of items allowed in memory.
	// When the limit is exceeded, the oldest items are purged first.
	// A value <= 0 means no limit.
	limit int

	// MemoryTimeout defines how long an item may stay in memory before
	// being automatically removed.
	// The value is expressed in milliseconds.
	// A value <= 0 disables timeout-based purging.
	timeOut time.Duration

	// items store memory entries indexed by their insertion timestamp.
	// The timestamp is used for ordering and expiration checks.
	items TimedMap[I]

	// mu protects all access to items and configuration fields.
	mu sync.RWMutex

	// autoPurgeMu protects the auto-purge lifecycle (ticker/goroutine) state.
	autoPurgeMu sync.Mutex

	// autoPurgeStop is closed to signal the auto-purge goroutine to stop.
	// A nil channel means auto-purge is not running.
	autoPurgeStop chan struct{}

	// autoPurgeTicker is the ticker used by the auto-purge goroutine.
	// It is nil when auto-purge is not running.
	autoPurgeTicker *time.Ticker

	// keyFactory generates collision-proof, totally ordered insertion keys.
	// This preserves the same semantics as using insertion time, while avoiding
	// overwrites when time resolution causes duplicate timestamps.
	keyFactory KeyFactory
}

// memoryJSON is the private on-disk / wire representation for Memory.
// It intentionally excludes synchronization and auto-purge lifecycle fields.
type memoryJSON[I any] struct {
	UUID UUID `json:"UUID"`

	// Limit is the maximum number of items kept in memory.
	// A value <= 0 means no limit.
	Limit int `json:"limit"`

	// TimeoutMS is the timeout (expiration) in milliseconds.
	// A value <= 0 disables expiration.
	TimeoutMS int64 `json:"timeout_ms"`

	// Items is a list representation of the time-indexed map.
	// We use a slice because JSON object keys must be strings.
	Items []memoryJSONEntry[I] `json:"items"`
}

type memoryJSONEntry[I any] struct {
	// Time is the insertion timestamp, encoded in RFC3339Nano.
	Time string `json:"time"`
	// Value is the stored item.
	Value I `json:"value"`
}

// NewMemory creates and returns a new Memory instance.
func NewMemory[I any](uuid UUID, limit int, timeout time.Duration, autoPurgeFrequency time.Duration) *Memory[I] {
	m := &Memory[I]{
		UUID:    uuid,
		limit:   limit,
		timeOut: timeout,
	}
	m.AutoPurge(autoPurgeFrequency)
	return m
}

// Add inserts an item into memory using the current time as its key.
//
// If the internal map is not initialized, it will be created lazily.
// After insertion, memory limits and timeouts are enforced.
func (m *Memory[I]) Add(item ...I) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.items == nil {
		// IMPORTANT:
		// Do not pass a negative size hint to make(map, hint) (it panics).
		// limit <= 0 means "no limit", so we allocate with a zero hint.
		if m.limit > 0 {
			m.items = make(TimedMap[I], m.limit)
		} else {
			m.items = make(TimedMap[I])
		}
	}
	for _, i := range item {
		// Use a collision-proof insertion key while preserving the semantics
		// of "insertion time as the key".
		m.items[m.keyFactory.NowKey()] = i
	}
	m.unsafePurgeIfNeeded()
}

// GetItems returns a shallow copy of the internal items map.
//
// The returned map is safe for the caller to read and iterate over
// without holding internal locks. Modifying the returned map will not
// affect the internal state of Memory.
//
// Note that the values stored in the map are copied by assignment.
// If I is a reference type, the underlying data it points to is not deep-copied.
func (m *Memory[I]) GetItems() TimedMap[I] {
	m.mu.RLock()
	defer m.mu.RUnlock()

	cpy := make(TimedMap[I], len(m.items))
	for k, v := range m.items {
		cpy[k] = v
	}
	return cpy
}

// GetSortedItems returns a slice of items sorted in ascending order using the map time stamps.
func (m *Memory[I]) GetSortedItems() []I {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.items.Sorted()
}

// Rewrite atomically rewrites the internal items map using a user-provided function.
//
// This is a powerful escape hatch. The provided function receives the current
// memory and returns a brand new one that will completely replace it.
//
// No one will ever observe the transition. History is rewritten in a single,
// authoritative act.
//
// Don't be evil. This method is unapologetically Orwellian.
//
// WARNING: The input map passed to the rewrite function is the internal map.
// It must be treated as read-only. Mutating it directly may cause data races.
func (m *Memory[I]) Rewrite(fn func(TimedMap[I]) TimedMap[I]) {
	if fn == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	newItems := fn(m.items)
	if newItems == nil {
		// Never allow a nil map to be stored.
		m.items = make(TimedMap[I])
		m.unsafePurgeIfNeeded()
		return
	}

	m.items = newItems
	m.unsafePurgeIfNeeded()
}

// Purge forces a purge pass according to the current limit and timeout configuration.
//
// This is useful when callers want expiration enforced on read (e.g. before injecting
// conversation history) and auto-purge is disabled.
//
// Purge is safe for concurrent use.
func (m *Memory[I]) Purge() {
	if m == nil {
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if len(m.items) == 0 {
		return
	}
	m.unsafePurgeIfNeeded()
}

// Size returns the current number of items stored in memory.
func (m *Memory[I]) Size() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.items)
}

// Timeout returns the configured memory timeout as a time.Duration.
//
// The value is returned exactly as stored and does not apply any unit
// conversion. Callers typically want milliseconds.
func (m *Memory[I]) Timeout() time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.timeOut
}

// SetMemoryLimit updates limit in a concurrency-safe way.
//
// If limit <= 0, the memory limit is disabled.
// After updating, the current contents are immediately purged to satisfy the new limit.
func (m *Memory[I]) SetMemoryLimit(limit int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.limit = limit
	m.unsafePurgeIfNeeded()
}

// SetMemoryTimeout updates MemoryTimeout in a concurrency-safe way.
//
// The timeout is stored in milliseconds (to match the MemoryTimeout field semantics).
// If timeout <= 0, timeout-based purging is disabled.
// After updating, the current contents are immediately purged to satisfy the new timeout.
func (m *Memory[I]) SetMemoryTimeout(timeout time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if timeout <= 0 {
		m.timeOut = 0
		m.unsafePurgeIfNeeded()
		return
	}

	m.timeOut = timeout
	m.unsafePurgeIfNeeded()
}

// AutoPurge starts (or restarts) a background goroutine that periodically
// purges items according to limit and MemoryTimeout.
//
// The purge runs every `every` duration. If `every` <= 0, AutoPurge does nothing.
// Calling AutoPurge while a previous auto-purge is running will restart it with
// the new frequency.
func (m *Memory[I]) AutoPurge(every time.Duration) {
	if every <= 0 {
		return
	}
	m.autoPurgeMu.Lock()
	defer m.autoPurgeMu.Unlock()

	// Halt if already running.
	m.haltAutoPurgeLocked()

	m.autoPurgeStop = make(chan struct{})
	m.autoPurgeTicker = time.NewTicker(every)
	stop := m.autoPurgeStop
	ticker := m.autoPurgeTicker

	go func() {
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				m.mu.Lock()
				if len(m.items) > 0 {
					m.unsafePurgeIfNeeded()
				}
				m.mu.Unlock()
			case <-stop:
				return
			}
		}
	}()
}

// SetAutoPurgeFrequency configures the auto-purge lifecycle.
//
// If every <= 0, any running auto-purge goroutine is stopped.
// If every > 0, auto-purge is started (or restarted) to run every `every`.
func (m *Memory[I]) SetAutoPurgeFrequency(every time.Duration) {
	if m == nil {
		return
	}
	if every <= 0 {
		m.HaltAutoPurge()
		return
	}
	m.AutoPurge(every)
}

// HaltAutoPurge stops a running auto-purge goroutine (if any).
//
// It is safe to call HaltAutoPurge multiple times.
func (m *Memory[I]) HaltAutoPurge() {
	m.autoPurgeMu.Lock()
	defer m.autoPurgeMu.Unlock()
	m.haltAutoPurgeLocked()
}

// haltAutoPurgeLocked stops auto-purge assuming autoPurgeMu is held.
func (m *Memory[I]) haltAutoPurgeLocked() {
	if m.autoPurgeStop != nil {
		close(m.autoPurgeStop)
		m.autoPurgeStop = nil
	}
	// The ticker is stopped by the goroutine via defer; we nil it here to
	// reflect the lifecycle state.
	m.autoPurgeTicker = nil
}

// unsafePurgeIfNeeded enforces memory constraints.
//
// This method assumes the caller already holds m.mu.
// It performs two independent purge operations:
//  1. Memory limit enforcement (oldest items are removed first)
//  2. Timeout-based expiration (items older than MemoryTimeout)
func (m *Memory[I]) unsafePurgeIfNeeded() {
	// Enforce memory limit by purging the oldest entries.
	if m.limit > 0 {
		for len(m.items) > m.limit {
			var oldest TimedKey
			first := true

			for k := range m.items {
				if first || k.Before(oldest) {
					oldest = k
					first = false
				}
			}
			// Defensive check: should never happen, but avoids deleting
			// a zero-value timestamp in case of corruption.
			if first {
				break
			}
			delete(m.items, oldest)
		}
	}

	// Purge expired entries based on timeout.
	if m.timeOut > 0 {
		// Equivalent to `time.Since(k) > m.timeOut` for time.Time keys,
		// implemented efficiently using the stored UnixNano timestamp.
		now := time.Now().UnixNano()
		cutoff := now - m.timeOut.Nanoseconds()

		for k := range m.items {
			if k.t < cutoff {
				delete(m.items, k)
			}
		}
	}
}
