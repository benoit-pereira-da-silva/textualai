package memories

import (
	"sync"
	"time"
)

// Storage manages a collection of Memory instances indexed by UUID.
//
// It provides concurrency-safe access to create, retrieve, and store
// multiple independent Memory objects.
type Storage[T any] struct {
	// Items map a UUID to its corresponding Memory instance.
	Items map[UUID]*Memory[T]

	// mu protects concurrent access to the Items map.
	mu sync.RWMutex
}

// NewStorage creates and returns a new Storage instance.
//
// The returned Storage is initialized with an empty memory map
// and is safe for concurrent use.
func NewStorage[T any]() *Storage[T] {
	return &Storage[T]{
		Items: make(map[UUID]*Memory[T]),
	}
}

// GetMemory retrieves a Memory by its UUID.
//
// The boolean return value indicates whether the memory was found.
// This method is safe for concurrent access.
func (s *Storage[T]) GetMemory(id UUID) (*Memory[T], bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.Items[id]
	return v, ok
}

// GetOrCreateMemory retrieves a Memory by UUID if it exists, otherwise it creates it.
//
// For an existing memory, this method also updates its limit/timeout/autopurge settings
// and forces a purge pass so reads immediately reflect the latest configuration.
//
// This method is safe for concurrent use.
func (s *Storage[T]) GetOrCreateMemory(uuid UUID, limit int, timeout time.Duration, autoPurgeFrequency time.Duration) *Memory[T] {
	s.mu.Lock()
	defer s.mu.Unlock()

	if m, ok := s.Items[uuid]; ok && m != nil {
		// Update configuration on reuse to ensure the caller-requested settings apply.
		m.SetMemoryLimit(limit)
		m.SetMemoryTimeout(timeout)
		m.SetAutoPurgeFrequency(autoPurgeFrequency)
		// SetMemoryLimit/SetMemoryTimeout already purge, but Purge() is cheap and keeps
		// the semantic explicit in case internals change.
		m.Purge()
		return m
	}

	m := NewMemory[T](uuid, limit, timeout, autoPurgeFrequency)
	s.Items[uuid] = m
	return m
}

// NewMemory creates, stores, and returns a new Memory associated with the given UUID.
//
// If a Memory already exists for the provided UUID, it will be overwritten.
// This method is safe for concurrent use.
func (s *Storage[T]) NewMemory(uuid UUID, limit int, timeout time.Duration, autoPurgeFrequency time.Duration) *Memory[T] {
	s.mu.Lock()
	defer s.mu.Unlock()

	// If overwriting, stop the previous auto-purge goroutine to avoid leaks.
	if old, ok := s.Items[uuid]; ok && old != nil {
		old.HaltAutoPurge()
	}

	m := NewMemory[T](uuid, limit, timeout, autoPurgeFrequency)
	s.Items[uuid] = m
	return m
}

// DeleteMemory removes a Memory associated with the given UUID.
//
// If no Memory exists for the provided UUID, the operation is a no-op.
// This method is safe for concurrent use.
func (s *Storage[T]) DeleteMemory(id UUID) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Stop any running auto-purge goroutine to avoid leaks.
	if old, ok := s.Items[id]; ok && old != nil {
		old.HaltAutoPurge()
	}

	delete(s.Items, id)
}

// ListMemories returns a slice of all UUIDs currently stored.
//
// The returned slice is a snapshot of the current state and is not
// affected by future modifications to the Storage.
// This method is safe for concurrent access.
func (s *Storage[T]) ListMemories() []UUID {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := make([]UUID, 0, len(s.Items))
	for id := range s.Items {
		ids = append(ids, id)
	}
	return ids
}
