package memories

import (
	"encoding/json"
	"errors"
	"io/fs"
	"sort"
	"time"
)

// FS is the filesystem abstraction used by this package for persistence.
//
// It is intentionally stricter than io/fs.FS: it requires write support via WriteFile.
// Reading is provided by embedding io/fs.FS (Open), which is enough for fs.ReadFile.
//
// Atomic writes are best-effort: if the filesystem also implements Rename
type FS interface {
	Open(name string) (fs.File, error)
	WriteFile(name string, data []byte, perm fs.FileMode) error
	Rename(oldName, newName string) error
}

// SaveJSONFile persists the provided Memory to a JSON file.
//
// Filesystem requirements:
//   - fsys MUST be non-nil.
//   - fsys MUST support writing via WriteFile.
//   - Atomic rename is attempted if the filesystem also supports a Rename method.
//     If rename is not available, the data is written directly to `path`.
//
// NOTE: The Go standard library's io/fs.FS is read-only by design, so this package
// defines its own memories.FS interface to require write support.
func SaveJSONFile[I any](fsys FS, m *Memory[I], path string) error {
	if m == nil {
		return errors.New("memories: nil memory")
	}
	if path == "" {
		return errors.New("memories: empty path")
	}
	if fsys == nil {
		return errors.New("memories: nil filesystem")
	}

	b, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if wErr := fsys.WriteFile(tmp, b, 0o600); wErr != nil {
		return wErr
	}
	return fsys.Rename(tmp, path)

}

// LoadJSONFile loads durable Memory state from a JSON file and returns a new Memory instance.
//
// Filesystem requirements:
//   - fsys MUST be non-nil and readable (implements Open).
//   - FS includes write methods, but only read is needed for Load.
//
// AutoPurge is not started on load to avoid surprising background behavior and accidental
// deletion immediately after restoring from disk. Callers can explicitly start it if desired.
func LoadJSONFile[I any](fsys FS, path string) (*Memory[I], error) {
	if path == "" {
		return nil, errors.New("memories: empty path")
	}
	if fsys == nil {
		return nil, errors.New("memories: nil filesystem")
	}

	b, err := fs.ReadFile(fsys, path)
	if err != nil {
		return nil, err
	}

	m := &Memory[I]{}
	if uErr := json.Unmarshal(b, m); uErr != nil {
		return nil, uErr
	}

	return m, nil
}

// MarshalJSON implements json.Marshaler.
//
// The JSON representation includes only the durable state (UUID, limit, timeout, items)
// and intentionally excludes internal synchronization and auto-purge lifecycle fields.
func (m *Memory[I]) MarshalJSON() ([]byte, error) {
	if m == nil {
		return []byte("null"), nil
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	// Build a stable, deterministic slice ordering for reproducible JSON output.
	entries := make([]memoryJSONEntry[I], 0, len(m.items))
	for t, v := range m.items {
		entries = append(entries, memoryJSONEntry[I]{
			Time:  t.UTC().Format(time.RFC3339Nano),
			Value: v,
		})
	}
	sort.Slice(entries, func(i, j int) bool {
		// RFC3339Nano strings sort lexicographically by time when all are UTC.
		return entries[i].Time < entries[j].Time
	})

	payload := memoryJSON[I]{
		UUID:      m.UUID,
		Limit:     m.limit,
		TimeoutMS: m.timeOut.Milliseconds(),
		Items:     entries,
	}

	return json.Marshal(payload)
}

// UnmarshalJSON implements json.Unmarshaler.
//
// It restores only the durable state (UUID, limit, timeout, items) and leaves
// internal synchronization and auto-purge lifecycle fields in their zero state.
// Callers can start auto-purging again by calling AutoPurge.
//
// Backward compatibility:
// - Supports "timeout_ms" (int64, preferred)
// - Also supports "timeout" as a Go duration string (e.g. "1500ms", "2s") if present.
func (m *Memory[I]) UnmarshalJSON(data []byte) error {
	if m == nil {
		return errors.New("memories: UnmarshalJSON on nil *Memory")
	}

	// Handle JSON null.
	if len(data) == 0 || string(data) == "null" {
		*m = Memory[I]{}
		return nil
	}

	// Use an intermediate struct to support backward compatibility.
	var raw struct {
		UUID      UUID                 `json:"UUID"`
		Limit     int                  `json:"limit"`
		TimeoutMS *int64               `json:"timeout_ms"`
		Timeout   *string              `json:"timeout"`
		Items     []memoryJSONEntry[I] `json:"items"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	timeoutMS := int64(0)
	if raw.TimeoutMS != nil {
		timeoutMS = *raw.TimeoutMS
	} else if raw.Timeout != nil {
		d, err := time.ParseDuration(*raw.Timeout)
		if err != nil {
			return err
		}
		timeoutMS = d.Milliseconds()
	}

	items := make(map[time.Time]I, len(raw.Items))
	for _, e := range raw.Items {
		if e.Time == "" {
			return errors.New("memories: invalid item time: empty")
		}
		t, err := time.Parse(time.RFC3339Nano, e.Time)
		if err != nil {
			return err
		}
		items[t] = e.Value
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	m.UUID = raw.UUID
	m.limit = raw.Limit
	if timeoutMS <= 0 {
		m.timeOut = 0
	} else {
		m.timeOut = time.Duration(timeoutMS) * time.Millisecond
	}
	m.items = items
	// Keep invariants: never store a nil map.
	if m.items == nil {
		m.items = make(map[time.Time]I)
	}
	return nil
}
