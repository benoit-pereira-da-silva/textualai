package memories

import (
	"encoding/json"
	"errors"
	"io"
	"sort"
	"time"
)

// WriteJSON writes the provided Memory as JSON into w.
//
// This is an io-based API (streams) rather than a file-based API.
//
// Requirements:
//   - w MUST be non-nil.
//   - m MUST be non-nil.
//
// The JSON produced uses Memory's custom MarshalJSON implementation, which
// includes only durable state (UUID, limit, timeout, items).
func (m *Memory[I]) WriteJSON(w io.Writer) error {
	return m.WriteJSONIndent(w, "", "  ")
}

// WriteJSONIndent writes the provided Memory as JSON into w using json.Encoder
// with indentation.
//
// Requirements:
//   - w MUST be non-nil.
//   - m MUST be non-nil.
//
// Use indent="" and prefix="" for compact output (no extra whitespace).
func (m *Memory[I]) WriteJSONIndent(w io.Writer, prefix, indent string) error {
	if m == nil {
		return errors.New("memories: nil memory")
	}
	if w == nil {
		return errors.New("memories: nil writer")
	}

	enc := json.NewEncoder(w)
	if indent != "" || prefix != "" {
		enc.SetIndent(prefix, indent)
	}
	// Encoder.Encode writes a trailing newline; this is usually desirable for
	// stream and file sinks alike.
	return enc.Encode(m)
}

// LoadJSON reads a Memory instance from r.
//
// This is an io-based API (streams) rather than a file-based API.
//
// Requirements:
//   - r MUST be non-nil.
//
// AutoPurge is not started on load to avoid surprising background behavior and accidental
// deletion immediately after restoring from disk or a remote stream. Callers can explicitly
// start it if desired.
func LoadJSON[I any](r io.Reader) (*Memory[I], error) {
	if r == nil {
		return nil, errors.New("memories: nil reader")
	}

	m := &Memory[I]{}
	if err := DecodeJSONInto[I](r, m); err != nil {
		return nil, err
	}
	return m, nil
}

// DecodeJSONInto decodes a Memory JSON payload from r into the provided Memory.
//
// Requirements:
//   - r MUST be non-nil.
//   - m MUST be non-nil.
func DecodeJSONInto[I any](r io.Reader, m *Memory[I]) error {
	if r == nil {
		return errors.New("memories: nil reader")
	}
	if m == nil {
		return errors.New("memories: nil memory")
	}

	dec := json.NewDecoder(r)
	return dec.Decode(m)
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
	for k, v := range m.items {
		entries = append(entries, memoryJSONEntry[I]{
			Time:  k.Time().UTC().Format(time.RFC3339Nano),
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

	items := make(TimedMap[I], len(raw.Items))
	// seqByTimeNS tracks how many entries we have already loaded for a given
	// UnixNano timestamp, so we can assign a stable tie-breaker to preserve
	// distinct entries even when times collide.
	seqByTimeNS := make(map[int64]uint64)

	for _, e := range raw.Items {
		if e.Time == "" {
			return errors.New("memories: invalid item time: empty")
		}
		tt, err := time.Parse(time.RFC3339Nano, e.Time)
		if err != nil {
			return err
		}

		ns := tt.UnixNano()
		seq := seqByTimeNS[ns]
		seqByTimeNS[ns] = seq + 1

		items[TimedKey{t: ns, n: seq}] = e.Value
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
		m.items = make(TimedMap[I])
	}
	return nil
}
