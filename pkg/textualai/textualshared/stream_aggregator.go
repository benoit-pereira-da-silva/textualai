package textualshared

import "unicode"

// --------------------------
// Stream aggregation helpers
// --------------------------

// StreamAggregator buffers streamed text chunks and emits "complete" segments
// according to AggregateType.
//
// # IMPORTANT STREAMING SEMANTICS
//
// StreamAggregator emits *incremental* segments only (deltas), not the full
// accumulated buffer. This is required for correct streaming pipelines and
// avoids duplicating previously emitted data.
type StreamAggregator struct {
	aggType AggregateType
	buffer  []rune
}

// NewStreamAggregator constructs a StreamAggregator.
// Unknown aggType values fall back to Word.
func NewStreamAggregator(aggType AggregateType) *StreamAggregator {
	switch aggType {
	case Word, Line, JSON:
		// ok
	default:
		aggType = Word
	}
	return &StreamAggregator{
		aggType: aggType,
		buffer:  make([]rune, 0, 256),
	}
}

// Append adds a new streamed chunk and returns zero or more complete segments,
// depending on the aggregation strategy.
func (a *StreamAggregator) Append(chunk string) []string {
	if chunk == "" {
		return nil
	}
	a.buffer = append(a.buffer, []rune(chunk)...)

	switch a.aggType {
	case Word:
		return a.collectByDelimiter(isWordBoundaryRune)
	case Line:
		return a.collectByDelimiter(func(r rune) bool { return r == '\n' })
	case JSON:
		return a.collectJSONValues(false)
	default:
		// Defensive fallback: emit everything we have.
		return a.collectAll()
	}
}

// Final flushes any remaining buffered data when the stream ends.
func (a *StreamAggregator) Final() []string {
	if len(a.buffer) == 0 {
		return nil
	}

	switch a.aggType {
	case JSON:
		// Try to emit any remaining complete JSON values, then flush any tail
		// (which may be incomplete JSON / noise) so downstream can surface errors.
		out := a.collectJSONValues(false)
		if len(a.buffer) == 0 {
			return out
		}
		out = append(out, string(a.buffer))
		a.buffer = a.buffer[:0]
		return out

	default:
		// Word / Line: flush remainder as-is.
		out := []string{string(a.buffer)}
		a.buffer = a.buffer[:0]
		return out
	}
}

// collectAll emits the entire current buffer and clears it.
func (a *StreamAggregator) collectAll() []string {
	if len(a.buffer) == 0 {
		return nil
	}
	out := []string{string(a.buffer)}
	a.buffer = a.buffer[:0]
	return out
}

// collectByDelimiter emits incremental segments ending at each delimiter rune.
// The delimiter rune is included in the emitted segment.
func (a *StreamAggregator) collectByDelimiter(delim func(rune) bool) []string {
	if len(a.buffer) == 0 {
		return nil
	}

	var out []string
	start := 0

	for i := 0; i < len(a.buffer); i++ {
		if delim(a.buffer[i]) {
			end := i + 1
			if end > start {
				out = append(out, string(a.buffer[start:end]))
				start = end
			}
		}
	}

	// Keep the tail (partial segment) in the buffer.
	if start > 0 {
		a.buffer = append([]rune(nil), a.buffer[start:]...)
	}
	return out
}

// collectJSONValues emits each complete top-level JSON value found in the buffer.
// It uses a lightweight framing algorithm (brace/bracket nesting + string handling).
//
// Behavior:
//   - Leading noise before the next '{' or '[' is discarded.
//   - Multiple JSON values in the same buffer are emitted one by one.
//   - If a value is incomplete, it is kept in the buffer until more data arrives
//     (or until Final() flushes the tail).
func (a *StreamAggregator) collectJSONValues(_atEOF bool) []string {
	var out []string

	for {
		if len(a.buffer) == 0 {
			return out
		}

		// Find the next '{' or '['; discard anything before it.
		start := -1
		for i, r := range a.buffer {
			if r == '{' || r == '[' {
				start = i
				break
			}
		}

		if start == -1 {
			// No JSON opening delimiter left; treat remaining as noise.
			a.buffer = a.buffer[:0]
			return out
		}

		if start > 0 {
			a.buffer = a.buffer[start:]
		}

		// a.buffer[0] is '{' or '['.
		stack := make([]rune, 0, 8)
		stack = append(stack, a.buffer[0])

		inString := false
		escaped := false

		for i := 1; i < len(a.buffer); i++ {
			r := a.buffer[i]

			if inString {
				if escaped {
					escaped = false
					continue
				}
				if r == '\\' {
					escaped = true
					continue
				}
				if r == '"' {
					inString = false
				}
				continue
			}

			switch r {
			case '"':
				inString = true

			case '{', '[':
				stack = append(stack, r)

			case '}', ']':
				if len(stack) == 0 {
					// Invalid framing; stop and let downstream decide how to handle tail.
					return out
				}
				top := stack[len(stack)-1]
				matches := (r == '}' && top == '{') || (r == ']' && top == '[')
				if !matches {
					// Invalid framing; stop and let downstream decide how to handle tail.
					return out
				}

				// Pop.
				stack = stack[:len(stack)-1]
				if len(stack) == 0 {
					// Complete JSON value ends at i.
					end := i + 1
					out = append(out, string(a.buffer[:end]))

					// Consume emitted value and continue scanning for next one.
					a.buffer = a.buffer[end:]
					goto nextValue
				}
			}
		}

		// Incomplete JSON value; wait for more data.
		return out

	nextValue:
		// Continue outer loop.
	}
}

func isWordBoundaryRune(r rune) bool {
	if unicode.IsSpace(r) {
		return true
	}
	switch r {
	case '.', ',', ';', ':', '!', '?', '…', '»', '«':
		return true
	default:
		return false
	}
}
