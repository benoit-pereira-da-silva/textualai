package textualshared

import "unicode"

// --------------------------
// Stream aggregation helpers
// --------------------------

type StreamAggregator struct {
	aggType     AggregateType
	buffer      []rune
	lastEmitPos int
}

func NewStreamAggregator(aggType AggregateType) *StreamAggregator {
	if aggType != Word && aggType != Line {
		aggType = Word
	}
	return &StreamAggregator{
		aggType:     aggType,
		buffer:      make([]rune, 0),
		lastEmitPos: 0,
	}
}

func (a *StreamAggregator) Append(chunk string) []string {
	if chunk == "" {
		return nil
	}
	a.buffer = append(a.buffer, []rune(chunk)...)

	switch a.aggType {
	case Word:
		return a.collect(isWordBoundaryRune)
	case Line:
		return a.collect(func(r rune) bool { return r == '\n' })
	default:
		return a.collect(nil)
	}
}

func (a *StreamAggregator) Final() []string {
	if len(a.buffer) == 0 || a.lastEmitPos >= len(a.buffer) {
		return nil
	}
	a.lastEmitPos = len(a.buffer)
	return []string{string(a.buffer)}
}

func (a *StreamAggregator) collect(delim func(rune) bool) []string {
	var out []string

	if delim == nil {
		if len(a.buffer) > a.lastEmitPos {
			out = append(out, string(a.buffer))
			a.lastEmitPos = len(a.buffer)
		}
		return out
	}

	for i := a.lastEmitPos; i < len(a.buffer); i++ {
		if delim(a.buffer[i]) {
			pos := i + 1
			if pos <= a.lastEmitPos {
				continue
			}
			out = append(out, string(a.buffer[:pos]))
			a.lastEmitPos = pos
		}
	}
	return out
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
