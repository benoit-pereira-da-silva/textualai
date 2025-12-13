package textualshared

// AggregateType controls how streamed chunks are turned into output carrier values.
//
//   - Word: emit when we cross a whitespace / punctuation boundary.
//   - Line: emit when we cross a newline boundary.
type AggregateType string

const (
	Word AggregateType = "word"
	Line AggregateType = "line"
)

// Role is the message role used for the top-level "message" input item when
// ResponseProcessor builds a chat-like input payload.
type Role string
