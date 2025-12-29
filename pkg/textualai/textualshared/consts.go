package textualshared

// AggregateType controls how streamed chunks are turned into output carrier values.
//
//   - Word: emit when we cross a whitespace / punctuation boundary.
//   - Line: emit when we cross a newline boundary.
//   - JSON: emit when we detect a complete top-level JSON value (object `{...}` or array `[...]`).
type AggregateType string

const (
	Word AggregateType = "word"
	Line AggregateType = "line"
	JSON AggregateType = "json"
)

// Role is the message role used for the top-level "message" input item when
// ResponseProcessor builds a chat-like input payload.
type Role string
