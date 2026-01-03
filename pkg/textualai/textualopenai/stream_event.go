package textualopenai

import "encoding/json"

/*
EventType represents the semantic event types emitted by the OpenAI
Responses API when streaming is enabled.

Each streamed message is a JSON object with a `type` field and optional
payload fields depending on the event.

The stream is ordered and SHOULD be processed sequentially.
*/
type EventType string

const (

	// AllEvent permit observing or listen any EventType
	AllEvent EventType = "all"

	// ─────────────────────────────────────────────────────────────
	// Lifecycle events
	// ─────────────────────────────────────────────────────────────

	// ResponseCreated is emitted once when the response object
	// has been created and inference begins.
	ResponseCreated EventType = "response.created"

	// ResponseInProgress is emitted while the model is actively
	// generating output.
	ResponseInProgress EventType = "response.in_progress"

	// ResponseCompleted signals that the response stream has
	// successfully finished.
	ResponseCompleted EventType = "response.completed"

	// ResponseFailed indicates that response generation failed.
	ResponseFailed EventType = "response.failed"

	// ─────────────────────────────────────────────────────────────
	// Text output events
	// ─────────────────────────────────────────────────────────────

	// OutputTextDelta contains an incremental chunk of generated text.
	// The `Delta` field will be populated.
	OutputTextDelta EventType = "response.output_text.delta"

	// TextDone indicates that all text output has been streamed.
	TextDone EventType = "response.text.done"

	// OutputTextAnnotationAdded carries metadata or annotations
	// associated with the generated text.
	OutputTextAnnotationAdded EventType = "response.output_text_annotation_added"

	// ─────────────────────────────────────────────────────────────
	// Reasoning summary events
	// ─────────────────────────────────────────────────────────────

	// ReasoningSummaryTextDelta contains an incremental chunk of the model's
	// reasoning summary text (if reasoning summaries are enabled).
	// The `Delta` field will be populated.
	ReasoningSummaryTextDelta EventType = "response.reasoning_summary_text.delta"

	// ReasoningSummaryTextDone indicates that all reasoning summary text
	// has been streamed.
	ReasoningSummaryTextDone EventType = "response.reasoning_summary_text.done"

	// ReasoningSummaryPartAdded signals that a new reasoning summary part
	// has been added (for multi-part summaries).
	ReasoningSummaryPartAdded EventType = "response.reasoning_summary_part.added"

	// ReasoningSummaryPartDone indicates that the current reasoning summary part
	// has completed.
	ReasoningSummaryPartDone EventType = "response.reasoning_summary_part.done"

	// ─────────────────────────────────────────────────────────────
	// Structured output events
	// ─────────────────────────────────────────────────────────────

	// OutputItemAdded signals that a structured output item
	// (e.g. tool call, message block) has been added.
	OutputItemAdded EventType = "response.output_item_added"

	// OutputItemDone indicates that the structured output item
	// has completed.
	OutputItemDone EventType = "response.output_item_done"

	// ─────────────────────────────────────────────────────────────
	// Function / tool call events
	// ─────────────────────────────────────────────────────────────

	// FunctionCallArgumentsDelta streams incremental JSON
	// arguments for a function or tool call.
	FunctionCallArgumentsDelta EventType = "response.function_call_arguments.delta"

	// FunctionCallArgumentsDone signals that the function
	// call arguments are fully streamed.
	FunctionCallArgumentsDone EventType = "response.function_call_arguments.done"

	// ─────────────────────────────────────────────────────────────
	// Code Interpreter events
	// ─────────────────────────────────────────────────────────────

	// CodeInterpreterInProgress indicates that the code
	// interpreter tool is executing.
	CodeInterpreterInProgress EventType = "response.code_interpreter_in_progress"

	// CodeInterpreterCallCodeDelta streams code being executed
	// by the interpreter.
	CodeInterpreterCallCodeDelta EventType = "response.code_interpreter_call_code_delta"

	// CodeInterpreterCallCodeDone signals that code streaming
	// has completed.
	CodeInterpreterCallCodeDone EventType = "response.code_interpreter_call_code_done"

	// CodeInterpreterCallInterpreting indicates that the
	// interpreter is evaluating results.
	CodeInterpreterCallInterpreting EventType = "response.code_interpreter_call_interpreting"

	// CodeInterpreterCallCompleted indicates the interpreter
	// call has fully completed.
	CodeInterpreterCallCompleted EventType = "response.code_interpreter_call_completed"

	// ─────────────────────────────────────────────────────────────
	// File search events
	// ─────────────────────────────────────────────────────────────

	// FileSearchCallInProgress indicates a file search tool
	// invocation has started.
	FileSearchCallInProgress EventType = "response.file_search_call_in_progress"

	// FileSearchCallSearching indicates that file search
	// is actively querying sources.
	FileSearchCallSearching EventType = "response.file_search_call_searching"

	// FileSearchCallCompleted indicates the file search
	// tool has completed.
	FileSearchCallCompleted EventType = "response.file_search_call_completed"

	// ─────────────────────────────────────────────────────────────
	// Refusal & error events
	// ─────────────────────────────────────────────────────────────

	// RefusalDelta streams partial refusal content.
	RefusalDelta EventType = "response.refusal.delta"

	// RefusalDone indicates refusal streaming has finished.
	RefusalDone EventType = "response.refusal.done"

	// Error represents a generic stream error.
	Error EventType = "error"
)

/*
StreamEvent represents a single event emitted from the Responses API
stream when `stream=true` is enabled.

Only a subset of fields will be populated depending on the event `Type`.

Field semantics:
  - Type: The event type identifier (always present)
  - Delta: Incremental text or JSON fragment
  - Text: Full text payload (non-streamed events)
  - Code: Code being executed (code interpreter events)
  - Message: Error or informational message
*/
type StreamEvent struct {
	Type    EventType `json:"type"`
	Delta   string    `json:"delta,omitempty"`
	Text    string    `json:"text,omitempty"`
	Code    string    `json:"code,omitempty"`
	Message string    `json:"message,omitempty"`
}

/*
IsTerminal returns true if this event represents a terminal state
for the stream (completed, failed, or error).
*/
func (s StreamEvent) IsTerminal() bool {
	switch s.Type {
	case ResponseCompleted,
		ResponseFailed,
		Error:
		return true
	default:
		return false
	}
}

/*
IsTextDelta returns true if the event carries incremental text output.
*/
func (s StreamEvent) IsTextDelta() bool {
	return s.Type == OutputTextDelta
}

func (s StreamEvent) ToJson() string {
	b, _ := json.Marshal(s)
	return string(b)
}
