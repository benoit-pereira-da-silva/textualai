package textualopenai

/*
StreamEventType represents the semantic event types emitted by the OpenAI
Responses API when streaming is enabled.

Each streamed message is a JSON object with a `type` field and optional
payload fields depending on the event.

The stream is ordered and SHOULD be processed sequentially.
*/
type StreamEventType string

const (
	// ─────────────────────────────────────────────────────────────
	// Lifecycle events
	// ─────────────────────────────────────────────────────────────

	// ResponseCreated is emitted once when the response object
	// has been created and inference begins.
	ResponseCreated StreamEventType = "response.created"

	// ResponseInProgress is emitted while the model is actively
	// generating output.
	ResponseInProgress StreamEventType = "response.in_progress"

	// ResponseCompleted signals that the response stream has
	// successfully finished.
	ResponseCompleted StreamEventType = "response.completed"

	// ResponseFailed indicates that response generation failed.
	ResponseFailed StreamEventType = "response.failed"

	// ─────────────────────────────────────────────────────────────
	// Text output events
	// ─────────────────────────────────────────────────────────────

	// OutputTextDelta contains an incremental chunk of generated text.
	// The `Delta` field will be populated.
	OutputTextDelta StreamEventType = "response.output_text.delta"

	// TextDone indicates that all text output has been streamed.
	TextDone StreamEventType = "response.text.done"

	// OutputTextAnnotationAdded carries metadata or annotations
	// associated with the generated text.
	OutputTextAnnotationAdded StreamEventType = "response.output_text_annotation_added"

	// ─────────────────────────────────────────────────────────────
	// Reasoning summary events
	// ─────────────────────────────────────────────────────────────

	// ReasoningSummaryTextDelta contains an incremental chunk of the model's
	// reasoning summary text (if reasoning summaries are enabled).
	// The `Delta` field will be populated.
	ReasoningSummaryTextDelta StreamEventType = "response.reasoning_summary_text.delta"

	// ReasoningSummaryTextDone indicates that all reasoning summary text
	// has been streamed.
	ReasoningSummaryTextDone StreamEventType = "response.reasoning_summary_text.done"

	// ReasoningSummaryPartAdded signals that a new reasoning summary part
	// has been added (for multi-part summaries).
	ReasoningSummaryPartAdded StreamEventType = "response.reasoning_summary_part.added"

	// ReasoningSummaryPartDone indicates that the current reasoning summary part
	// has completed.
	ReasoningSummaryPartDone StreamEventType = "response.reasoning_summary_part.done"

	// ─────────────────────────────────────────────────────────────
	// Structured output events
	// ─────────────────────────────────────────────────────────────

	// OutputItemAdded signals that a structured output item
	// (e.g. tool call, message block) has been added.
	OutputItemAdded StreamEventType = "response.output_item_added"

	// OutputItemDone indicates that the structured output item
	// has completed.
	OutputItemDone StreamEventType = "response.output_item_done"

	// ─────────────────────────────────────────────────────────────
	// Function / tool call events
	// ─────────────────────────────────────────────────────────────

	// FunctionCallArgumentsDelta streams incremental JSON
	// arguments for a function or tool call.
	FunctionCallArgumentsDelta StreamEventType = "response.function_call_arguments.delta"

	// FunctionCallArgumentsDone signals that the function
	// call arguments are fully streamed.
	FunctionCallArgumentsDone StreamEventType = "response.function_call_arguments.done"

	// ─────────────────────────────────────────────────────────────
	// Code Interpreter events
	// ─────────────────────────────────────────────────────────────

	// CodeInterpreterInProgress indicates that the code
	// interpreter tool is executing.
	CodeInterpreterInProgress StreamEventType = "response.code_interpreter_in_progress"

	// CodeInterpreterCallCodeDelta streams code being executed
	// by the interpreter.
	CodeInterpreterCallCodeDelta StreamEventType = "response.code_interpreter_call_code_delta"

	// CodeInterpreterCallCodeDone signals that code streaming
	// has completed.
	CodeInterpreterCallCodeDone StreamEventType = "response.code_interpreter_call_code_done"

	// CodeInterpreterCallInterpreting indicates that the
	// interpreter is evaluating results.
	CodeInterpreterCallInterpreting StreamEventType = "response.code_interpreter_call_interpreting"

	// CodeInterpreterCallCompleted indicates the interpreter
	// call has fully completed.
	CodeInterpreterCallCompleted StreamEventType = "response.code_interpreter_call_completed"

	// ─────────────────────────────────────────────────────────────
	// File search events
	// ─────────────────────────────────────────────────────────────

	// FileSearchCallInProgress indicates a file search tool
	// invocation has started.
	FileSearchCallInProgress StreamEventType = "response.file_search_call_in_progress"

	// FileSearchCallSearching indicates that file search
	// is actively querying sources.
	FileSearchCallSearching StreamEventType = "response.file_search_call_searching"

	// FileSearchCallCompleted indicates the file search
	// tool has completed.
	FileSearchCallCompleted StreamEventType = "response.file_search_call_completed"

	// ─────────────────────────────────────────────────────────────
	// Refusal & error events
	// ─────────────────────────────────────────────────────────────

	// RefusalDelta streams partial refusal content.
	RefusalDelta StreamEventType = "response.refusal.delta"

	// RefusalDone indicates refusal streaming has finished.
	RefusalDone StreamEventType = "response.refusal.done"

	// Error represents a generic stream error.
	Error StreamEventType = "error"
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
	Type    StreamEventType `json:"type"`
	Delta   string          `json:"delta,omitempty"`
	Text    string          `json:"text,omitempty"`
	Code    string          `json:"code,omitempty"`
	Message string          `json:"message,omitempty"`
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
