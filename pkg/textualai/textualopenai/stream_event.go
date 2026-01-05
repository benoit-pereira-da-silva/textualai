// Copyright 2026 Benoit Pereira da Silva
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package textualopenai

import "encoding/json"

/*
EventType represents the semantic event types emitted by the OpenAI
Responses API when streaming is enabled.

Each streamed message is a JSON object with a `type` field and optional
payload fields depending on the event.

The stream is ordered and SHOULD be processed sequentially.

Streaming event schemas:
https://platform.openai.com/docs/api-reference/responses-streaming
*/
type EventType string

const (

	// AllEvent permit observing or listen any EventType.
	// This is not an OpenAI event type: it is a local wildcard.
	AllEvent EventType = "all"

	// ─────────────────────────────────────────────────────────────
	// Lifecycle events
	// ─────────────────────────────────────────────────────────────

	// ResponseCreated is emitted once when the response object
	// has been created and inference begins.
	ResponseCreated EventType = "response.created"

	// ResponseQueued is emitted when a response is queued and waiting
	// to be processed.
	ResponseQueued EventType = "response.queued"

	// ResponseInProgress is emitted while the model is actively
	// generating output.
	ResponseInProgress EventType = "response.in_progress"

	// ResponseCompleted signals that the response stream has
	// successfully finished.
	ResponseCompleted EventType = "response.completed"

	// ResponseFailed indicates that response generation failed.
	ResponseFailed EventType = "response.failed"

	// ─────────────────────────────────────────────────────────────
	// Output item events
	// ─────────────────────────────────────────────────────────────

	// OutputItemAdded signals that a structured output item
	// (e.g. tool call, message block) has been added.
	OutputItemAdded EventType = "response.output_item.added"

	// OutputItemDone indicates that the structured output item
	// has completed.
	OutputItemDone EventType = "response.output_item.done"

	// ─────────────────────────────────────────────────────────────
	// Text output events
	// ─────────────────────────────────────────────────────────────

	// OutputTextDelta contains an incremental chunk of generated text.
	// The `Delta` field will be populated.
	OutputTextDelta EventType = "response.output_text.delta"

	// TextDone indicates that all text content for a given output item / content part
	// has been streamed.
	TextDone EventType = "response.output_text.done"

	// OutputTextAnnotationAdded carries metadata or annotations
	// associated with the generated text.
	OutputTextAnnotationAdded EventType = "response.output_text.annotation.added"

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
	// Function / tool call events
	// ─────────────────────────────────────────────────────────────

	// FunctionCallArgumentsDelta streams incremental JSON
	// arguments for a function/tool call.
	FunctionCallArgumentsDelta EventType = "response.function_call_arguments.delta"

	// FunctionCallArgumentsDone signals that the function
	// call arguments are fully streamed.
	FunctionCallArgumentsDone EventType = "response.function_call_arguments.done"

	// CustomToolCallInputDelta streams incremental custom tool input.
	CustomToolCallInputDelta EventType = "response.custom_tool_call_input.delta"

	// CustomToolCallInputDone signals that the custom tool input is finalized.
	CustomToolCallInputDone EventType = "response.custom_tool_call_input.done"

	// ─────────────────────────────────────────────────────────────
	// Code Interpreter events
	// ─────────────────────────────────────────────────────────────

	// CodeInterpreterInProgress indicates that a code interpreter tool call
	// is in progress.
	CodeInterpreterInProgress EventType = "response.code_interpreter_call.in_progress"

	// CodeInterpreterCallCodeDelta streams code being executed
	// by the interpreter.
	CodeInterpreterCallCodeDelta EventType = "response.code_interpreter_call_code.delta"

	// CodeInterpreterCallCodeDone signals that code streaming
	// has completed.
	CodeInterpreterCallCodeDone EventType = "response.code_interpreter_call_code.done"

	// CodeInterpreterCallInterpreting indicates that the
	// interpreter is evaluating results.
	CodeInterpreterCallInterpreting EventType = "response.code_interpreter_call.interpreting"

	// CodeInterpreterCallCompleted indicates the interpreter
	// call has fully completed.
	CodeInterpreterCallCompleted EventType = "response.code_interpreter_call.completed"

	// ─────────────────────────────────────────────────────────────
	// File search events
	// ─────────────────────────────────────────────────────────────

	// FileSearchCallInProgress indicates a file search tool
	// invocation has started.
	FileSearchCallInProgress EventType = "response.file_search_call.in_progress"

	// FileSearchCallSearching indicates that file search
	// is actively querying sources.
	FileSearchCallSearching EventType = "response.file_search_call.searching"

	// FileSearchCallCompleted indicates the file search
	// tool has completed.
	FileSearchCallCompleted EventType = "response.file_search_call.completed"

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
  - SequenceNumber: The sequence number of this event, used to order streaming events.
  - ResponseID: The response id that this event relates to.
  - OutputIndex: The output array index.
  - ItemID: The output item id.
  - ContentIndex: The content part index within an output item.
  - AnnotationIndex: The annotation index within a content part.
  - Delta: Incremental text or JSON fragment
  - Text: Final text payload for *.done events
  - Refusal: Final refusal payload for refusal *.done events
  - Name / Arguments: Function/tool call name and arguments (function calling)
  - Code: Code snippet (code interpreter) OR error code (error events)
  - Message: Error or informational message
  - Item: Structured output item payload (output_item.* events)
  - Response: Full response payload (response.* lifecycle events)
  - Annotation: Annotation payload (output_text.annotation.added)
*/
type StreamEvent struct {
	Type EventType `json:"type"`

	// Common stream metadata
	SequenceNumber  int    `json:"sequence_number,omitempty"`
	ResponseID      string `json:"response_id,omitempty"`
	OutputIndex     int    `json:"output_index,omitempty"`
	ItemID          string `json:"item_id,omitempty"`
	ContentIndex    int    `json:"content_index,omitempty"`
	AnnotationIndex int    `json:"annotation_index,omitempty"`

	// Deltas / finalized payloads
	Delta   string `json:"delta,omitempty"`
	Text    string `json:"text,omitempty"`
	Refusal string `json:"refusal,omitempty"`

	// Function calling payload
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`

	// Code interpreter / error payload
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Param   any    `json:"param,omitempty"`

	// Structured payloads
	Item       json.RawMessage `json:"item,omitempty"`
	Response   json.RawMessage `json:"response,omitempty"`
	Annotation json.RawMessage `json:"annotation,omitempty"`
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
