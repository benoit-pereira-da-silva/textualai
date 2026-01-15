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

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"sync"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/models"
)

// InputItem is the minimal "message-like" shape used in the Responses `input` array.
//
// In the API, `content` can be:
//   - a plain string, OR
//   - a structured list of content parts.
//
// We keep it as `any` so callers can provide either representation.
type InputItem struct {
	Role    string `json:"role,omitempty"`
	Content any    `json:"content,omitempty"`
}

// ResponsesRequest
// https://platform.openai.com/docs/api-reference/responses
type ResponsesRequest struct {

	// Main request fields.
	//
	// This struct is a streaming-first mapping of the REST /responses "Create a response"
	// request body.
	//
	// Notes:
	//  - Input is `any` so callers can pass either a string, or the full "input item"
	//    array (text, images, files, tool outputs, ...).
	//  - This client currently streams only; Stream must be set to true.
	//  - Tool/function calling is supported via the `tools`/`tool_choice` fields and the
	//    built-in function delegate helpers (see function_delegate.go).
	Model        models.ModelID `json:"model,omitempty"`
	Input        any            `json:"input,omitempty"`
	Stream       bool           `json:"stream,omitempty"`
	Instructions string         `json:"instructions,omitempty"`

	// ─────────────────────────────────────────────────────────────
	// Output configuration
	// ─────────────────────────────────────────────────────────────

	// MaxOutputTokens limits the number of tokens that can be generated for this response.
	MaxOutputTokens int `json:"max_output_tokens,omitempty"`

	// Thinking is a legacy flag kept for backward compatibility with earlier experiments.
	// Prefer the `Reasoning` field for modern reasoning controls.
	Thinking bool `json:"thinking,omitempty"`

	// Temperature controls sampling randomness. Use a pointer so callers can explicitly
	// What sampling temperature to use, between 0 and 2.
	// Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
	// We generally recommend altering this or top_p but not both.
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP enables nucleus sampling. Pointer keeps the ability to explicitly send 0/1
	// An alternative to sampling with temperature, called nucleus sampling,
	// where the model considers the results of the tokens with top_p probability mass.
	// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
	// We generally recommend altering this or temperature but not both.
	TopP *float64 `json:"top_p,omitempty"`

	// TopLogProbs enables token-level logprobs for the top N tokens (0..20).
	// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.
	TopLogProbs int `json:"top_logprobs,omitempty"`

	// Text holds text-specific configuration (e.g. formatting / structured output).
	// Its exact schema depends on the model and selected features; keep it generic.
	Text any `json:"text,omitempty"`

	// Truncation controls what happens if the input would exceed the model context window.
	// Typical values: "auto" or "disabled".
	Truncation string `json:"truncation,omitempty"`

	// ─────────────────────────────────────────────────────────────
	// Tools & function calling
	// ─────────────────────────────────────────────────────────────

	// Tools defines built-in tools (web_search, file_search, code_interpreter, ...) and/or
	// custom tools (function calling). This is intentionally `[]any` to support all tool types.
	Tools []any `json:"tools,omitempty"`

	// ToolChoice controls how tools are selected ("auto", "required", "none") or can be an
	// object forcing a specific tool (e.g. {"type":"function","name":"get_weather"}).
	ToolChoice any `json:"tool_choice,omitempty"`

	// ParallelToolCalls controls whether the model may call multiple tools in parallel.
	// This is a pointer because the API defaults to true and callers may need to send false explicitly.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	// MaxToolCalls limits the number of tool calls allowed for this response.
	MaxToolCalls int `json:"max_tool_calls,omitempty"`

	// ─────────────────────────────────────────────────────────────
	// Conversation state, prompts, caching
	// ─────────────────────────────────────────────────────────────

	// PreviousResponseID continues from a prior response (stateful interactions).
	PreviousResponseID string `json:"previous_response_id,omitempty"`

	// Conversation can be a conversation ID or an object. When set, input items are
	// prepended with existing conversation items and the response is added to that conversation.
	Conversation any `json:"conversation,omitempty"`

	// Prompt holds prompt-template configuration (prompt IDs, variables, ...).
	Prompt any `json:"prompt,omitempty"`

	// PromptCacheKey helps improve prompt caching hit rates.
	PromptCacheKey string `json:"prompt_cache_key,omitempty"`

	// PromptCacheRetention controls the prompt caching retention window.
	PromptCacheRetention string `json:"prompt_cache_retention,omitempty"`

	// Store controls whether the response is stored. Pointer allows explicitly sending false.
	Store *bool `json:"store,omitempty"`

	// ServiceTier controls the service tier used for this request.
	ServiceTier string `json:"service_tier,omitempty"`

	// ─────────────────────────────────────────────────────────────
	// Reasoning, safety, metadata, streaming extras
	// ─────────────────────────────────────────────────────────────

	// Background runs the response in background mode.
	Background bool `json:"background,omitempty"`

	// Reasoning holds reasoning-specific configuration (effort, summary, ...).
	Reasoning any `json:"reasoning,omitempty"`

	// SafetyIdentifier is a stable identifier for your end-users.
	SafetyIdentifier string `json:"safety_identifier,omitempty"`

	// Metadata is a free-form object for request metadata.
	Metadata map[string]any `json:"metadata,omitempty"`

	// Include requests additional fields to be included in the response.
	Include []string `json:"include,omitempty"`

	// StreamOptions configures streaming behaviour (e.g. include usage on the final event).
	StreamOptions any `json:"stream_options,omitempty"`

	// User is deprecated by OpenAI but kept for compatibility.
	User string `json:"user,omitempty"`

	// Non serializable
	ctx       context.Context
	splitFunc bufio.SplitFunc

	// Listeners
	mu        sync.Mutex
	listeners map[EventType]func(e textual.JsonGenericCarrier[StreamEvent]) textual.StringCarrier
	observers map[EventType]func(e textual.JsonGenericCarrier[StreamEvent])

	// Function calling delegate (non-serializable).
	// Initialized lazily when the first function tool is registered.
	functionTools                 map[string]registeredFunctionTool
	functionCalls                 map[string]*functionCallState
	functionCallOutputs           []FunctionCallOutputItem
	functionCallOutputIndexByCall map[string]int
	functionCallObserver          FunctionCallObserver
}

type registeredFunctionTool struct {
	Tool    FunctionTool
	Handler JSONFunction
}

type functionCallState struct {
	ItemID      string
	CallID      string
	Name        string
	OutputIndex int
	Args        strings.Builder
	Done        bool
}

type outputItemEnvelope struct {
	Type      string `json:"type,omitempty"`
	ID        string `json:"id,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

func NewResponsesRequest(ctx context.Context, model models.Model) *ResponsesRequest {
	return &ResponsesRequest{
		ctx:             ctx,
		splitFunc:       textual.ScanJSON,
		Model:           model.ID,
		Input:           nil,
		Stream:          true,
		MaxOutputTokens: 0,
		Thinking:        false,
		listeners:       make(map[EventType]func(e textual.JsonGenericCarrier[StreamEvent]) textual.StringCarrier),
		observers:       make(map[EventType]func(e textual.JsonGenericCarrier[StreamEvent])),
	}
}

func (r *ResponsesRequest) Context() context.Context {
	return r.ctx
}

func (r *ResponsesRequest) URL(baseURL string) (string, error) {
	if strings.TrimSpace(baseURL) == "" {
		return "", errors.New("textualopenai: missing OpenAI base URL")
	}
	u, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("textualopenai: invalid base URL: %w", err)
	}
	basePath := strings.TrimSuffix(u.Path, "/")
	u.Path = basePath + "/responses"
	return u.String(), nil
}

func (r *ResponsesRequest) Validate() error {
	if r.Stream == false {
		return errors.New("textualopenai: streaming must be enabled")
	}

	// In the public API, `input` is optional because callers can also rely on conversation,
	// prompt templates, or previous_response_id. This library keeps a minimal validation
	// layer and only checks that at least one "source of context" is present.
	if r.Input == nil && strings.TrimSpace(r.PreviousResponseID) == "" && !r.conversationProvided() && r.Prompt == nil {
		return errors.New("textualopenai: input is required (or provide previous_response_id, conversation, or prompt)")
	}

	// According to the API, previous_response_id and conversation are mutually exclusive.
	if strings.TrimSpace(r.PreviousResponseID) != "" && r.conversationProvided() {
		return errors.New("textualopenai: previous_response_id cannot be used with conversation")
	}

	return nil
}

func (r *ResponsesRequest) conversationProvided() bool {
	if r.Conversation == nil {
		return false
	}
	switch v := r.Conversation.(type) {
	case string:
		return strings.TrimSpace(v) != ""
	default:
		// Any non-nil value is considered provided for object forms.
		return true
	}
}

func (r *ResponsesRequest) SplitFunc() bufio.SplitFunc {
	return r.splitFunc
}

func (r *ResponsesRequest) AddListeners(f func(e textual.JsonGenericCarrier[StreamEvent]) textual.StringCarrier, et ...EventType) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, t := range et {
		if _, ok := r.listeners[t]; ok {
			return errors.New("duplicate listener call back")
		}
		r.listeners[t] = f
	}
	return nil
}

func (r *ResponsesRequest) RemoveListener(et EventType) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.listeners[et]; ok {
		delete(r.listeners, et)
		return nil
	}
	return errors.New(fmt.Sprintf("listener %s not found", et))
}

func (r *ResponsesRequest) RemoveListeners() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for eventName := range r.listeners {
		delete(r.listeners, eventName)
	}
}

func (r *ResponsesRequest) AddObservers(f func(e textual.JsonGenericCarrier[StreamEvent]), et ...EventType) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, eventName := range et {
		if _, ok := r.observers[eventName]; ok {
			return errors.New("duplicate observer call back")
		}
		r.observers[eventName] = f
	}
	return nil
}

func (r *ResponsesRequest) RemoveObserver(et EventType) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.observers[et]; ok {
		delete(r.observers, et)
		return nil
	}
	return errors.New(fmt.Sprintf("observer %s not found", et))
}

func (r *ResponsesRequest) RemoveObservers() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for eventName := range r.observers {
		delete(r.observers, eventName)
	}
}

// Transcoder returns a Transcoder that execute observation logic, and emits StreamEvent that have listeners.
func (r *ResponsesRequest) Transcoder() textual.TranscoderFunc[textual.JsonGenericCarrier[StreamEvent], textual.StringCarrier] {
	return func(ctx context.Context, in <-chan textual.JsonGenericCarrier[StreamEvent]) <-chan textual.StringCarrier {
		return textual.AsyncEmitter(ctx, in, func(ctx context.Context, c textual.JsonGenericCarrier[StreamEvent], emit func(s textual.StringCarrier)) {
			ev := c.Value

			// Built-in delegate: handle function calling support.
			// This runs before user observers/listeners so the request state is up to date
			// when they receive the event.
			r.processFunctionCalling(ctx, ev)

			// Snapshot callbacks under lock, then call them outside the lock.
			r.mu.Lock()
			observerFunc := r.observers[ev.Type]
			if observerFunc == nil {
				observerFunc = r.observers[AllEvent]
			}
			listenerFunc := r.listeners[ev.Type]
			if listenerFunc == nil {
				listenerFunc = r.listeners[AllEvent]
			}
			r.mu.Unlock()

			// the StreamEvent is unaltered.
			if observerFunc != nil {
				observerFunc(c)
			}

			// The StreamEvent will be processed: we emit the result of the listener function.
			if listenerFunc != nil {
				emit(listenerFunc(c))
			}
		})
	}
}

/////////////////////////////////////
// Tools support
/////////////////////////////////////

// Usage sample :
// req:= textualopenai.NewResponsesRequest(ctx, textualopenai.MODEL_GPT_4_1)
// req.Input = "What's the weather in Paris?"
//
// schema:= map[string]any{
// 	"type": "object",
// 	"properties": map[string]any{
// 		"location": map[string]any{"type": "string"},
// 	},
// 	"required":             []string{"location"},
// 	"additionalProperties": false,
// }
//
// _ = req.RegisterFunctionTool("get_weather", "Get current weather", schema,
// 	func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
// 		// args is raw JSON like: {"location":"Paris"}
// 		// return raw JSON (must be valid JSON bytes)
// 		return json.RawMessage(`{"temp_c":12,"condition":"cloudy"}`), nil
// 	},
// )
//
// // Optional: observe executed calls and grab tool outputs to send in the next request.
// req.SetFunctionCallObserver(func(ctx context.Context, call textualopenai.FunctionCall, out *textualopenai.FunctionCallOutputItem, err error) {
// 	// out is ready to be appended to a follow-up request input as a "function_call_output" item.
// })
//

// RegisterFunctionToolStrict is the same as RegisterFunctionTool but allows setting `strict`
// on the tool definition when supported by the API/model.
func (r *ResponsesRequest) RegisterFunctionToolStrict(name, description string, parameters any, strict bool, fn JSONFunction) error {
	return r.registerFunctionTool(FunctionTool{
		Type:        "function",
		Name:        name,
		Description: description,
		Parameters:  parameters,
		Strict:      BoolPtr(strict),
	}, fn)
}

// RegisterFunctionToolTyped registers a custom "function" tool whose handler uses typed Go
// arguments and results. JSON marshaling/unmarshaling is handled internally using the standard
// library only (encoding/json).
func RegisterFunctionToolTyped[A any, R any](r *ResponsesRequest, name, description string, parameters any, fn func(context.Context, A) (R, error)) error {
	wrapped := func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
		if len(args) == 0 {
			args = json.RawMessage(`{}`)
		}
		var a A
		if err := json.Unmarshal(args, &a); err != nil {
			return nil, err
		}

		res, err := fn(ctx, a)
		if err != nil {
			return nil, err
		}

		b, err := json.Marshal(res)
		if err != nil {
			return nil, err
		}
		return json.RawMessage(b), nil
	}
	return r.RegisterFunctionTool(name, description, parameters, wrapped)
}

// UnregisterFunctionTool removes a previously registered function tool and
// removes its definition from the `Tools` request field (when present).
func (r *ResponsesRequest) UnregisterFunctionTool(name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return errors.New("textualopenai: function tool name is required")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.functionTools == nil {
		return fmt.Errorf("textualopenai: function tool not found: %s", name)
	}
	if _, ok := r.functionTools[name]; !ok {
		return fmt.Errorf("textualopenai: function tool not found: %s", name)
	}

	delete(r.functionTools, name)

	if len(r.Tools) == 0 {
		return nil
	}

	// Remove any matching function tools from the request tools slice.
	filtered := r.Tools[:0]
	for _, t := range r.Tools {
		if isSameFunctionTool(t, name) {
			continue
		}
		filtered = append(filtered, t)
	}
	r.Tools = filtered
	return nil
}

func (r *ResponsesRequest) ensureFunctionDelegateLocked() {
	if r.functionTools == nil {
		r.functionTools = make(map[string]registeredFunctionTool)
	}
	if r.functionCalls == nil {
		r.functionCalls = make(map[string]*functionCallState)
	}
	if r.functionCallOutputIndexByCall == nil {
		r.functionCallOutputIndexByCall = make(map[string]int)
	}
}

func (r *ResponsesRequest) SetFunctionCallObserver(f FunctionCallObserver) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ensureFunctionDelegateLocked()
	r.functionCallObserver = f
}

// RegisterFunctionTool registers a custom "function" tool for the Responses API and
// binds it to a JSON-only handler.
//
// - name: tool name (must match the function call name emitted by the model)
// - description: natural language description for the model
// - parameters: JSON Schema object (use map[string]any or a struct that marshals to the schema)
// - fn: handler invoked with raw JSON arguments, returns raw JSON output
func (r *ResponsesRequest) RegisterFunctionTool(name, description string, parameters any, fn JSONFunction) error {
	return r.registerFunctionTool(FunctionTool{
		Type:        "function",
		Name:        name,
		Description: description,
		Parameters:  parameters,
	}, fn)
}

// FunctionCallOutputs returns the tool output items produced by the embedded delegate.
// The returned slice is a copy and safe to modify.
func (r *ResponsesRequest) FunctionCallOutputs() []FunctionCallOutputItem {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.functionCallOutputs) == 0 {
		return nil
	}
	out := make([]FunctionCallOutputItem, len(r.functionCallOutputs))
	copy(out, r.functionCallOutputs)
	return out
}

// ClearFunctionCallOutputs clears the collected tool output items.
func (r *ResponsesRequest) ClearFunctionCallOutputs() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.functionCallOutputs = nil
	if r.functionCallOutputIndexByCall != nil {
		for k := range r.functionCallOutputIndexByCall {
			delete(r.functionCallOutputIndexByCall, k)
		}
	}
}

func (r *ResponsesRequest) registerFunctionTool(tool FunctionTool, fn JSONFunction) error {
	if r == nil {
		return errors.New("textualopenai: nil ResponsesRequest")
	}
	name := strings.TrimSpace(tool.Name)
	if name == "" {
		return errors.New("textualopenai: function tool name is required")
	}
	if fn == nil {
		return errors.New("textualopenai: function tool handler is required")
	}

	tool.Type = "function"
	tool.Name = name

	r.mu.Lock()
	defer r.mu.Unlock()

	r.ensureFunctionDelegateLocked()

	if _, exists := r.functionTools[name]; exists {
		return fmt.Errorf("textualopenai: function tool already registered: %s", name)
	}

	r.functionTools[name] = registeredFunctionTool{
		Tool:    tool,
		Handler: fn,
	}

	r.upsertFunctionToolIntoRequestLocked(tool)
	return nil
}

func (r *ResponsesRequest) upsertFunctionToolIntoRequestLocked(tool FunctionTool) {
	if r.Tools == nil {
		r.Tools = make([]any, 0, 1)
	}
	for i, t := range r.Tools {
		if isSameFunctionTool(t, tool.Name) {
			r.Tools[i] = tool
			return
		}
	}
	r.Tools = append(r.Tools, tool)
}

func isSameFunctionTool(t any, name string) bool {
	switch v := t.(type) {
	case FunctionTool:
		return v.Type == "function" && v.Name == name
	case *FunctionTool:
		return v != nil && v.Type == "function" && v.Name == name
	case map[string]any:
		typ, _ := v["type"].(string)
		nm, _ := v["name"].(string)
		return typ == "function" && nm == name
	default:
		return false
	}
}

// processFunctionCalling inspects streaming events and, when function calling is used,
// automatically executes registered tools and collects their outputs.
//
// This method is intentionally side-effecting and is called internally by ResponsesRequest.Transcoder().
func (r *ResponsesRequest) processFunctionCalling(ctx context.Context, ev StreamEvent) {
	if r == nil {
		return
	}
	// If no tools are registered, skip quickly.
	r.mu.Lock()
	hasTools := len(r.functionTools) > 0
	r.mu.Unlock()
	if !hasTools {
		return
	}

	switch ev.Type {
	case OutputItemAdded:
		r.captureFunctionCallItem(ev)

	case FunctionCallArgumentsDelta:
		r.captureFunctionCallArgumentsDelta(ev)

	case FunctionCallArgumentsDone:
		r.captureFunctionCallArgumentsDoneAndExecute(ctx, ev)
	}
}

func (r *ResponsesRequest) captureFunctionCallItem(ev StreamEvent) {
	if len(ev.Item) == 0 {
		return
	}

	var item outputItemEnvelope
	if err := json.Unmarshal(ev.Item, &item); err != nil {
		return
	}
	if item.Type != "function_call" || strings.TrimSpace(item.ID) == "" {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.ensureFunctionDelegateLocked()

	st, ok := r.functionCalls[item.ID]
	if !ok || st == nil {
		st = &functionCallState{
			ItemID:      item.ID,
			CallID:      item.CallID,
			Name:        item.Name,
			OutputIndex: ev.OutputIndex,
		}
		r.functionCalls[item.ID] = st
		return
	}

	// Update missing fields when needed.
	if st.CallID == "" {
		st.CallID = item.CallID
	}
	if st.Name == "" {
		st.Name = item.Name
	}
	st.OutputIndex = ev.OutputIndex
}

func (r *ResponsesRequest) captureFunctionCallArgumentsDelta(ev StreamEvent) {
	itemID := strings.TrimSpace(ev.ItemID)
	if itemID == "" || ev.Delta == "" {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.ensureFunctionDelegateLocked()

	st, ok := r.functionCalls[itemID]
	if !ok || st == nil {
		st = &functionCallState{
			ItemID:      itemID,
			OutputIndex: ev.OutputIndex,
		}
		r.functionCalls[itemID] = st
	}
	st.Args.WriteString(ev.Delta)
	st.OutputIndex = ev.OutputIndex
}

func (r *ResponsesRequest) captureFunctionCallArgumentsDoneAndExecute(ctx context.Context, ev StreamEvent) {
	itemID := strings.TrimSpace(ev.ItemID)
	name := strings.TrimSpace(ev.Name)
	argsStr := strings.TrimSpace(ev.Arguments)

	var callID string
	outputIndex := ev.OutputIndex

	// Snapshot state and handler under lock.
	var handler JSONFunction
	var observer FunctionCallObserver
	var argsFromState string

	r.mu.Lock()
	r.ensureFunctionDelegateLocked()

	if st, ok := r.functionCalls[itemID]; ok && st != nil {
		callID = st.CallID
		if name == "" {
			name = st.Name
		}
		if argsStr == "" {
			argsFromState = strings.TrimSpace(st.Args.String())
		}
		st.Done = true
	}

	if argsStr == "" && argsFromState != "" {
		argsStr = argsFromState
	}

	if reg, ok := r.functionTools[name]; ok {
		handler = reg.Handler
	}
	observer = r.functionCallObserver
	r.mu.Unlock()

	if name == "" {
		if observer != nil {
			observer(ctx, FunctionCall{
				ItemID:      itemID,
				CallID:      callID,
				Name:        "",
				Arguments:   nil,
				OutputIndex: outputIndex,
			}, nil, errors.New("textualopenai: function call missing name"))
		}
		return
	}

	if handler == nil {
		if observer != nil {
			observer(ctx, FunctionCall{
				ItemID:      itemID,
				CallID:      callID,
				Name:        name,
				Arguments:   json.RawMessage(argsStr),
				OutputIndex: outputIndex,
			}, nil, fmt.Errorf("textualopenai: no handler registered for function: %s", name))
		}
		return
	}

	argsJSON := json.RawMessage(argsStr)
	if len(argsJSON) == 0 {
		argsJSON = json.RawMessage(`{}`)
	}

	outJSON, err := handler(ctx, argsJSON)

	outItem := FunctionCallOutputItem{
		Type:   "function_call_output",
		CallID: callID,
		Output: string(outJSON),
	}

	if err != nil {
		// Still provide a JSON payload the model can interpret as an error.
		b, _ := json.Marshal(map[string]any{"error": err.Error()})
		outItem.Output = string(b)
	} else if strings.TrimSpace(outItem.Output) == "" {
		outItem.Output = "null"
	}

	// Persist the output for the next request (requires call_id).
	if strings.TrimSpace(callID) != "" {
		r.mu.Lock()
		r.ensureFunctionDelegateLocked()

		if idx, ok := r.functionCallOutputIndexByCall[callID]; ok && idx >= 0 && idx < len(r.functionCallOutputs) {
			r.functionCallOutputs[idx] = outItem
		} else {
			r.functionCallOutputIndexByCall[callID] = len(r.functionCallOutputs)
			r.functionCallOutputs = append(r.functionCallOutputs, outItem)
		}

		r.mu.Unlock()
	}

	if observer != nil {
		tmp := outItem
		observer(ctx, FunctionCall{
			ItemID:      itemID,
			CallID:      callID,
			Name:        name,
			Arguments:   argsJSON,
			OutputIndex: outputIndex,
		}, &tmp, err)
	}
}
