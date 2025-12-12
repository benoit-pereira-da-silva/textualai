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

package textualollama

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"text/template"
	"unicode"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
)

// TextualProcessorChatEvent is kept for backward compatibility with earlier
// event naming conventions used elsewhere in the project.
const TextualProcessorChatEvent = "ollama.responses"

// AggregateType controls how streamed chunks are turned into output carrier values.
//
//   - Word: emit when we cross a whitespace / punctuation boundary.
//   - Line: emit when we cross a newline boundary.
type AggregateType string

const (
	Word AggregateType = "word"
	Line AggregateType = "line"
)

// Role is the message role used for the top-level message when ResponseProcessor
// builds a chat-like payload.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
)

// Endpoint selects which Ollama API endpoint is used.
type Endpoint string

const (
	EndpointChat     Endpoint = "chat"     // POST /api/chat
	EndpointGenerate Endpoint = "generate" // POST /api/generate
)

// ResponseProcessor is a textual.Processor that calls Ollama's HTTP API
// (POST /api/chat or POST /api/generate) and re-emits streamed output as
// carrier values.
//
// It intentionally mirrors the style of the OpenAI Responses API processor:
//
//   - It uses a Go text/template to build the prompt from each incoming carrier.
//   - It streams the response, aggregates it (Word or Line), and emits carrier
//     values on the output channel.
//   - It exposes all request options from Ollama's API via chainable With* methods.
//
// Important semantics (same as the OpenAI processor in this repo):
//
//   - If you set an explicit Prompt (generate) or Messages (chat), the incoming
//     carrier text is no longer injected. You fully control the request payload.
//
// Environment:
//
//   - By default, BaseURL is resolved from OLLAMA_HOST (if set) or falls back
//     to http://localhost:11434.
type ResponseProcessor[S textual.Carrier[S]] struct {
	// BaseURL is the base URL of the Ollama server (e.g. "http://localhost:11434").
	BaseURL string `json:"baseURL,omitempty"`

	// Endpoint selects /api/chat or /api/generate.
	Endpoint Endpoint `json:"endpoint,omitempty"`

	// Model is the Ollama model identifier (e.g. "llama3.1", "qwen2.5", ...).
	Model string `json:"model,omitempty"`

	// Template is the prompt template used to build the message/prompt text.
	// It is a Go text/template executed with templateData (Input/Text/Item).
	Template template.Template `json:"template"`

	// AggregateType controls how streamed content is chunked into outputs.
	AggregateType AggregateType `json:"aggregateType"`

	// Role is used when the processor constructs a default chat message.
	Role Role `json:"role,omitempty"`

	// ---------------------------
	// Shared Ollama request fields
	// ---------------------------

	// Stream controls streaming mode (default is true).
	Stream *bool `json:"stream,omitempty"`

	// Format can be "json" or a JSON schema object (Structured Outputs).
	Format any `json:"format,omitempty"`

	// Options controls model/runtime parameters (temperature, top_p, num_ctx, ...).
	Options *ModelOptions `json:"options,omitempty"`

	// KeepAlive controls how long the model stays loaded in memory.
	// The API accepts values like "5m" or 0 (unload immediately).
	KeepAlive any `json:"keep_alive,omitempty"`

	// Think enables/disables thinking for thinking-capable models.
	Think *bool `json:"think,omitempty"`

	// ---------------------------------
	// Chat endpoint request fields
	// ---------------------------------

	// Messages overrides the default chat payload.
	// If nil or empty, the processor will build:
	//   - optional system message (from System),
	//   - one message with role=Role and content=prompt.
	Messages []ChatMessage `json:"messages,omitempty"`

	// Tools declares tools available to the model for tool calling.
	Tools []Tool `json:"tools,omitempty"`

	// ---------------------------------
	// Generate endpoint request fields
	// ---------------------------------

	// Prompt overrides the prompt built from the Go Template.
	Prompt *string `json:"prompt,omitempty"`

	// Suffix is appended after the prompt for /api/generate.
	Suffix *string `json:"suffix,omitempty"`

	// Images is a list of base64-encoded images for multimodal models.
	Images []string `json:"images,omitempty"`

	// System overrides the system prompt (generate) or is inserted as the first
	// system message (chat) when Messages is not explicitly provided.
	System *string `json:"system,omitempty"`

	// ModelTemplate overrides the model prompt template for /api/generate.
	// (This corresponds to the "template" field in Ollama's /api/generate request.)
	ModelTemplate *string `json:"-"`

	// Context is a list of token IDs used to keep short conversational memory.
	// It is marked as deprecated in Ollama's docs, but still supported.
	Context []int `json:"context,omitempty"`

	// Raw disables template/system processing for /api/generate.
	Raw *bool `json:"raw,omitempty"`
}

// ModelOptions is the "options" object accepted by Ollama's API.
//
// The list below covers the options demonstrated in the official API docs.
// Additional/unknown keys can be provided through Extra.
type ModelOptions struct {
	NumKeep          *int     `json:"num_keep,omitempty"`
	Seed             *int     `json:"seed,omitempty"`
	NumPredict       *int     `json:"num_predict,omitempty"`
	TopK             *int     `json:"top_k,omitempty"`
	TopP             *float64 `json:"top_p,omitempty"`
	MinP             *float64 `json:"min_p,omitempty"`
	TypicalP         *float64 `json:"typical_p,omitempty"`
	RepeatLastN      *int     `json:"repeat_last_n,omitempty"`
	Temperature      *float64 `json:"temperature,omitempty"`
	RepeatPenalty    *float64 `json:"repeat_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PenalizeNewline  *bool    `json:"penalize_newline,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	NumA             *bool    `json:"numa,omitempty"`
	NumCtx           *int     `json:"num_ctx,omitempty"`
	NumBatch         *int     `json:"num_batch,omitempty"`
	NumGPU           *int     `json:"num_gpu,omitempty"`
	MainGPU          *int     `json:"main_gpu,omitempty"`
	UseMMap          *bool    `json:"use_mmap,omitempty"`
	NumThread        *int     `json:"num_thread,omitempty"`

	// Extra holds additional Ollama options not explicitly listed above.
	// Extra keys never override explicitly modeled fields.
	Extra map[string]any `json:"-"`
}

// MarshalJSON merges typed fields with Extra.
func (o ModelOptions) MarshalJSON() ([]byte, error) {
	m := map[string]any{}

	// Typed fields first.
	if o.NumKeep != nil {
		m["num_keep"] = *o.NumKeep
	}
	if o.Seed != nil {
		m["seed"] = *o.Seed
	}
	if o.NumPredict != nil {
		m["num_predict"] = *o.NumPredict
	}
	if o.TopK != nil {
		m["top_k"] = *o.TopK
	}
	if o.TopP != nil {
		m["top_p"] = *o.TopP
	}
	if o.MinP != nil {
		m["min_p"] = *o.MinP
	}
	if o.TypicalP != nil {
		m["typical_p"] = *o.TypicalP
	}
	if o.RepeatLastN != nil {
		m["repeat_last_n"] = *o.RepeatLastN
	}
	if o.Temperature != nil {
		m["temperature"] = *o.Temperature
	}
	if o.RepeatPenalty != nil {
		m["repeat_penalty"] = *o.RepeatPenalty
	}
	if o.PresencePenalty != nil {
		m["presence_penalty"] = *o.PresencePenalty
	}
	if o.FrequencyPenalty != nil {
		m["frequency_penalty"] = *o.FrequencyPenalty
	}
	if o.PenalizeNewline != nil {
		m["penalize_newline"] = *o.PenalizeNewline
	}
	if len(o.Stop) > 0 {
		m["stop"] = append([]string(nil), o.Stop...)
	}
	if o.NumA != nil {
		m["numa"] = *o.NumA
	}
	if o.NumCtx != nil {
		m["num_ctx"] = *o.NumCtx
	}
	if o.NumBatch != nil {
		m["num_batch"] = *o.NumBatch
	}
	if o.NumGPU != nil {
		m["num_gpu"] = *o.NumGPU
	}
	if o.MainGPU != nil {
		m["main_gpu"] = *o.MainGPU
	}
	if o.UseMMap != nil {
		m["use_mmap"] = *o.UseMMap
	}
	if o.NumThread != nil {
		m["num_thread"] = *o.NumThread
	}

	// Extra fields last, but never overriding typed fields.
	for k, v := range o.Extra {
		if _, exists := m[k]; exists {
			continue
		}
		m[k] = v
	}

	return json.Marshal(m)
}

// ChatMessage is the message representation used by /api/chat.
type ChatMessage struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`

	// Thinking is populated by thinking-capable models.
	Thinking string `json:"thinking,omitempty"`

	// Images is a list of base64-encoded images for multimodal models.
	Images []string `json:"images,omitempty"`

	// ToolCalls is set when the model wants to call tools.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// ToolName is used for tool results messages.
	ToolName string `json:"tool_name,omitempty"`
}

// Tool is a generic tool declaration for Ollama's /api/chat.
// It mirrors the OpenAI processor's Tool type to ease porting.
//
// Example:
//
//	Tool{
//	  Type: "function",
//	  Function: &FunctionTool{
//	    Name: "get_current_weather",
//	    Description: "Get the current weather for a location.",
//	    Parameters: map[string]any{...},
//	  },
//	}
type Tool struct {
	Type     string        `json:"type"`               // e.g. "function"
	Function *FunctionTool `json:"function,omitempty"` // for Type=="function"

	// Extra allows vendor extensions without changing the struct.
	// Extra keys never override Type or Function.
	Extra map[string]any `json:"-"`
}

func (t Tool) MarshalJSON() ([]byte, error) {
	m := map[string]any{
		"type": t.Type,
	}
	if t.Function != nil {
		m["function"] = t.Function
	}
	for k, v := range t.Extra {
		if k == "type" || k == "function" {
			continue
		}
		m[k] = v
	}
	return json.Marshal(m)
}

// FunctionTool defines a custom function tool.
type FunctionTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"` // JSON Schema
}

// ToolCall is the minimal tool call representation returned by Ollama.
type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string         `json:"name,omitempty"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

// templateData is the context passed to the processor's Template.
type templateData[S any] struct {
	Input string // input.UTF8String()
	Text  string // alias for readability
	Item  S      // the full carrier value
}

// NewResponseProcessor builds a ResponseProcessor from a model name and a template string.
//
// Validation performed:
//   - Ensures model is not empty.
//   - Parses the template string.
//   - Ensures the template references {{.Input}} (or {{ .Input }}) so the
//     incoming text is injected.
//
// Defaults:
//   - BaseURL: from OLLAMA_HOST or http://localhost:11434.
//   - Endpoint: /api/chat.
//   - Stream: true.
//   - AggregateType: Word.
//   - Role: user.
func NewResponseProcessor[S textual.Carrier[S]](model, templateStr string) (*ResponseProcessor[S], error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return nil, fmt.Errorf("model must not be empty")
	}

	tmpl, err := template.New("ollama").Parse(templateStr)
	if err != nil {
		return nil, fmt.Errorf("parse template: %w", err)
	}

	// Ensure there is an injection point for the incoming text.
	if !strings.Contains(templateStr, "{{.Input}}") &&
		!strings.Contains(templateStr, "{{ .Input }}") {
		return nil, fmt.Errorf("template must contain an {{.Input}} placeholder")
	}

	// Defaults: streaming is the whole point of this processor.
	stream := true

	return &ResponseProcessor[S]{
		BaseURL:       defaultBaseURL(),
		Endpoint:      EndpointChat,
		Model:         model,
		Template:      *tmpl,
		AggregateType: Word,
		Role:          RoleUser,
		Stream:        &stream,
	}, nil
}

// Apply implements textual.Processor.
//
// It consumes incoming carrier values, sends each through the Ollama API
// pipeline, and emits zero or more carrier values per input (streamed output).
func (p ResponseProcessor[S]) Apply(ctx context.Context, in <-chan S) <-chan S {
	if ctx == nil {
		ctx = context.Background()
	}

	out := make(chan S)

	go func() {
		defer close(out)

		for {
			select {
			case <-ctx.Done():
				// Stop processing on cancellation and drain upstream so we don't
				// block senders.
				for range in {
				}
				return

			case input, ok := <-in:
				if !ok {
					return
				}

				if err := p.processOne(ctx, input, out); err != nil {
					// Attach the error to the item, keep stream alive.
					errRes := input.WithError(err)
					select {
					case <-ctx.Done():
						return
					case out <- errRes:
					}
				}
			}
		}
	}()

	return out
}

func (p ResponseProcessor[S]) processOne(ctx context.Context, input S, out chan<- S) error {
	prompt, err := p.buildPrompt(input)
	if err != nil {
		return fmt.Errorf("build prompt: %w", err)
	}

	endpointURL, endpoint, reqBody, err := p.buildRequest(prompt)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	if err := p.streamOllama(ctx, endpointURL, endpoint, reqBody, out); err != nil {
		return fmt.Errorf("ollama stream: %w", err)
	}
	return nil
}

func (p ResponseProcessor[S]) buildPrompt(input S) (string, error) {
	// Zero-valued Template has no parse tree; use the plain input text.
	if p.Template.Tree == nil {
		return input.UTF8String(), nil
	}
	var buf bytes.Buffer
	data := templateData[S]{
		Input: input.UTF8String(),
		Text:  input.UTF8String(),
		Item:  input,
	}
	if err := (&p.Template).Execute(&buf, data); err != nil {
		return "", err
	}
	return buf.String(), nil
}

func defaultBaseURL() string {
	raw := strings.TrimSpace(os.Getenv("OLLAMA_HOST"))
	if raw == "" {
		return "http://localhost:11434"
	}
	// Ollama users often set OLLAMA_HOST to "127.0.0.1:11434" (no scheme).
	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}
	raw = strings.TrimRight(raw, "/")
	return raw
}

func normalizeBaseURL(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultBaseURL()
	}
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	s = strings.TrimRight(s, "/")
	return s
}

func (p ResponseProcessor[S]) buildRequest(prompt string) (url string, endpoint Endpoint, body map[string]any, err error) {
	base := normalizeBaseURL(p.BaseURL)

	endpoint = p.Endpoint
	if endpoint != EndpointGenerate {
		endpoint = EndpointChat
	}

	// Model is required by the API.
	model := strings.TrimSpace(p.Model)
	if model == "" {
		return "", endpoint, nil, fmt.Errorf("model must not be empty")
	}

	body = map[string]any{
		"model": model,
	}

	switch endpoint {
	case EndpointChat:
		if len(p.Messages) == 0 {
			msgs := make([]ChatMessage, 0, 2)
			if p.System != nil && strings.TrimSpace(*p.System) != "" {
				msgs = append(msgs, ChatMessage{
					Role:    string(RoleSystem),
					Content: *p.System,
				})
			}

			role := strings.TrimSpace(string(p.Role))
			if role == "" {
				role = string(RoleUser)
			}
			msgs = append(msgs, ChatMessage{
				Role:    role,
				Content: prompt,
			})

			body["messages"] = msgs
		} else {
			dup := make([]ChatMessage, len(p.Messages))
			copy(dup, p.Messages)
			body["messages"] = dup
		}

		if len(p.Tools) > 0 {
			body["tools"] = append([]Tool(nil), p.Tools...)
		}

		url = base + "/api/chat"

	case EndpointGenerate:
		// Default prompt if none is provided: build from template.
		if p.Prompt == nil {
			body["prompt"] = prompt
		} else {
			body["prompt"] = *p.Prompt
		}

		if p.Suffix != nil {
			body["suffix"] = *p.Suffix
		}
		if len(p.Images) > 0 {
			body["images"] = append([]string(nil), p.Images...)
		}
		if p.Format != nil {
			// generate also accepts format; keep it in shared below as well, but
			// we allow it here for clarity. (We'll set it again in shared section;
			// same key, same value.)
			body["format"] = p.Format
		}
		if p.System != nil {
			body["system"] = *p.System
		}
		if p.ModelTemplate != nil {
			body["template"] = *p.ModelTemplate
		}
		if len(p.Context) > 0 {
			dup := make([]int, len(p.Context))
			copy(dup, p.Context)
			body["context"] = dup
		}
		if p.Raw != nil {
			body["raw"] = *p.Raw
		}

		url = base + "/api/generate"
	default:
		return "", endpoint, nil, fmt.Errorf("unsupported endpoint: %s", endpoint)
	}

	// Shared options (apply to both endpoints).
	if p.Think != nil {
		body["think"] = *p.Think
	}
	if p.Format != nil {
		body["format"] = p.Format
	}
	if p.Options != nil {
		body["options"] = p.Options
	}
	if p.Stream != nil {
		body["stream"] = *p.Stream
	}
	if p.KeepAlive != nil {
		body["keep_alive"] = p.KeepAlive
	}

	return url, endpoint, body, nil
}

// ollamaErrorResponse matches the common error payload shape.
type ollamaErrorResponse struct {
	Error string `json:"error,omitempty"`
}

// generateResponseEvent is the minimal subset used to extract streamed deltas.
type generateResponseEvent struct {
	Response   string `json:"response,omitempty"`
	Done       bool   `json:"done,omitempty"`
	DoneReason string `json:"done_reason,omitempty"`
	Error      string `json:"error,omitempty"`
}

// chatResponseEvent is the minimal subset used to extract streamed deltas.
type chatResponseEvent struct {
	Message    ChatMessage `json:"message,omitempty"`
	Done       bool        `json:"done,omitempty"`
	DoneReason string      `json:"done_reason,omitempty"`
	Error      string      `json:"error,omitempty"`
}

// streamOllama performs a streaming POST to Ollama, parses newline-delimited JSON
// events (or a single JSON object when stream=false), aggregates deltas
// according to AggregateType, and emits carrier values on out.
func (p ResponseProcessor[S]) streamOllama(
	ctx context.Context,
	url string,
	endpoint Endpoint,
	reqBody map[string]any,
	out chan<- S,
) error {
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("perform request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		limited, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))

		// Try to surface "error" field if present.
		var er ollamaErrorResponse
		if json.Unmarshal(limited, &er) == nil && strings.TrimSpace(er.Error) != "" {
			return fmt.Errorf("ollama API returned %d: %s", resp.StatusCode, strings.TrimSpace(er.Error))
		}

		return fmt.Errorf("ollama API returned %d: %s", resp.StatusCode, strings.TrimSpace(string(limited)))
	}

	agg := newStreamAggregator(p.AggregateType)
	dec := json.NewDecoder(resp.Body)

	proto := *new(S)

	emitSegments := func(delta string) error {
		if delta == "" {
			return nil
		}
		segments := agg.Append(delta)
		for _, s := range segments {
			res := proto.FromUTF8String(s)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case out <- res:
			}
		}
		return nil
	}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		switch endpoint {
		case EndpointGenerate:
			var ev generateResponseEvent
			if err := dec.Decode(&ev); err != nil {
				if err == io.EOF {
					goto flush
				}
				return fmt.Errorf("decode response: %w", err)
			}
			if strings.TrimSpace(ev.Error) != "" {
				return fmt.Errorf("ollama error: %s", strings.TrimSpace(ev.Error))
			}
			if err := emitSegments(ev.Response); err != nil {
				return err
			}
			if ev.Done {
				goto flush
			}

		case EndpointChat:
			var ev chatResponseEvent
			if err := dec.Decode(&ev); err != nil {
				if err == io.EOF {
					goto flush
				}
				return fmt.Errorf("decode response: %w", err)
			}
			if strings.TrimSpace(ev.Error) != "" {
				return fmt.Errorf("ollama error: %s", strings.TrimSpace(ev.Error))
			}
			// We only stream message.content. Tool calls and thinking are currently
			// not emitted as textual output (they remain available in the raw JSON).
			if err := emitSegments(ev.Message.Content); err != nil {
				return err
			}
			if ev.Done {
				goto flush
			}

		default:
			return fmt.Errorf("unsupported endpoint: %s", endpoint)
		}
	}

flush:
	// Flush any remaining partial text.
	tails := agg.Final()
	for _, s := range tails {
		res := proto.FromUTF8String(s)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case out <- res:
		}
	}
	return nil
}

// -----------------------
// Chainable With* methods
// -----------------------

// WithAggregateType returns a copy with AggregateType updated.
func (p ResponseProcessor[S]) WithAggregateType(t AggregateType) ResponseProcessor[S] {
	p.AggregateType = t
	return p
}

// WithRole sets the role used when the processor constructs its default chat message.
func (p ResponseProcessor[S]) WithRole(role Role) ResponseProcessor[S] {
	p.Role = role
	return p
}

// WithBaseURL sets the Ollama server base URL.
func (p ResponseProcessor[S]) WithBaseURL(baseURL string) ResponseProcessor[S] {
	p.BaseURL = strings.TrimSpace(baseURL)
	return p
}

// WithEndpoint sets which Ollama endpoint is used.
func (p ResponseProcessor[S]) WithEndpoint(endpoint Endpoint) ResponseProcessor[S] {
	p.Endpoint = endpoint
	return p
}

// WithChatEndpoint switches the processor to /api/chat.
func (p ResponseProcessor[S]) WithChatEndpoint() ResponseProcessor[S] {
	p.Endpoint = EndpointChat
	return p
}

// WithGenerateEndpoint switches the processor to /api/generate.
func (p ResponseProcessor[S]) WithGenerateEndpoint() ResponseProcessor[S] {
	p.Endpoint = EndpointGenerate
	return p
}

// WithModel sets the model.
func (p ResponseProcessor[S]) WithModel(model string) ResponseProcessor[S] {
	p.Model = strings.TrimSpace(model)
	return p
}

// WithStream sets stream mode.
func (p ResponseProcessor[S]) WithStream(v bool) ResponseProcessor[S] {
	p.Stream = &v
	return p
}

// WithFormat sets the "format" request option.
//
// Values:
//   - "json" (string), or
//   - a JSON schema object (map / struct) for structured outputs.
func (p ResponseProcessor[S]) WithFormat(format any) ResponseProcessor[S] {
	p.Format = format
	return p
}

// WithFormatJSON sets format="json".
func (p ResponseProcessor[S]) WithFormatJSON() ResponseProcessor[S] {
	p.Format = "json"
	return p
}

// WithOptions sets the "options" object.
func (p ResponseProcessor[S]) WithOptions(opts *ModelOptions) ResponseProcessor[S] {
	p.Options = cloneModelOptions(opts)
	return p
}

// WithOption sets a single "options" key via the Extra map.
//
// This is useful when Ollama adds new options not yet modeled in ModelOptions.
func (p ResponseProcessor[S]) WithOption(key string, value any) ResponseProcessor[S] {
	key = strings.TrimSpace(key)
	if key == "" {
		return p
	}
	p.Options = cloneModelOptions(p.Options)
	if p.Options.Extra == nil {
		p.Options.Extra = make(map[string]any)
	}
	p.Options.Extra[key] = value
	return p
}

// WithKeepAlive sets the keep_alive option (string like "5m" or 0).
func (p ResponseProcessor[S]) WithKeepAlive(v any) ResponseProcessor[S] {
	p.KeepAlive = v
	return p
}

// WithThink sets the think option (for thinking models).
func (p ResponseProcessor[S]) WithThink(v bool) ResponseProcessor[S] {
	p.Think = &v
	return p
}

// WithInstructions sets the system prompt.
//
// This mirrors the OpenAI processor's naming. For Ollama:
//   - /api/generate: sets the "system" field
//   - /api/chat: when Messages is not explicitly provided, inserts a first system message
func (p ResponseProcessor[S]) WithInstructions(system string) ResponseProcessor[S] {
	p.System = &system
	return p
}

// WithSystem is an alias for WithInstructions.
func (p ResponseProcessor[S]) WithSystem(system string) ResponseProcessor[S] {
	return p.WithInstructions(system)
}

// ---- Chat-specific With* ----

// WithMessages overrides the entire messages list for /api/chat.
//
// When Messages is set (non-empty), the processor does not inject incoming text.
// You fully control the prompt/messages.
func (p ResponseProcessor[S]) WithMessages(messages ...ChatMessage) ResponseProcessor[S] {
	p.Messages = append([]ChatMessage(nil), messages...)
	return p
}

// WithTools sets the tools available to the model.
func (p ResponseProcessor[S]) WithTools(tools ...Tool) ResponseProcessor[S] {
	p.Tools = append([]Tool(nil), tools...)
	return p
}

// ---- Generate-specific With* ----

// WithPrompt sets the "prompt" field for /api/generate, overriding the Go Template.
func (p ResponseProcessor[S]) WithPrompt(prompt string) ResponseProcessor[S] {
	p.Prompt = &prompt
	return p
}

// WithSuffix sets the "suffix" field for /api/generate.
func (p ResponseProcessor[S]) WithSuffix(suffix string) ResponseProcessor[S] {
	p.Suffix = &suffix
	return p
}

// WithImages sets the base64-encoded images list.
func (p ResponseProcessor[S]) WithImages(images ...string) ResponseProcessor[S] {
	p.Images = append([]string(nil), images...)
	return p
}

// WithModelTemplate sets the Ollama "template" field for /api/generate.
func (p ResponseProcessor[S]) WithModelTemplate(tmpl string) ResponseProcessor[S] {
	p.ModelTemplate = &tmpl
	return p
}

// WithContext sets the (deprecated) "context" field for /api/generate.
func (p ResponseProcessor[S]) WithContext(tokens ...int) ResponseProcessor[S] {
	p.Context = append([]int(nil), tokens...)
	return p
}

// WithRaw sets the "raw" field for /api/generate.
func (p ResponseProcessor[S]) WithRaw(v bool) ResponseProcessor[S] {
	p.Raw = &v
	return p
}

// ---- Convenience With* for common model options ----

// WithTemperature sets options.temperature.
func (p ResponseProcessor[S]) WithTemperature(v float64) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.Temperature = &v
	return p
}

// WithTopP sets options.top_p.
func (p ResponseProcessor[S]) WithTopP(v float64) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.TopP = &v
	return p
}

// WithTopK sets options.top_k.
func (p ResponseProcessor[S]) WithTopK(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.TopK = &v
	return p
}

// WithMinP sets options.min_p.
func (p ResponseProcessor[S]) WithMinP(v float64) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.MinP = &v
	return p
}

// WithTypicalP sets options.typical_p.
func (p ResponseProcessor[S]) WithTypicalP(v float64) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.TypicalP = &v
	return p
}

// WithSeed sets options.seed.
func (p ResponseProcessor[S]) WithSeed(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.Seed = &v
	return p
}

// WithNumPredict sets options.num_predict.
func (p ResponseProcessor[S]) WithNumPredict(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.NumPredict = &v
	return p
}

// WithNumCtx sets options.num_ctx.
func (p ResponseProcessor[S]) WithNumCtx(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.NumCtx = &v
	return p
}

// WithNumBatch sets options.num_batch.
func (p ResponseProcessor[S]) WithNumBatch(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.NumBatch = &v
	return p
}

// WithNumGPU sets options.num_gpu.
func (p ResponseProcessor[S]) WithNumGPU(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.NumGPU = &v
	return p
}

// WithMainGPU sets options.main_gpu.
func (p ResponseProcessor[S]) WithMainGPU(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.MainGPU = &v
	return p
}

// WithUseMMap sets options.use_mmap.
func (p ResponseProcessor[S]) WithUseMMap(v bool) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.UseMMap = &v
	return p
}

// WithNumThread sets options.num_thread.
func (p ResponseProcessor[S]) WithNumThread(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.NumThread = &v
	return p
}

// WithRepeatLastN sets options.repeat_last_n.
func (p ResponseProcessor[S]) WithRepeatLastN(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.RepeatLastN = &v
	return p
}

// WithRepeatPenalty sets options.repeat_penalty.
func (p ResponseProcessor[S]) WithRepeatPenalty(v float64) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.RepeatPenalty = &v
	return p
}

// WithPresencePenalty sets options.presence_penalty.
func (p ResponseProcessor[S]) WithPresencePenalty(v float64) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.PresencePenalty = &v
	return p
}

// WithFrequencyPenalty sets options.frequency_penalty.
func (p ResponseProcessor[S]) WithFrequencyPenalty(v float64) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.FrequencyPenalty = &v
	return p
}

// WithPenalizeNewline sets options.penalize_newline.
func (p ResponseProcessor[S]) WithPenalizeNewline(v bool) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.PenalizeNewline = &v
	return p
}

// WithStop sets options.stop.
func (p ResponseProcessor[S]) WithStop(stops ...string) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.Stop = append([]string(nil), stops...)
	return p
}

// WithNumKeep sets options.num_keep.
func (p ResponseProcessor[S]) WithNumKeep(v int) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.NumKeep = &v
	return p
}

// WithNumA sets options.numa.
func (p ResponseProcessor[S]) WithNumA(v bool) ResponseProcessor[S] {
	p.Options = cloneModelOptions(p.Options)
	p.Options.NumA = &v
	return p
}

// cloneModelOptions ensures a non-nil options struct and attempts to avoid
// sharing the same pointer instance across copies.
func cloneModelOptions(in *ModelOptions) *ModelOptions {
	if in == nil {
		return &ModelOptions{}
	}
	dup := *in
	if in.Stop != nil {
		dup.Stop = append([]string(nil), in.Stop...)
	}
	if in.Extra != nil {
		dup.Extra = make(map[string]any, len(in.Extra))
		for k, v := range in.Extra {
			dup.Extra[k] = v
		}
	}
	return &dup
}

// --------------------------
// Stream aggregation helpers
// --------------------------

type streamAggregator struct {
	aggType     AggregateType
	buffer      []rune
	lastEmitPos int
}

func newStreamAggregator(aggType AggregateType) *streamAggregator {
	if aggType != Word && aggType != Line {
		aggType = Word
	}
	return &streamAggregator{
		aggType:     aggType,
		buffer:      make([]rune, 0),
		lastEmitPos: 0,
	}
}

func (a *streamAggregator) Append(chunk string) []string {
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

func (a *streamAggregator) Final() []string {
	if len(a.buffer) == 0 || a.lastEmitPos >= len(a.buffer) {
		return nil
	}
	a.lastEmitPos = len(a.buffer)
	return []string{string(a.buffer)}
}

func (a *streamAggregator) collect(delim func(rune) bool) []string {
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
