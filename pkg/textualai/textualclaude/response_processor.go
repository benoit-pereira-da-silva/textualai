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

package textualclaude

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"text/template"
	"unicode"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
)

var apiKey = os.Getenv("ANTHROPIC_API_KEY")

// TextualProcessorChatEvent is kept for backward compatibility with earlier
// event naming conventions used elsewhere in the project.
const TextualProcessorChatEvent = "claude.messages"

// AggregateType controls how streamed chunks are turned into output carrier values.
//
//   - Word: emit when we cross a whitespace / punctuation boundary.
//   - Line: emit when we cross a newline boundary.
type AggregateType string

const (
	Word AggregateType = "word"
	Line AggregateType = "line"
)

// Role is the message role used when ResponseProcessor builds a default
// messages array.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system" // Not a Claude messages role; used only as a convenience label.
)

// ResponseProcessor is a textual.Processor that calls Anthropic's Claude Messages API
// (POST /v1/messages) with streaming enabled and re-emits the streamed
// content as carrier values.
//
// It intentionally mirrors the style of the OpenAI, Gemini, Mistral, and Ollama
// processors in this repository:
//
//   - It uses a Go text/template to build the prompt from each incoming carrier.
//   - It streams the response (SSE), aggregates it (Word or Line), and emits
//     carrier values on the output channel.
//   - It exposes common request options via chainable With* methods.
//
// Important semantics (same as the other processors in this repo):
//
//   - If you set an explicit Messages list (WithMessages), the incoming carrier
//     text is no longer injected. You fully control the request payload.
//
// Environment:
//
//   - ANTHROPIC_API_KEY is required.
//   - BaseURL defaults to ANTHROPIC_BASE_URL (if set) or https://api.anthropic.com.
//   - APIVersion defaults to ANTHROPIC_VERSION (if set) or 2023-06-01.
type ResponseProcessor[S textual.Carrier[S]] struct {
	// BaseURL is the Anthropic API base URL (default: https://api.anthropic.com).
	BaseURL string `json:"baseURL,omitempty"`

	// APIVersion is sent via the `anthropic-version` header (default: 2023-06-01).
	APIVersion string `json:"apiVersion,omitempty"`

	// Model is the Claude model identifier (e.g. "claude-3-5-sonnet-latest").
	Model string `json:"model,omitempty"`

	// Template is the prompt template used to build the message content.
	// It is a Go text/template executed with templateData (Input/Text/Item).
	Template template.Template `json:"template"`

	// AggregateType controls how streamed content is chunked into outputs.
	AggregateType AggregateType `json:"aggregateType"`

	// Role is used when the processor constructs a default message.
	Role Role `json:"role,omitempty"`

	// ---------------------------
	// Messages API request body
	// ---------------------------

	// System sets the top-level system prompt.
	// In the Claude Messages API, `system` is not part of the messages array.
	System *string `json:"system,omitempty"`

	// Messages overrides the default request messages.
	// If nil or empty, the processor will build a single message with role=Role and content=prompt.
	Messages []Message `json:"messages,omitempty"`

	// MaxTokens sets max_tokens (required by the API).
	MaxTokens *int `json:"max_tokens,omitempty"`

	// Temperature sets temperature (optional).
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP sets top_p (optional).
	TopP *float64 `json:"top_p,omitempty"`

	// TopK sets top_k (optional).
	TopK *int `json:"top_k,omitempty"`

	// StopSequences sets stop_sequences.
	StopSequences []string `json:"stop_sequences,omitempty"`

	// Metadata sets the top-level metadata object.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Stream controls SSE streaming mode (default: true).
	Stream *bool `json:"stream,omitempty"`

	// Tools declares tools available to the model for tool use.
	Tools []Tool `json:"tools,omitempty"`

	// ToolChoice controls tool selection behavior (string or object).
	ToolChoice any `json:"tool_choice,omitempty"`

	// EmitToolUse controls whether tool_use blocks are surfaced as textual output.
	//
	// When enabled, the processor emits:
	//   - tool_use input JSON deltas (streaming input_json_delta.partial_json)
	//   - tool_use input JSON (non-streaming tool_use.input)
	//
	// This is useful to implement schema-based structured outputs by forcing a tool call
	// whose input_schema matches the desired output schema.
	EmitToolUse *bool `json:"-"`
}

// Message is the Claude Messages API message representation.
//
// In the Messages API, message content can be:
//   - a string (legacy/simple usage), or
//   - an array of content blocks.
//
// We model content as `any` so callers can supply either.
type Message struct {
	Role    string `json:"role,omitempty"`
	Content any    `json:"content,omitempty"`
}

// Tool declares a tool for Claude tool use.
//
// See Anthropic tool use docs for details.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
}

// ToolChoice is a common object form for tool_choice.
//
// Example:
//
//	ToolChoice{Type: "tool", Name: "response"}
type ToolChoice struct {
	Type string `json:"type,omitempty"` // e.g. "tool", "auto", "any"
	Name string `json:"name,omitempty"` // tool name when Type=="tool"
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
//   - Ensures ANTHROPIC_API_KEY looks non-empty (length >= 10).
//   - Ensures model is not empty.
//   - Parses the template string.
//   - Ensures the template references {{.Input}} (or {{ .Input }}) so the
//     incoming text is injected.
//
// Defaults:
//   - BaseURL: from ANTHROPIC_BASE_URL or https://api.anthropic.com.
//   - APIVersion: from ANTHROPIC_VERSION or 2023-06-01.
//   - Stream: true.
//   - AggregateType: Word.
//   - Role: user.
//   - MaxTokens: 1024.
func NewResponseProcessor[S textual.Carrier[S]](model, templateStr string) (*ResponseProcessor[S], error) {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return nil, fmt.Errorf("invalid or missing ANTHROPIC_API_KEY")
	}

	model = strings.TrimSpace(model)
	if model == "" {
		return nil, fmt.Errorf("model must not be empty")
	}

	tmpl, err := template.New("claude.messages").Parse(templateStr)
	if err != nil {
		return nil, fmt.Errorf("parse template: %w", err)
	}

	// Ensure there is an injection point for the incoming text.
	if !strings.Contains(templateStr, "{{.Input}}") &&
		!strings.Contains(templateStr, "{{ .Input }}") {
		return nil, fmt.Errorf("template must contain an {{.Input}} placeholder")
	}

	stream := true
	maxTokens := 1024

	return &ResponseProcessor[S]{
		BaseURL:       defaultBaseURL(),
		APIVersion:    defaultAPIVersion(),
		Model:         model,
		Template:      *tmpl,
		AggregateType: Word,
		Role:          RoleUser,
		Stream:        &stream,
		MaxTokens:     &maxTokens,
	}, nil
}

// Apply implements textual.Processor.
//
// It consumes incoming carrier values, sends each through the Claude Messages API
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

	streaming, reqBody, err := p.buildRequestBody(prompt)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	if err := p.callMessagesAPI(ctx, streaming, reqBody, out); err != nil {
		return fmt.Errorf("claude call: %w", err)
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
	raw := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL"))
	if raw == "" {
		return "https://api.anthropic.com"
	}
	if !strings.Contains(raw, "://") {
		raw = "https://" + raw
	}
	raw = strings.TrimRight(raw, "/")
	return raw
}

func defaultAPIVersion() string {
	raw := strings.TrimSpace(os.Getenv("ANTHROPIC_VERSION"))
	if raw == "" {
		return "2023-06-01"
	}
	return raw
}

func normalizeBaseURL(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultBaseURL()
	}
	if !strings.Contains(s, "://") {
		s = "https://" + s
	}
	s = strings.TrimRight(s, "/")
	return s
}

func normalizeAPIVersion(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultAPIVersion()
	}
	return s
}

func (p ResponseProcessor[S]) buildRequestBody(prompt string) (streaming bool, body map[string]any, err error) {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return false, nil, fmt.Errorf("invalid or missing ANTHROPIC_API_KEY")
	}

	model := strings.TrimSpace(p.Model)
	if model == "" {
		return false, nil, fmt.Errorf("model must not be empty")
	}

	maxTokens := 0
	if p.MaxTokens != nil {
		maxTokens = *p.MaxTokens
	}
	if maxTokens <= 0 {
		return false, nil, fmt.Errorf("max_tokens must be > 0")
	}

	// Stream default.
	streaming = true
	if p.Stream != nil {
		streaming = *p.Stream
	}

	body = map[string]any{
		"model":      model,
		"max_tokens": maxTokens,
	}

	// System.
	if p.System != nil && strings.TrimSpace(*p.System) != "" {
		body["system"] = *p.System
	}

	// Messages.
	if len(p.Messages) == 0 {
		role := strings.TrimSpace(string(p.Role))
		// Claude only understands "user" and "assistant" roles.
		// If callers set RoleSystem (or any invalid role), fall back to user.
		if role != string(RoleUser) && role != string(RoleAssistant) {
			role = string(RoleUser)
		}
		body["messages"] = []Message{{
			Role:    role,
			Content: prompt,
		}}
	} else {
		dup := make([]Message, len(p.Messages))
		copy(dup, p.Messages)
		body["messages"] = dup
	}

	// Optional sampling controls.
	if p.Temperature != nil {
		body["temperature"] = *p.Temperature
	}
	if p.TopP != nil {
		body["top_p"] = *p.TopP
	}
	if p.TopK != nil {
		body["top_k"] = *p.TopK
	}

	// Stop sequences.
	if len(p.StopSequences) > 0 {
		body["stop_sequences"] = append([]string(nil), p.StopSequences...)
	}

	// Metadata.
	if p.Metadata != nil {
		dup := make(map[string]string, len(p.Metadata))
		for k, v := range p.Metadata {
			dup[k] = v
		}
		body["metadata"] = dup
	}

	// Tools.
	if len(p.Tools) > 0 {
		body["tools"] = append([]Tool(nil), p.Tools...)
	}
	if p.ToolChoice != nil {
		body["tool_choice"] = p.ToolChoice
	}

	// Streaming.
	if p.Stream != nil {
		body["stream"] = streaming
	}

	return streaming, body, nil
}

// ----------------------
// Streaming (SSE) parsing
// ----------------------

// anthropicErrorResponse matches a common error payload shape.
// The Messages API typically returns {"type":"error","error":{...}}.
type anthropicErrorResponse struct {
	Type  string `json:"type,omitempty"`
	Error struct {
		Type    string `json:"type,omitempty"`
		Message string `json:"message,omitempty"`
	} `json:"error,omitempty"`
}

// messagesResponse is a minimal subset of the non-streaming response.
type messagesResponse struct {
	Content []struct {
		Type  string                 `json:"type,omitempty"`
		Text  string                 `json:"text,omitempty"`
		Name  string                 `json:"name,omitempty"`
		Input map[string]any         `json:"input,omitempty"`
		Extra map[string]any         `json:"-"`
		Raw   map[string]interface{} `json:"-"`
	} `json:"content,omitempty"`
	StopReason string `json:"stop_reason,omitempty"`
	Error      any    `json:"error,omitempty"`
}

// messagesStreamEvent is a minimal union of streaming event shapes.
// We keep it permissive to tolerate schema changes and additional fields.
type messagesStreamEvent struct {
	Type string `json:"type,omitempty"`

	// content_block_delta
	Delta struct {
		Type        string `json:"type,omitempty"`         // "text_delta" | "input_json_delta" | ...
		Text        string `json:"text,omitempty"`         // for text_delta
		PartialJSON string `json:"partial_json,omitempty"` // for input_json_delta
	} `json:"delta,omitempty"`

	// error
	Error struct {
		Type    string `json:"type,omitempty"`
		Message string `json:"message,omitempty"`
	} `json:"error,omitempty"`
}

func (p ResponseProcessor[S]) callMessagesAPI(
	ctx context.Context,
	streaming bool,
	reqBody map[string]any,
	out chan<- S,
) error {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return fmt.Errorf("invalid or missing ANTHROPIC_API_KEY")
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	base := normalizeBaseURL(p.BaseURL)
	url := base + "/v1/messages"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", strings.TrimSpace(apiKey))
	req.Header.Set("anthropic-version", normalizeAPIVersion(p.APIVersion))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("perform request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		limited, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))

		// Attempt to surface structured error messages.
		var er anthropicErrorResponse
		if json.Unmarshal(limited, &er) == nil && strings.TrimSpace(er.Error.Message) != "" {
			return fmt.Errorf("claude API returned %d: %s", resp.StatusCode, strings.TrimSpace(er.Error.Message))
		}

		return fmt.Errorf("claude API returned %d: %s", resp.StatusCode, strings.TrimSpace(string(limited)))
	}

	if !streaming {
		return p.handleNonStreaming(ctx, resp.Body, out)
	}
	return p.handleStreamingSSE(ctx, resp.Body, out)
}

func (p ResponseProcessor[S]) handleNonStreaming(ctx context.Context, r io.Reader, out chan<- S) error {
	limited, err := io.ReadAll(io.LimitReader(r, 32*1024*1024))
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	// Try structured error envelope.
	var er anthropicErrorResponse
	if json.Unmarshal(limited, &er) == nil && strings.TrimSpace(er.Error.Message) != "" {
		return fmt.Errorf("claude error: %s", strings.TrimSpace(er.Error.Message))
	}

	var res messagesResponse
	if err := json.Unmarshal(limited, &res); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	text := extractText(res)
	if text == "" {
		// Optionally emit tool_use blocks (JSON string) for structured outputs.
		if p.emitToolUseEnabled() {
			toolJSON := extractToolUseInputJSON(res)
			text = toolJSON
		}
	}
	if text == "" {
		return nil
	}

	agg := newStreamAggregator(p.AggregateType)
	segments := agg.Append(text)
	segments = append(segments, agg.Final()...)

	proto := *new(S)
	for _, s := range segments {
		if s == "" {
			continue
		}
		item := proto.FromUTF8String(s)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case out <- item:
		}
	}

	return nil
}

func (p ResponseProcessor[S]) handleStreamingSSE(ctx context.Context, r io.Reader, out chan<- S) error {
	scanner := bufio.NewScanner(r)

	// Allow reasonably large SSE lines.
	const maxScanTokenSize = 1024 * 1024 // 1 MiB
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, maxScanTokenSize)

	aggText := newStreamAggregator(p.AggregateType)
	aggTool := newStreamAggregator(p.AggregateType)
	proto := *new(S)

	emit := func(agg *streamAggregator, delta string) error {
		if delta == "" {
			return nil
		}
		segments := agg.Append(delta)
		for _, s := range segments {
			item := proto.FromUTF8String(s)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case out <- item:
			}
		}
		return nil
	}

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			// Ignore SSE fields like "event:".
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		// Some SSE implementations send [DONE]. Tolerate it.
		if data == "[DONE]" {
			break
		}

		var ev messagesStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			// Skip malformed events but keep streaming.
			continue
		}

		// Error events.
		if strings.EqualFold(strings.TrimSpace(ev.Type), "error") || strings.TrimSpace(ev.Error.Message) != "" {
			msg := strings.TrimSpace(ev.Error.Message)
			if msg == "" {
				msg = data
			}
			return fmt.Errorf("claude stream error: %s", msg)
		}

		switch ev.Type {
		case "message_stop":
			goto flush
		case "content_block_delta":
			// text_delta emits text.
			if ev.Delta.Type == "text_delta" {
				if err := emit(aggText, ev.Delta.Text); err != nil {
					return err
				}
				continue
			}
			// tool_use input JSON stream.
			if p.emitToolUseEnabled() && ev.Delta.Type == "input_json_delta" {
				if err := emit(aggTool, ev.Delta.PartialJSON); err != nil {
					return err
				}
				continue
			}
		default:
			// Ignore message_start/content_block_start/message_delta/content_block_stop/ping/etc.
			continue
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read stream: %w", err)
	}

flush:
	// Flush any remaining partial text.
	for _, s := range aggText.Final() {
		item := proto.FromUTF8String(s)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case out <- item:
		}
	}
	if p.emitToolUseEnabled() {
		for _, s := range aggTool.Final() {
			item := proto.FromUTF8String(s)
			select {
			case <-ctx.Done():
				return ctx.Err()
			case out <- item:
			}
		}
	}

	return nil
}

func extractText(res messagesResponse) string {
	if len(res.Content) == 0 {
		return ""
	}
	var b strings.Builder
	for _, c := range res.Content {
		if c.Type != "text" {
			continue
		}
		if c.Text == "" {
			continue
		}
		b.WriteString(c.Text)
	}
	return b.String()
}

func extractToolUseInputJSON(res messagesResponse) string {
	if len(res.Content) == 0 {
		return ""
	}
	for _, c := range res.Content {
		if c.Type != "tool_use" {
			continue
		}
		if c.Input == nil {
			continue
		}
		b, err := json.Marshal(c.Input)
		if err != nil {
			continue
		}
		return string(b)
	}
	return ""
}

func (p ResponseProcessor[S]) emitToolUseEnabled() bool {
	if p.EmitToolUse == nil {
		return false
	}
	return *p.EmitToolUse
}

// -----------------------
// Chainable With* methods
// -----------------------

// WithAggregateType returns a copy with AggregateType updated.
func (p ResponseProcessor[S]) WithAggregateType(t AggregateType) ResponseProcessor[S] {
	p.AggregateType = t
	return p
}

// WithRole sets the role used when the processor constructs its default message.
func (p ResponseProcessor[S]) WithRole(role Role) ResponseProcessor[S] {
	p.Role = role
	return p
}

// WithBaseURL sets the Claude API base URL.
func (p ResponseProcessor[S]) WithBaseURL(baseURL string) ResponseProcessor[S] {
	p.BaseURL = strings.TrimSpace(baseURL)
	return p
}

// WithAPIVersion sets the anthropic-version header value.
func (p ResponseProcessor[S]) WithAPIVersion(v string) ResponseProcessor[S] {
	p.APIVersion = strings.TrimSpace(v)
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

// WithMaxTokens sets max_tokens (required by the API).
func (p ResponseProcessor[S]) WithMaxTokens(v int) ResponseProcessor[S] {
	p.MaxTokens = &v
	return p
}

// WithInstructions sets the top-level system prompt.
func (p ResponseProcessor[S]) WithInstructions(system string) ResponseProcessor[S] {
	p.System = &system
	return p
}

// WithSystem is an alias for WithInstructions.
func (p ResponseProcessor[S]) WithSystem(system string) ResponseProcessor[S] {
	return p.WithInstructions(system)
}

// WithMessages overrides the entire messages list.
//
// When Messages is set (non-empty), the processor does not inject incoming text.
// You fully control the prompt/messages.
func (p ResponseProcessor[S]) WithMessages(messages ...Message) ResponseProcessor[S] {
	p.Messages = append([]Message(nil), messages...)
	return p
}

// WithTemperature sets temperature.
func (p ResponseProcessor[S]) WithTemperature(v float64) ResponseProcessor[S] {
	p.Temperature = &v
	return p
}

// WithTopP sets top_p.
func (p ResponseProcessor[S]) WithTopP(v float64) ResponseProcessor[S] {
	p.TopP = &v
	return p
}

// WithTopK sets top_k.
func (p ResponseProcessor[S]) WithTopK(v int) ResponseProcessor[S] {
	p.TopK = &v
	return p
}

// WithStopSequences sets stop_sequences.
func (p ResponseProcessor[S]) WithStopSequences(stops ...string) ResponseProcessor[S] {
	p.StopSequences = append([]string(nil), stops...)
	return p
}

// WithMetadata sets metadata.
func (p ResponseProcessor[S]) WithMetadata(m map[string]string) ResponseProcessor[S] {
	if m == nil {
		p.Metadata = nil
		return p
	}
	dup := make(map[string]string, len(m))
	for k, v := range m {
		dup[k] = v
	}
	p.Metadata = dup
	return p
}

// WithTools sets tools.
func (p ResponseProcessor[S]) WithTools(tools ...Tool) ResponseProcessor[S] {
	p.Tools = append([]Tool(nil), tools...)
	return p
}

// WithToolChoice sets tool_choice (string or object).
func (p ResponseProcessor[S]) WithToolChoice(choice any) ResponseProcessor[S] {
	p.ToolChoice = choice
	return p
}

// WithEmitToolUse enables/disables emitting tool_use blocks as textual output.
func (p ResponseProcessor[S]) WithEmitToolUse(v bool) ResponseProcessor[S] {
	p.EmitToolUse = &v
	return p
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

// --------------------------
// Small internal validations
// --------------------------

func ensureAPIKey() error {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return errors.New("invalid or missing ANTHROPIC_API_KEY")
	}
	return nil
}
