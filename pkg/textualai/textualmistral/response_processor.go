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

package textualmistral

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualshared"
)

var apiKey = os.Getenv("MISTRAL_API_KEY")

// TextualProcessorChatEvent is kept for backward compatibility with earlier
// event naming conventions used elsewhere in the project.
const TextualProcessorChatEvent = "mistral.chat_completions"

const (
	RoleUser      textualshared.Role = "user"
	RoleAssistant textualshared.Role = "assistant"
	RoleSystem    textualshared.Role = "system"
	RoleTool      textualshared.Role = "tool"
)

// ResponseProcessor is a textual.Processor that calls Mistral's Chat Completions API
// (POST /v1/chat/completions) with streaming enabled and re-emits the streamed
// content as carrier values.
//
// It intentionally mirrors the style of the OpenAI Responses API processor and
// the Ollama processor in this repository:
//
//   - It uses a Go text/template to build the prompt from each incoming carrier.
//   - It streams the response, aggregates it (Word or Line), and emits carrier
//     values on the output channel.
//   - It exposes common request options from Mistral's API via chainable With* methods.
//
// Important semantics (same as the OpenAI/Ollama processors in this repo):
//
//   - If you set an explicit Messages list (WithMessages), the incoming carrier
//     text is no longer injected. You fully control the request payload.
//
// Environment:
//
//   - MISTRAL_API_KEY is required.
//   - BaseURL defaults to MISTRAL_BASE_URL (if set) or https://api.mistral.ai.
type ResponseProcessor[S textual.Carrier[S]] struct {
	// Shared behavior: prompt templating + aggregation settings.
	// Embedded for DRY reuse across provider processors.
	textualshared.ResponseProcessor[S]

	// BaseURL is the Mistral API base URL (default https://api.mistral.ai).
	BaseURL string `json:"baseURL,omitempty"`

	// ---------------------------
	// Chat Completions request body
	// ---------------------------

	Messages []ChatMessage `json:"messages,omitempty"`

	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	N           *int     `json:"n,omitempty"` // ?? @bpds ??

	Stream *bool `json:"stream,omitempty"`

	// Mistral options commonly present in their API surface.
	SafePrompt *bool   `json:"safe_prompt,omitempty"`
	RandomSeed *int    `json:"random_seed,omitempty"`
	PromptMode *string `json:"prompt_mode,omitempty"`

	// Tool calling.
	Tools             []Tool `json:"tools,omitempty"`
	ToolChoice        any    `json:"tool_choice,omitempty"` // string or object
	ParallelToolCalls *bool  `json:"parallel_tool_calls,omitempty"`

	// Penalties (if supported by your model / API version).
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`

	// Structured outputs / JsonCarrier mode.
	ResponseFormat any `json:"response_format,omitempty"`

	// Instructions is mapped to a system message when Messages is not explicitly set.
	Instructions *string `json:"-"`

	// Extra holds additional request keys not explicitly modeled above.
	// Extra keys never override explicitly modeled fields.
	Extra map[string]any `json:"-"`
}

// ChatMessage is the message representation used by /v1/chat/completions.
type ChatMessage struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`

	// Name is optional and used for tool/function messages in some APIs.
	Name string `json:"name,omitempty"`

	// ToolCallID is used in tool result messages.
	ToolCallID string `json:"tool_call_id,omitempty"`

	// ToolCalls is set when the model wants to call tools.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// Tool is a generic tool declaration for Mistral chat completions.
// We mirror the OpenAI-style schema to ease porting.
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
	Parameters  map[string]any `json:"parameters,omitempty"` // JsonCarrier Schema
}

// ToolCall is the minimal tool call representation returned by Mistral.
type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function,omitempty"`
}

type ToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// TextFormatText is the simplest format: plain text.
type TextFormatText struct {
	Type string `json:"type"` // "text"
}

// TextFormatJSONObject requests a JsonCarrier object response (no schema).
type TextFormatJSONObject struct {
	Type string `json:"type"` // "json_object"
}

// JSONSchemaFormat is a Structured Outputs wrapper (schema-based).
type JSONSchemaFormat struct {
	Type       string     `json:"type"` // "json_schema"
	JSONSchema JSONSchema `json:"json_schema"`
}

type JSONSchema struct {
	Name   string         `json:"name,omitempty"`
	Schema map[string]any `json:"schema,omitempty"`
	Strict *bool          `json:"strict,omitempty"`
}

// NewResponseProcessor builds a ResponseProcessor from a model name and a template string.
//
// Validation performed:
//   - Ensures MISTRAL_API_KEY looks non-empty (length >= 10).
//   - Ensures model is not empty.
//   - Parses the template string.
//   - Ensures the template references {{.Input}} (or {{ .Input }}) so the
//     incoming text is injected.
func NewResponseProcessor[S textual.Carrier[S]](model, templateStr string) (*ResponseProcessor[S], error) {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return nil, fmt.Errorf("invalid or missing MISTRAL_API_KEY")
	}

	model = strings.TrimSpace(model)
	if model == "" {
		return nil, fmt.Errorf("model must not be empty")
	}

	tmpl, err := textualshared.ParseTemplate("mistral.chat_completions", templateStr)
	if err != nil {
		return nil, err
	}

	// Defaults: streaming is the whole point of this processor.
	stream := true

	return &ResponseProcessor[S]{
		ResponseProcessor: textualshared.ResponseProcessor[S]{
			Model:         model,
			Role:          RoleUser,
			Template:      tmpl,
			AggregateType: textualshared.Word,
		},
		BaseURL: defaultBaseURL(),

		Stream: &stream,
	}, nil
}

// Apply implements textual.Processor.
//
// It consumes incoming carrier values, sends each through Mistral's API
// pipeline, and emits zero or more carrier values per input (streamed output).
func (p ResponseProcessor[S]) Apply(ctx context.Context, in <-chan S) <-chan S {
	return p.ResponseProcessor.Apply(ctx, in, p.handlePrompt)
}

func (p ResponseProcessor[S]) handlePrompt(ctx context.Context, _ S, prompt string, out chan<- S) error {
	reqBody, err := p.buildRequestBody(prompt)
	if err != nil {
		return fmt.Errorf("build mistral request: %w", err)
	}

	if err := p.streamChatCompletions(ctx, reqBody, out); err != nil {
		return fmt.Errorf("mistral stream: %w", err)
	}
	return nil
}

func defaultBaseURL() string {
	raw := strings.TrimSpace(os.Getenv("MISTRAL_BASE_URL"))
	if raw == "" {
		return "https://api.mistral.ai"
	}
	// Users might set without scheme; default to https.
	if !strings.Contains(raw, "://") {
		raw = "https://" + raw
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
		s = "https://" + s
	}
	s = strings.TrimRight(s, "/")
	return s
}

func (p ResponseProcessor[S]) buildRequestBody(prompt string) (map[string]any, error) {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return nil, fmt.Errorf("invalid or missing MISTRAL_API_KEY")
	}

	body := map[string]any{}

	// Required-ish.
	if strings.TrimSpace(p.Model) != "" {
		body["model"] = p.Model
	}

	// Default messages if none provided: optional system + one user message.
	if len(p.Messages) == 0 {
		msgs := make([]ChatMessage, 0, 2)

		if p.Instructions != nil && strings.TrimSpace(*p.Instructions) != "" {
			msgs = append(msgs, ChatMessage{
				Role:    string(RoleSystem),
				Content: *p.Instructions,
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

	// Standard options.
	if p.Temperature != nil {
		body["temperature"] = *p.Temperature
	}
	if p.TopP != nil {
		body["top_p"] = *p.TopP
	}
	if p.MaxTokens != nil {
		body["max_tokens"] = *p.MaxTokens
	}
	if len(p.Stop) > 0 {
		body["stop"] = append([]string(nil), p.Stop...)
	}
	if p.Stream != nil {
		body["stream"] = *p.Stream
	}

	// Mistral-specific.
	if p.SafePrompt != nil {
		body["safe_prompt"] = *p.SafePrompt
	}
	if p.RandomSeed != nil {
		body["random_seed"] = *p.RandomSeed
	}
	if p.PromptMode != nil && strings.TrimSpace(*p.PromptMode) != "" {
		body["prompt_mode"] = strings.TrimSpace(*p.PromptMode)
	}
	if p.ParallelToolCalls != nil {
		body["parallel_tool_calls"] = *p.ParallelToolCalls
	}
	if p.FrequencyPenalty != nil {
		body["frequency_penalty"] = *p.FrequencyPenalty
	}
	if p.PresencePenalty != nil {
		body["presence_penalty"] = *p.PresencePenalty
	}

	// Tools.
	if len(p.Tools) > 0 {
		body["tools"] = p.Tools
	}
	if p.ToolChoice != nil {
		body["tool_choice"] = p.ToolChoice
	}

	// Response formatting.
	if p.ResponseFormat != nil {
		body["response_format"] = p.ResponseFormat
	}

	// Extra keys last, never overriding modeled keys.
	for k, v := range p.Extra {
		if _, exists := body[k]; exists {
			continue
		}
		body[k] = v
	}

	return body, nil
}

// ----------------------
// Streaming (SSE) parsing
// ----------------------

// mistralErrorResponse matches the common error payload shape.
type mistralErrorResponse struct {
	Error any `json:"error,omitempty"`
}

// chatCompletionStreamEvent is a minimal SSE event for streamed deltas.
//
// Typical Mistral streaming payload resembles OpenAI ChatCompletions:
//
//	{
//	  "id": "...",
//	  "object": "chat.completion.chunk",
//	  "choices": [
//	    {
//	      "index": 0,
//	      "delta": {"role":"assistant","content":"..."},
//	      "finish_reason": null
//	    }
//	  ]
//	}
type chatCompletionStreamEvent struct {
	Choices []struct {
		Index int `json:"index,omitempty"`
		Delta struct {
			Role      string     `json:"role,omitempty"`
			Content   string     `json:"content,omitempty"`
			ToolCalls []ToolCall `json:"tool_calls,omitempty"`
		} `json:"delta,omitempty"`
		Message *struct {
			Role      string     `json:"role,omitempty"`
			Content   string     `json:"content,omitempty"`
			ToolCalls []ToolCall `json:"tool_calls,omitempty"`
		} `json:"message,omitempty"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices,omitempty"`

	// Some APIs include top-level error.
	Error any `json:"error,omitempty"`
}

// chatCompletionResponse is the non-stream response envelope.
type chatCompletionResponse struct {
	Choices []struct {
		Index   int `json:"index,omitempty"`
		Message struct {
			Role    string `json:"role,omitempty"`
			Content string `json:"content,omitempty"`
		} `json:"message,omitempty"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices,omitempty"`
	Error any `json:"error,omitempty"`
}

func (p ResponseProcessor[S]) streamChatCompletions(ctx context.Context, reqBody map[string]any, out chan<- S) error {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return fmt.Errorf("invalid or missing MISTRAL_API_KEY")
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	base := normalizeBaseURL(p.BaseURL)
	url := base + "/v1/chat/completions"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(apiKey))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("perform request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		limited, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		// Try to surface structured error if present.
		var er mistralErrorResponse
		if json.Unmarshal(limited, &er) == nil && er.Error != nil {
			return fmt.Errorf("mistral API returned %d: %v", resp.StatusCode, er.Error)
		}
		return fmt.Errorf("mistral API returned %d: %s", resp.StatusCode, strings.TrimSpace(string(limited)))
	}

	// If stream=false, we get a single JsonCarrier response.
	stream := true
	if p.Stream != nil {
		stream = *p.Stream
	}
	if !stream {
		limited, _ := io.ReadAll(io.LimitReader(resp.Body, 32*1024*1024))
		var full chatCompletionResponse
		if err := json.Unmarshal(limited, &full); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
		if full.Error != nil {
			return fmt.Errorf("mistral error: %v", full.Error)
		}
		if len(full.Choices) == 0 {
			return nil
		}
		text := full.Choices[0].Message.Content
		proto := *new(S)
		res := proto.FromUTF8String(text)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case out <- res:
			return nil
		}
	}

	// Stream=true (SSE).
	agg := textualshared.NewStreamAggregator(p.AggregateType)
	scanner := bufio.NewScanner(resp.Body)

	// Allow reasonably large SSE lines.
	const maxScanTokenSize = 1024 * 1024 // 1 MiB
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, maxScanTokenSize)

	proto := *new(S)

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
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		// Mistral (like OpenAI) typically ends streams with [DONE].
		if data == "[DONE]" {
			break
		}

		var ev chatCompletionStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			// Skip malformed events but keep streaming.
			continue
		}
		if ev.Error != nil {
			return fmt.Errorf("mistral stream error: %v", ev.Error)
		}
		if len(ev.Choices) == 0 {
			continue
		}

		// Delta content.
		delta := ev.Choices[0].Delta.Content
		if delta == "" && ev.Choices[0].Message != nil {
			// Some servers may send a "message" field in chunks.
			delta = ev.Choices[0].Message.Content
		}
		if delta == "" {
			continue
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
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read stream: %w", err)
	}

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
func (p ResponseProcessor[S]) WithAggregateType(t textualshared.AggregateType) ResponseProcessor[S] {
	p.AggregateType = t
	return p
}

// WithRole sets the role used when the processor constructs its default chat message.
func (p ResponseProcessor[S]) WithRole(role textualshared.Role) ResponseProcessor[S] {
	p.Role = role
	return p
}

// WithBaseURL sets the Mistral API base URL.
func (p ResponseProcessor[S]) WithBaseURL(baseURL string) ResponseProcessor[S] {
	p.BaseURL = strings.TrimSpace(baseURL)
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

// WithInstructions sets the system prompt when Messages is not explicitly provided.
func (p ResponseProcessor[S]) WithInstructions(system string) ResponseProcessor[S] {
	p.Instructions = &system
	return p
}

// WithSystem is an alias for WithInstructions.
func (p ResponseProcessor[S]) WithSystem(system string) ResponseProcessor[S] {
	return p.WithInstructions(system)
}

// WithMessages overrides the entire messages list for chat completions.
//
// When Messages is set (non-empty), the processor does not inject incoming text.
// You fully control the prompt/messages.
func (p ResponseProcessor[S]) WithMessages(messages ...ChatMessage) ResponseProcessor[S] {
	p.Messages = append([]ChatMessage(nil), messages...)
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

// WithMaxTokens sets max_tokens.
func (p ResponseProcessor[S]) WithMaxTokens(n int) ResponseProcessor[S] {
	p.MaxTokens = &n
	return p
}

// WithStop sets stop sequences.
func (p ResponseProcessor[S]) WithStop(stops ...string) ResponseProcessor[S] {
	p.Stop = append([]string(nil), stops...)
	return p
}

func (p ResponseProcessor[S]) WithN(n int) ResponseProcessor[S] {
	p.N = &n
	return p
}

// WithSafePrompt sets safe_prompt.
func (p ResponseProcessor[S]) WithSafePrompt(v bool) ResponseProcessor[S] {
	p.SafePrompt = &v
	return p
}

// WithRandomSeed sets random_seed.
func (p ResponseProcessor[S]) WithRandomSeed(v int) ResponseProcessor[S] {
	p.RandomSeed = &v
	return p
}

// WithPromptMode sets prompt_mode (model-dependent).
func (p ResponseProcessor[S]) WithPromptMode(mode string) ResponseProcessor[S] {
	mode = strings.TrimSpace(mode)
	if mode == "" {
		p.PromptMode = nil
		return p
	}
	p.PromptMode = &mode
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

// WithParallelToolCalls sets parallel_tool_calls.
func (p ResponseProcessor[S]) WithParallelToolCalls(v bool) ResponseProcessor[S] {
	p.ParallelToolCalls = &v
	return p
}

// WithFrequencyPenalty sets frequency_penalty.
func (p ResponseProcessor[S]) WithFrequencyPenalty(v float64) ResponseProcessor[S] {
	p.FrequencyPenalty = &v
	return p
}

// WithPresencePenalty sets presence_penalty.
func (p ResponseProcessor[S]) WithPresencePenalty(v float64) ResponseProcessor[S] {
	p.PresencePenalty = &v
	return p
}

// WithResponseFormat sets response_format.
func (p ResponseProcessor[S]) WithResponseFormat(format any) ResponseProcessor[S] {
	p.ResponseFormat = format
	return p
}

// WithResponseFormatText sets response_format to plain text.
func (p ResponseProcessor[S]) WithResponseFormatText() ResponseProcessor[S] {
	p.ResponseFormat = TextFormatText{Type: "text"}
	return p
}

// WithResponseFormatJSONObject requests JsonCarrier object output (no schema).
func (p ResponseProcessor[S]) WithResponseFormatJSONObject() ResponseProcessor[S] {
	p.ResponseFormat = TextFormatJSONObject{Type: "json_object"}
	return p
}

// WithResponseFormatJSONSchema sets response_format to JsonCarrier schema structured output.
func (p ResponseProcessor[S]) WithResponseFormatJSONSchema(schema JSONSchemaFormat) ResponseProcessor[S] {
	p.ResponseFormat = schema
	return p
}

// WithExtra sets an additional request key/value.
// Extra keys never override explicitly modeled fields.
func (p ResponseProcessor[S]) WithExtra(key string, value any) ResponseProcessor[S] {
	key = strings.TrimSpace(key)
	if key == "" {
		return p
	}
	if p.Extra == nil {
		p.Extra = make(map[string]any)
	}
	p.Extra[key] = value
	return p
}
