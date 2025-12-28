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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/carrier"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualshared"
)

var apiKey = os.Getenv("OPENAI_API_KEY")

// TextualProcessorChatEvent is kept for backward compatibility with earlier
// event naming conventions used elsewhere in the project.
const TextualProcessorChatEvent = "openai.responses"

const (
	RoleUser      textualshared.Role = "user"
	RoleAssistant textualshared.Role = "assistant"
	RoleSystem    textualshared.Role = "system"
	RoleDeveloper textualshared.Role = "developer"
)

// TruncationStrategy controls how the model handles context overflow.
type TruncationStrategy string

const (
	TruncationDisabled TruncationStrategy = "disabled"
	TruncationAuto     TruncationStrategy = "auto"
)

// ServiceTier controls request processing tier selection.
type ServiceTier string

const (
	ServiceTierAuto     ServiceTier = "auto"
	ServiceTierDefault  ServiceTier = "default"
	ServiceTierFlex     ServiceTier = "flex"
	ServiceTierPriority ServiceTier = "priority"
)

// PromptCacheRetention controls extended prompt cache retention.
type PromptCacheRetention string

const (
	PromptCacheRetention24h PromptCacheRetention = "24h"
)

// ResponseProcessor is a textual.Processor that calls OpenAI's Responses API
// (POST /v1/responses) with streaming enabled and re-emits the streamed
// output_text as carrier values.
//
// It exposes every request-body option surfaced in the Responses API reference
// (including nested options) via chainable With* methods.
//
// By default, ResponseProcessor builds an "input" payload as a single message item:
//
//	[ { "type":"message", "role":"user", "content":[{"type":"input_text","text":"..."}] } ]
//
// You can override Input directly via WithInputString / WithInputItems (or set it
// to nil and let the processor build the default payload).
type ResponseProcessor[S carrier.Carrier[S]] struct {
	// Shared behavior: prompt templating + aggregation settings.
	// Embedded for DRY reuse across provider processors.
	textualshared.ResponseProcessor[S]

	// ---------------------------
	// Responses API request body
	// ---------------------------

	Background           *bool             `json:"background,omitempty"`
	Conversation         any               `json:"conversation,omitempty"` // string or object
	Include              []string          `json:"include,omitempty"`
	Input                any               `json:"input,omitempty"` // string or array (input items)
	Instructions         *string           `json:"instructions,omitempty"`
	MaxOutputTokens      *int              `json:"max_output_tokens,omitempty"`
	MaxToolCalls         *int              `json:"max_tool_calls,omitempty"`
	Metadata             map[string]string `json:"metadata,omitempty"`
	ParallelToolCalls    *bool             `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID   *string           `json:"previous_response_id,omitempty"`
	Prompt               *PromptRef        `json:"prompt,omitempty"`
	PromptCacheKey       *string           `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention *string           `json:"prompt_cache_retention,omitempty"`
	Reasoning            *ReasoningConfig  `json:"reasoning,omitempty"`
	SafetyIdentifier     *string           `json:"safety_identifier,omitempty"`
	ServiceTier          *string           `json:"service_tier,omitempty"`
	Store                *bool             `json:"store,omitempty"`
	Stream               *bool             `json:"stream,omitempty"`
	StreamOptions        *StreamOptions    `json:"stream_options,omitempty"`
	Temperature          *float64          `json:"temperature,omitempty"`
	Text                 *TextConfig       `json:"text,omitempty"`
	ToolChoice           any               `json:"tool_choice,omitempty"` // string or object
	Tools                []Tool            `json:"tools,omitempty"`
	TopLogprobs          *int              `json:"top_logprobs,omitempty"`
	TopP                 *float64          `json:"top_p,omitempty"`
	Truncation           *string           `json:"truncation,omitempty"`
	User                 *string           `json:"user,omitempty"` // Deprecated
}

// PromptRef references a prompt template and variables.
type PromptRef struct {
	ID        string         `json:"id,omitempty"`
	Variables map[string]any `json:"variables,omitempty"`
	Version   string         `json:"version,omitempty"`
}

// ReasoningConfig configures reasoning models.
type ReasoningConfig struct {
	Effort  *string `json:"effort,omitempty"`  // e.g. "none","low","medium","high","xhigh" (model-dependent)
	Summary *string `json:"summary,omitempty"` // e.g. "auto","concise","detailed" (model-dependent)
}

// StreamOptions configures streaming behavior.
type StreamOptions struct {
	// include_usage: include a final usage event (if supported).
	IncludeUsage *bool `json:"include_usage,omitempty"`
}

// TextConfig configures text output, either plain text or structured JSON.
type TextConfig struct {
	Format any `json:"format,omitempty"` // object; typically {type:"text"} or {type:"json_schema", json_schema:{...}}
}

// TextFormatText is the simplest format: plain text.
type TextFormatText struct {
	Type string `json:"type"` // "text"
}

// JSONSchemaFormat is a Structured Outputs format wrapper.
type JSONSchemaFormat struct {
	Type       string     `json:"type"` // "json_schema"
	JSONSchema JSONSchema `json:"json_schema"`
}

type JSONSchema struct {
	Name   string         `json:"name,omitempty"`
	Schema map[string]any `json:"schema,omitempty"`
	Strict *bool          `json:"strict,omitempty"`
}

// Tool is a generic tool declaration for the Responses API.
// It supports built-in tools and custom function tools.
// (You can extend this struct for specific tool types as needed.)
type Tool struct {
	Type     string        `json:"type"`               // e.g. "web_search", "file_search", "function", "mcp"
	Function *FunctionTool `json:"function,omitempty"` // for Type=="function"
	// Tool-specific additional fields can be provided via Extra.
	Extra map[string]any `json:"-"`
}

func (t Tool) MarshalJSON() ([]byte, error) {
	// Merge declared fields + Extra
	m := map[string]any{
		"type": t.Type,
	}
	if t.Function != nil {
		m["function"] = t.Function
	}
	for k, v := range t.Extra {
		// Don't allow Extra to override core fields.
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

// NewResponseProcessor builds a ResponseProcessor from a model name and a template string.
//
// Validation performed:
//   - Ensures OPENAI_API_KEY looks non-empty (length >= 10).
//   - Ensures model is not empty.
//   - Parses the template string.
//   - Ensures the template references {{.Input}} (or {{ .Input }}) so the
//     incoming text is injected.
func NewResponseProcessor[S carrier.Carrier[S]](model, templateStr string) (*ResponseProcessor[S], error) {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return nil, fmt.Errorf("invalid or missing OPENAI_API_KEY")
	}

	model = strings.TrimSpace(model)
	if model == "" {
		return nil, fmt.Errorf("model must not be empty")
	}

	tmpl, err := textualshared.ParseTemplate("responses", templateStr)
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
		Stream: &stream,
	}, nil
}

// Apply implements textual.Processor.
//
// It consumes incoming carrier values, sends each through the Responses API
// pipeline, and emits zero or more carrier values per input (streamed output).
func (p ResponseProcessor[S]) Apply(ctx context.Context, in <-chan S) <-chan S {
	return p.ResponseProcessor.Apply(ctx, in, p.handlePrompt)
}

func (p ResponseProcessor[S]) handlePrompt(ctx context.Context, _ S, prompt string, out chan<- S) error {
	reqBody, err := p.buildRequestBody(prompt)
	if err != nil {
		return fmt.Errorf("build responses request: %w", err)
	}
	if err := p.streamResponses(ctx, reqBody, out); err != nil {
		return fmt.Errorf("responses stream: %w", err)
	}
	return nil
}

// ResponseInputItem is a simplified representation of an input item accepted by /v1/responses.
// For most chat-like usage, you only need Type="message", Role="user", and Content=[{type:"input_text",text:"..."}].
type ResponseInputItem struct {
	ID      string            `json:"id,omitempty"`
	Type    string            `json:"type"` // "message" for chat-like input
	Role    string            `json:"role,omitempty"`
	Content []ResponseContent `json:"content,omitempty"`
}

type ResponseContent struct {
	Type string `json:"type"` // "input_text", "input_image", ...
	Text string `json:"text,omitempty"`
	// Additional content fields can be added as needed (image_url, file_id, ...)
}

func (p ResponseProcessor[S]) buildRequestBody(prompt string) (map[string]any, error) {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return nil, fmt.Errorf("invalid or missing OPENAI_API_KEY")
	}

	body := map[string]any{}

	// Required-ish for meaningful requests.
	if strings.TrimSpace(p.Model) != "" {
		body["model"] = p.Model
	}

	// Default input if none is provided: a single "message" with input_text.
	if p.Input == nil {
		role := string(p.Role)
		if role == "" {
			role = string(RoleUser)
		}
		body["input"] = []ResponseInputItem{
			{
				Type: "message",
				Role: role,
				Content: []ResponseContent{
					{Type: "input_text", Text: prompt},
				},
			},
		}
	} else {
		body["input"] = p.Input
	}

	// Pass through every other option if set.
	if p.Background != nil {
		body["background"] = *p.Background
	}
	if p.Conversation != nil {
		body["conversation"] = p.Conversation
	}
	if len(p.Include) > 0 {
		body["include"] = p.Include
	}
	if p.Instructions != nil {
		body["instructions"] = *p.Instructions
	}
	if p.MaxOutputTokens != nil {
		body["max_output_tokens"] = *p.MaxOutputTokens
	}
	if p.MaxToolCalls != nil {
		body["max_tool_calls"] = *p.MaxToolCalls
	}
	if p.Metadata != nil {
		body["metadata"] = p.Metadata
	}
	if p.ParallelToolCalls != nil {
		body["parallel_tool_calls"] = *p.ParallelToolCalls
	}
	if p.PreviousResponseID != nil {
		body["previous_response_id"] = *p.PreviousResponseID
	}
	if p.Prompt != nil {
		body["prompt"] = p.Prompt
	}
	if p.PromptCacheKey != nil {
		body["prompt_cache_key"] = *p.PromptCacheKey
	}
	if p.PromptCacheRetention != nil {
		body["prompt_cache_retention"] = *p.PromptCacheRetention
	}
	if p.Reasoning != nil {
		body["reasoning"] = p.Reasoning
	}
	if p.SafetyIdentifier != nil {
		body["safety_identifier"] = *p.SafetyIdentifier
	}
	if p.ServiceTier != nil {
		body["service_tier"] = *p.ServiceTier
	}
	if p.Store != nil {
		body["store"] = *p.Store
	}
	if p.Stream != nil {
		body["stream"] = *p.Stream
	}
	if p.StreamOptions != nil {
		body["stream_options"] = p.StreamOptions
	}
	if p.Temperature != nil {
		body["temperature"] = *p.Temperature
	}
	if p.Text != nil {
		body["text"] = p.Text
	}
	if p.ToolChoice != nil {
		body["tool_choice"] = p.ToolChoice
	}
	if len(p.Tools) > 0 {
		body["tools"] = p.Tools
	}
	if p.TopLogprobs != nil {
		body["top_logprobs"] = *p.TopLogprobs
	}
	if p.TopP != nil {
		body["top_p"] = *p.TopP
	}
	if p.Truncation != nil {
		body["truncation"] = *p.Truncation
	}
	if p.User != nil {
		body["user"] = *p.User
	}

	return body, nil
}

// responsesStreamEvent is the minimal subset used to extract output text deltas.
type responsesStreamEvent struct {
	Type  string `json:"type"`
	Delta string `json:"delta,omitempty"`
	Text  string `json:"text,omitempty"`

	// Some events include error objects. We keep it generic.
	Error any `json:"error,omitempty"`
}

// streamResponses performs a streaming POST to the Responses API, parses SSE
// events, aggregates output_text deltas according to AggregateType, and emits
// carrier values on out.
func (p ResponseProcessor[S]) streamResponses(ctx context.Context, reqBody map[string]any, out chan<- S) error {
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal responses request: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		"https://api.openai.com/v1/responses",
		bytes.NewReader(bodyBytes),
	)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("perform request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		limited, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("openai responses API returned %d: %s", resp.StatusCode, strings.TrimSpace(string(limited)))
	}

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
		// Some SSE implementations send [DONE]; tolerate it even if not documented.
		if data == "[DONE]" {
			break
		}

		var ev responsesStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			// Skip malformed events but keep streaming.
			continue
		}

		// We only aggregate response.output_text.delta events.
		// response.output_text.done may contain the full text; we don't need it
		// because we already track the accumulation.
		if ev.Type != "response.output_text.delta" {
			// If the API reports a failure event, surface it.
			if ev.Type == "response.failed" {
				return fmt.Errorf("responses stream failed: %s", data)
			}
			continue
		}

		if ev.Delta == "" {
			continue
		}

		segments := agg.Append(ev.Delta)
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

// WithAggregateType returns a copy with the AggregateType updated.
func (p ResponseProcessor[S]) WithAggregateType(t textualshared.AggregateType) ResponseProcessor[S] {
	p.AggregateType = t
	return p
}

// WithRole sets the role used when the processor constructs its default input message item.
func (p ResponseProcessor[S]) WithRole(role textualshared.Role) ResponseProcessor[S] {
	p.Role = role
	return p
}

// WithBackground sets the `background` request option.
func (p ResponseProcessor[S]) WithBackground(v bool) ResponseProcessor[S] {
	p.Background = &v
	return p
}

// WithConversation sets `conversation` (string or object).
func (p ResponseProcessor[S]) WithConversation(conversation any) ResponseProcessor[S] {
	p.Conversation = conversation
	return p
}

// WithInclude sets the `include` list.
func (p ResponseProcessor[S]) WithInclude(include ...string) ResponseProcessor[S] {
	p.Include = append([]string(nil), include...)
	return p
}

// WithInputString sets input as a raw string (instead of message items).
func (p ResponseProcessor[S]) WithInputString(input string) ResponseProcessor[S] {
	p.Input = input
	return p
}

// WithInputItems sets input as an array of input items.
func (p ResponseProcessor[S]) WithInputItems(items []ResponseInputItem) ResponseProcessor[S] {
	dup := make([]ResponseInputItem, len(items))
	copy(dup, items)
	p.Input = dup
	return p
}

// WithInstructions sets the `instructions` option (system/developer message inserted into context).
func (p ResponseProcessor[S]) WithInstructions(instructions string) ResponseProcessor[S] {
	p.Instructions = &instructions
	return p
}

// WithMaxOutputTokens sets `max_output_tokens`.
func (p ResponseProcessor[S]) WithMaxOutputTokens(n int) ResponseProcessor[S] {
	p.MaxOutputTokens = &n
	return p
}

// WithMaxToolCalls sets `max_tool_calls`.
func (p ResponseProcessor[S]) WithMaxToolCalls(n int) ResponseProcessor[S] {
	p.MaxToolCalls = &n
	return p
}

// WithMetadata sets `metadata`.
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

// WithModel sets the `model` option.
func (p ResponseProcessor[S]) WithModel(model string) ResponseProcessor[S] {
	p.Model = strings.TrimSpace(model)
	return p
}

// WithParallelToolCalls sets `parallel_tool_calls`.
func (p ResponseProcessor[S]) WithParallelToolCalls(v bool) ResponseProcessor[S] {
	p.ParallelToolCalls = &v
	return p
}

// WithPreviousResponseID sets `previous_response_id`.
func (p ResponseProcessor[S]) WithPreviousResponseID(id string) ResponseProcessor[S] {
	p.PreviousResponseID = &id
	return p
}

// WithPrompt sets `prompt`.
func (p ResponseProcessor[S]) WithPrompt(prompt *PromptRef) ResponseProcessor[S] {
	p.Prompt = prompt
	return p
}

// WithPromptCacheKey sets `prompt_cache_key`.
func (p ResponseProcessor[S]) WithPromptCacheKey(key string) ResponseProcessor[S] {
	p.PromptCacheKey = &key
	return p
}

// WithPromptCacheRetention sets `prompt_cache_retention` (e.g. "24h").
func (p ResponseProcessor[S]) WithPromptCacheRetention(retention PromptCacheRetention) ResponseProcessor[S] {
	v := string(retention)
	p.PromptCacheRetention = &v
	return p
}

// WithReasoning sets the `reasoning` config.
func (p ResponseProcessor[S]) WithReasoning(cfg *ReasoningConfig) ResponseProcessor[S] {
	p.Reasoning = cfg
	return p
}

// WithSafetyIdentifier sets `safety_identifier`.
func (p ResponseProcessor[S]) WithSafetyIdentifier(id string) ResponseProcessor[S] {
	p.SafetyIdentifier = &id
	return p
}

// WithServiceTier sets `service_tier`.
func (p ResponseProcessor[S]) WithServiceTier(tier ServiceTier) ResponseProcessor[S] {
	v := string(tier)
	p.ServiceTier = &v
	return p
}

// WithStore sets `store`.
func (p ResponseProcessor[S]) WithStore(v bool) ResponseProcessor[S] {
	p.Store = &v
	return p
}

// WithStream sets `stream` (this processor is intended for stream=true).
func (p ResponseProcessor[S]) WithStream(v bool) ResponseProcessor[S] {
	p.Stream = &v
	return p
}

// WithStreamOptions sets `stream_options`.
func (p ResponseProcessor[S]) WithStreamOptions(opts *StreamOptions) ResponseProcessor[S] {
	p.StreamOptions = opts
	return p
}

// WithTemperature sets `temperature`.
func (p ResponseProcessor[S]) WithTemperature(v float64) ResponseProcessor[S] {
	p.Temperature = &v
	return p
}

// WithTextConfig sets `text`.
func (p ResponseProcessor[S]) WithTextConfig(cfg *TextConfig) ResponseProcessor[S] {
	p.Text = cfg
	return p
}

// WithTextFormatText sets text.format to {type:"text"}.
func (p ResponseProcessor[S]) WithTextFormatText() ResponseProcessor[S] {
	p.Text = &TextConfig{Format: TextFormatText{Type: "text"}}
	return p
}

// WithTextFormatJSONSchema sets text.format to {type:"json_schema", json_schema:{...}}.
func (p ResponseProcessor[S]) WithTextFormatJSONSchema(schema JSONSchemaFormat) ResponseProcessor[S] {
	p.Text = &TextConfig{Format: schema}
	return p
}

// WithToolChoice sets `tool_choice` (string or object).
func (p ResponseProcessor[S]) WithToolChoice(choice any) ResponseProcessor[S] {
	p.ToolChoice = choice
	return p
}

// WithTools sets `tools`.
func (p ResponseProcessor[S]) WithTools(tools ...Tool) ResponseProcessor[S] {
	p.Tools = append([]Tool(nil), tools...)
	return p
}

// WithTopLogprobs sets `top_logprobs`.
func (p ResponseProcessor[S]) WithTopLogprobs(n int) ResponseProcessor[S] {
	p.TopLogprobs = &n
	return p
}

// WithTopP sets `top_p`.
func (p ResponseProcessor[S]) WithTopP(v float64) ResponseProcessor[S] {
	p.TopP = &v
	return p
}

// WithTruncation sets `truncation` ("auto" or "disabled").
func (p ResponseProcessor[S]) WithTruncation(strategy TruncationStrategy) ResponseProcessor[S] {
	v := string(strategy)
	p.Truncation = &v
	return p
}

// WithUser sets `user` (deprecated in the API; prefer prompt_cache_key / safety_identifier).
func (p ResponseProcessor[S]) WithUser(user string) ResponseProcessor[S] {
	p.User = &user
	return p
}
