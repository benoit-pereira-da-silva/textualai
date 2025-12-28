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

package textualgemini

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

	"github.com/benoit-pereira-da-silva/textual/pkg/carrier"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualshared"
)

var apiKey = os.Getenv("GEMINI_API_KEY")

// TextualProcessorChatEvent is kept for backward compatibility with earlier
// event naming conventions used elsewhere in the project.
const TextualProcessorChatEvent = "gemini.responses"

const (
	RoleUser  textualshared.Role = "user"
	RoleModel textualshared.Role = "model"
)

// ResponseProcessor is a textual.Processor that calls the Gemini API
// (generateContent or streamGenerateContent) and re-emits generated text as
// carrier values.
//
// It intentionally mirrors the style of the OpenAI and Ollama processors in
// this repository:
//
//   - It uses a Go text/template to build the prompt from each incoming carrier.
//   - It streams the response (SSE), aggregates it (Word or Line), and emits
//     carrier values on the output channel.
//   - It exposes common request options via chainable With* methods.
//
// Important semantics (same as the OpenAI/Ollama processors in this repo):
//
//   - If you set explicit Contents, the incoming carrier text is no longer
//     injected. You fully control the request payload.
//
// Environment:
//
//   - GEMINI_API_KEY is required.
type ResponseProcessor[S carrier.Carrier[S]] struct {
	// Shared behavior: prompt templating + aggregation settings.
	// Embedded for DRY reuse across provider processors.
	textualshared.ResponseProcessor[S]

	// BaseURL is the base URL of the Gemini API (default: https://generativelanguage.googleapis.com).
	BaseURL string `json:"baseURL,omitempty"`

	// APIVersion is the REST API version (default: v1beta).
	APIVersion string `json:"apiVersion,omitempty"`

	// Stream controls whether we call the SSE streaming endpoint
	// (streamGenerateContent) or the non-streaming endpoint (generateContent).
	// Default is true.
	Stream *bool `json:"stream,omitempty"`

	// SystemInstruction sets the system prompt/instructions.
	// Mapped to the Gemini `systemInstruction` request field.
	SystemInstruction *string `json:"systemInstruction,omitempty"`

	// Contents overrides the default request contents.
	// If nil or empty, the processor will build a single content item with role=Role and parts=[{text: prompt}].
	Contents []Content `json:"contents,omitempty"`

	// Tools declares tools available to the model (function calling).
	Tools []any `json:"tools,omitempty"`

	// ToolConfig configures tool calling behavior.
	ToolConfig any `json:"toolConfig,omitempty"`

	// SafetySettings controls safety behavior.
	SafetySettings []SafetySetting `json:"safetySettings,omitempty"`

	// GenerationConfig controls decoding and output formatting.
	GenerationConfig *GenerationConfig `json:"generationConfig,omitempty"`
}

// Content matches Gemini "Content" objects inside the `contents` array.
type Content struct {
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts,omitempty"`
}

type Part struct {
	Text string `json:"text,omitempty"`
	// Additional part kinds (inlineData, fileData, functionCall, functionResponse, ...) can be added later if needed.
}

// SafetySetting is a minimal representation of Gemini safety settings.
// See Gemini docs for allowed categories and thresholds.
type SafetySetting struct {
	Category  string `json:"category,omitempty"`
	Threshold string `json:"threshold,omitempty"`
}

// GenerationConfig is a minimal/extendable representation of Gemini generationConfig.
// The list below covers commonly used fields; additional/unknown keys can be provided through Extra.
// Extra keys never override explicitly modeled fields.
type GenerationConfig struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"topP,omitempty"`
	TopK             *int     `json:"topK,omitempty"`
	CandidateCount   *int     `json:"candidateCount,omitempty"`
	MaxOutputTokens  *int     `json:"maxOutputTokens,omitempty"`
	StopSequences    []string `json:"stopSequences,omitempty"`
	ResponseMIMEType *string  `json:"responseMimeType,omitempty"`

	// ResponseSchema: request-side schema for structured output.
	// Use responseMimeType="application/json" + responseSchema=<schema>.
	ResponseSchema any `json:"responseSchema,omitempty"`

	// Extra holds additional generationConfig keys not explicitly listed above.
	// Extra keys never override explicitly modeled fields.
	Extra map[string]any `json:"-"`
}

// MarshalJSON merges typed fields with Extra.
func (o GenerationConfig) MarshalJSON() ([]byte, error) {
	m := map[string]any{}

	if o.Temperature != nil {
		m["temperature"] = *o.Temperature
	}
	if o.TopP != nil {
		m["topP"] = *o.TopP
	}
	if o.TopK != nil {
		m["topK"] = *o.TopK
	}
	if o.CandidateCount != nil {
		m["candidateCount"] = *o.CandidateCount
	}
	if o.MaxOutputTokens != nil {
		m["maxOutputTokens"] = *o.MaxOutputTokens
	}
	if len(o.StopSequences) > 0 {
		m["stopSequences"] = append([]string(nil), o.StopSequences...)
	}
	if o.ResponseMIMEType != nil {
		m["responseMimeType"] = *o.ResponseMIMEType
	}
	if o.ResponseSchema != nil {
		m["responseSchema"] = o.ResponseSchema
	}

	for k, v := range o.Extra {
		if _, exists := m[k]; exists {
			continue
		}
		m[k] = v
	}

	return json.Marshal(m)
}

// NewResponseProcessor builds a ResponseProcessor from a model name and a template string.
//
// Validation performed:
//   - Ensures GEMINI_API_KEY looks non-empty (length >= 10).
//   - Ensures model is not empty.
//   - Parses the template string.
//   - Ensures the template references {{.Input}} (or {{ .Input }}) so the
//     incoming text is injected.
//
// Defaults:
//   - BaseURL: https://generativelanguage.googleapis.com
//   - APIVersion: v1beta
//   - Stream: true
//   - AggregateType: Word
//   - Role: user
func NewResponseProcessor[S carrier.Carrier[S]](model, templateStr string) (*ResponseProcessor[S], error) {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return nil, fmt.Errorf("invalid or missing GEMINI_API_KEY")
	}

	model = strings.TrimSpace(model)
	if model == "" {
		return nil, fmt.Errorf("model must not be empty")
	}

	tmpl, err := textualshared.ParseTemplate("gemini", templateStr)
	if err != nil {
		return nil, err
	}

	stream := true

	return &ResponseProcessor[S]{
		ResponseProcessor: textualshared.ResponseProcessor[S]{
			Model:         normalizeModel(model),
			Role:          RoleUser,
			Template:      tmpl,
			AggregateType: textualshared.Word,
		},
		BaseURL:    defaultBaseURL(),
		APIVersion: defaultAPIVersion(),

		Stream: &stream,
	}, nil
}

func defaultBaseURL() string {
	return "https://generativelanguage.googleapis.com"
}

func defaultAPIVersion() string {
	return "v1beta"
}

func normalizeBaseURL(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultBaseURL()
	}
	s = strings.TrimRight(s, "/")
	return s
}

func normalizeAPIVersion(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return defaultAPIVersion()
	}
	s = strings.Trim(s, "/")
	return s
}

// normalizeModel maps "gemini-2.5-flash" -> "models/gemini-2.5-flash".
// If the string already looks like a resource name ("models/..." or "tunedModels/..."),
// it is returned as-is.
func normalizeModel(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return s
	}
	if strings.HasPrefix(s, "models/") || strings.HasPrefix(s, "tunedModels/") {
		return s
	}
	return "models/" + s
}

// Apply implements textual.Processor.
//
// It consumes incoming carrier values, sends each through the Gemini API
// pipeline, and emits zero or more carrier values per input (streamed output).
func (p ResponseProcessor[S]) Apply(ctx context.Context, in <-chan S) <-chan S {
	return p.ResponseProcessor.Apply(ctx, in, p.handlePrompt)
}

func (p ResponseProcessor[S]) handlePrompt(ctx context.Context, _ S, prompt string, out chan<- S) error {
	url, streaming, reqBody, err := p.buildRequest(prompt)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}

	if err := p.callGemini(ctx, url, streaming, reqBody, out); err != nil {
		return fmt.Errorf("gemini call: %w", err)
	}

	return nil
}

func (p ResponseProcessor[S]) buildRequest(prompt string) (url string, streaming bool, body map[string]any, err error) {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return "", false, nil, fmt.Errorf("invalid or missing GEMINI_API_KEY")
	}

	base := normalizeBaseURL(p.BaseURL)
	ver := normalizeAPIVersion(p.APIVersion)

	model := strings.TrimSpace(p.Model)
	if model == "" {
		return "", false, nil, fmt.Errorf("model must not be empty")
	}
	model = normalizeModel(model)

	// Determine endpoint based on Stream flag.
	streaming = true
	if p.Stream != nil {
		streaming = *p.Stream
	}

	endpoint := "generateContent"
	if streaming {
		endpoint = "streamGenerateContent?alt=sse"
	}
	url = fmt.Sprintf("%s/%s/%s:%s", base, ver, model, endpoint)

	body = map[string]any{}

	// Contents default.
	if len(p.Contents) == 0 {
		role := strings.TrimSpace(string(p.Role))
		if role == "" {
			role = string(RoleUser)
		}
		body["contents"] = []Content{
			{
				Role: role,
				Parts: []Part{
					{Text: prompt},
				},
			},
		}
	} else {
		dup := make([]Content, len(p.Contents))
		copy(dup, p.Contents)
		body["contents"] = dup
	}

	// System instruction.
	if p.SystemInstruction != nil && strings.TrimSpace(*p.SystemInstruction) != "" {
		body["systemInstruction"] = Content{
			Role: string(RoleUser),
			Parts: []Part{
				{Text: *p.SystemInstruction},
			},
		}
	}

	// Tools / tool config.
	if len(p.Tools) > 0 {
		body["tools"] = append([]any(nil), p.Tools...)
	}
	if p.ToolConfig != nil {
		body["toolConfig"] = p.ToolConfig
	}

	// Safety.
	if len(p.SafetySettings) > 0 {
		dup := make([]SafetySetting, len(p.SafetySettings))
		copy(dup, p.SafetySettings)
		body["safetySettings"] = dup
	}

	// Generation config.
	if p.GenerationConfig != nil {
		body["generationConfig"] = p.GenerationConfig
	}

	return url, streaming, body, nil
}

// googleErrorResponse matches common Google JSON error shape.
type googleErrorResponse struct {
	Error struct {
		Code    int    `json:"code,omitempty"`
		Message string `json:"message,omitempty"`
		Status  string `json:"status,omitempty"`
	} `json:"error,omitempty"`
}

// generateContentResponse is a minimal subset of the Gemini response.
// We keep it flexible enough to work for both streaming and non-streaming.
type generateContentResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text,omitempty"`
			} `json:"parts,omitempty"`
		} `json:"content,omitempty"`
	} `json:"candidates,omitempty"`
}

// callGemini sends the request either to the streaming SSE endpoint or the non-streaming endpoint.
// It aggregates output and emits snapshot carrier values into out.
func (p ResponseProcessor[S]) callGemini(
	ctx context.Context,
	url string,
	streaming bool,
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
	req.Header.Set("x-goog-api-key", apiKey)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("perform request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		limited, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))

		// Attempt to parse Google error shape for nicer messages.
		var ge googleErrorResponse
		if json.Unmarshal(limited, &ge) == nil && strings.TrimSpace(ge.Error.Message) != "" {
			return fmt.Errorf("gemini API returned %d: %s", resp.StatusCode, strings.TrimSpace(ge.Error.Message))
		}

		return fmt.Errorf("gemini API returned %d: %s", resp.StatusCode, strings.TrimSpace(string(limited)))
	}

	if !streaming {
		return p.handleNonStreaming(ctx, resp.Body, out)
	}
	return p.handleStreamingSSE(ctx, resp.Body, out)
}

func (p ResponseProcessor[S]) handleNonStreaming(ctx context.Context, r io.Reader, out chan<- S) error {
	limited, err := io.ReadAll(io.LimitReader(r, 16*1024*1024))
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}

	var res generateContentResponse
	if err := json.Unmarshal(limited, &res); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	text := extractText(res)
	if text == "" {
		return nil
	}

	agg := textualshared.NewStreamAggregator(p.AggregateType)
	segments := agg.Append(text)
	segments = append(segments, agg.Final()...)

	proto := *new(S)
	var accumulated string
	for _, s := range segments {
		if s == "" {
			continue
		}
		accumulated = s // aggregator emits snapshots (prefixes)
		item := proto.FromUTF8String(accumulated)
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

	agg := textualshared.NewStreamAggregator(p.AggregateType)
	proto := *new(S)

	var lastText string
	var full string

	emitSegments := func(delta string) error {
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
		if data == "[DONE]" {
			break
		}

		var res generateContentResponse
		if err := json.Unmarshal([]byte(data), &res); err != nil {
			// Skip malformed events but keep streaming.
			continue
		}

		currText := extractText(res)

		// The streaming API may send either:
		//   - incremental deltas, or
		//   - a growing full text so far.
		// We normalize by diffing against the last extracted text.
		deltaText := currText
		if strings.HasPrefix(currText, lastText) {
			deltaText = currText[len(lastText):]
		}
		if deltaText != "" {
			full += deltaText
			lastText = full

			if err := emitSegments(deltaText); err != nil {
				return err
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read stream: %w", err)
	}

	// Flush any remaining partial text.
	tails := agg.Final()
	for _, s := range tails {
		item := proto.FromUTF8String(s)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case out <- item:
		}
	}

	return nil
}

func extractText(res generateContentResponse) string {
	if len(res.Candidates) == 0 {
		return ""
	}
	parts := res.Candidates[0].Content.Parts
	if len(parts) == 0 {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Text == "" {
			continue
		}
		b.WriteString(p.Text)
	}
	return b.String()
}

// -----------------------
// Chainable With* methods
// -----------------------

// WithAggregateType returns a copy with AggregateType updated.
func (p ResponseProcessor[S]) WithAggregateType(t textualshared.AggregateType) ResponseProcessor[S] {
	p.AggregateType = t
	return p
}

// WithRole sets the role used when the processor constructs its default content item.
func (p ResponseProcessor[S]) WithRole(role textualshared.Role) ResponseProcessor[S] {
	p.Role = role
	return p
}

// WithBaseURL sets the Gemini API base URL.
func (p ResponseProcessor[S]) WithBaseURL(baseURL string) ResponseProcessor[S] {
	p.BaseURL = strings.TrimSpace(baseURL)
	return p
}

// WithAPIVersion sets the Gemini REST API version (e.g. v1beta).
func (p ResponseProcessor[S]) WithAPIVersion(version string) ResponseProcessor[S] {
	p.APIVersion = strings.TrimSpace(version)
	return p
}

// WithModel sets the model.
func (p ResponseProcessor[S]) WithModel(model string) ResponseProcessor[S] {
	p.Model = normalizeModel(strings.TrimSpace(model))
	return p
}

// WithStream selects streaming (streamGenerateContent) or non-streaming (generateContent).
func (p ResponseProcessor[S]) WithStream(v bool) ResponseProcessor[S] {
	p.Stream = &v
	return p
}

// WithInstructions sets the system instruction.
func (p ResponseProcessor[S]) WithInstructions(system string) ResponseProcessor[S] {
	p.SystemInstruction = &system
	return p
}

// WithSystem is an alias for WithInstructions.
func (p ResponseProcessor[S]) WithSystem(system string) ResponseProcessor[S] {
	return p.WithInstructions(system)
}

// WithContents overrides the entire contents list.
// When Contents is set (non-empty), the processor does not inject incoming text.
func (p ResponseProcessor[S]) WithContents(contents ...Content) ResponseProcessor[S] {
	p.Contents = append([]Content(nil), contents...)
	return p
}

// WithTools sets the tools available to the model. Tool types are left as any
// to keep this package lightweight and future-proof.
func (p ResponseProcessor[S]) WithTools(tools ...any) ResponseProcessor[S] {
	p.Tools = append([]any(nil), tools...)
	return p
}

// WithToolConfig sets toolConfig.
func (p ResponseProcessor[S]) WithToolConfig(cfg any) ResponseProcessor[S] {
	p.ToolConfig = cfg
	return p
}

// WithSafetySettings sets safetySettings.
func (p ResponseProcessor[S]) WithSafetySettings(settings ...SafetySetting) ResponseProcessor[S] {
	p.SafetySettings = append([]SafetySetting(nil), settings...)
	return p
}

// WithGenerationConfig sets the full generationConfig.
func (p ResponseProcessor[S]) WithGenerationConfig(cfg *GenerationConfig) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(cfg)
	return p
}

// WithGenerationConfigField sets a single generationConfig key via the Extra map.
// This is useful when Gemini adds new fields not yet modeled in GenerationConfig.
func (p ResponseProcessor[S]) WithGenerationConfigField(key string, value any) ResponseProcessor[S] {
	key = strings.TrimSpace(key)
	if key == "" {
		return p
	}
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	if p.GenerationConfig.Extra == nil {
		p.GenerationConfig.Extra = make(map[string]any)
	}
	p.GenerationConfig.Extra[key] = value
	return p
}

// WithTemperature sets generationConfig.temperature.
func (p ResponseProcessor[S]) WithTemperature(v float64) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	p.GenerationConfig.Temperature = &v
	return p
}

// WithTopP sets generationConfig.topP.
func (p ResponseProcessor[S]) WithTopP(v float64) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	p.GenerationConfig.TopP = &v
	return p
}

// WithTopK sets generationConfig.topK.
func (p ResponseProcessor[S]) WithTopK(v int) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	p.GenerationConfig.TopK = &v
	return p
}

// WithCandidateCount sets generationConfig.candidateCount.
func (p ResponseProcessor[S]) WithCandidateCount(v int) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	p.GenerationConfig.CandidateCount = &v
	return p
}

// WithMaxOutputTokens sets generationConfig.maxOutputTokens.
func (p ResponseProcessor[S]) WithMaxOutputTokens(v int) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	p.GenerationConfig.MaxOutputTokens = &v
	return p
}

// WithStopSequences sets generationConfig.stopSequences.
func (p ResponseProcessor[S]) WithStopSequences(stops ...string) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	p.GenerationConfig.StopSequences = append([]string(nil), stops...)
	return p
}

// WithResponseMIMEType sets generationConfig.responseMimeType (e.g. "application/json").
func (p ResponseProcessor[S]) WithResponseMIMEType(mime string) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	mime = strings.TrimSpace(mime)
	if mime == "" {
		p.GenerationConfig.ResponseMIMEType = nil
		return p
	}
	p.GenerationConfig.ResponseMIMEType = &mime
	return p
}

// WithResponseJSONSchema sets generationConfig.responseSchema for structured output.
func (p ResponseProcessor[S]) WithResponseJSONSchema(schema any) ResponseProcessor[S] {
	p.GenerationConfig = cloneGenerationConfig(p.GenerationConfig)
	p.GenerationConfig.ResponseSchema = schema
	return p
}

// cloneGenerationConfig ensures a non-nil config and avoids sharing pointers across copies.
func cloneGenerationConfig(in *GenerationConfig) *GenerationConfig {
	if in == nil {
		return &GenerationConfig{}
	}
	dup := *in
	if in.StopSequences != nil {
		dup.StopSequences = append([]string(nil), in.StopSequences...)
	}
	if in.ResponseSchema != nil {
		dup.ResponseSchema = in.ResponseSchema
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
// Small internal validations
// --------------------------

func validateNonEmpty(s, name string) error {
	if strings.TrimSpace(s) == "" {
		return fmt.Errorf("%s must not be empty", name)
	}
	return nil
}

func ensureAPIKey() error {
	if len(strings.TrimSpace(apiKey)) < 10 {
		return errors.New("invalid or missing GEMINI_API_KEY")
	}
	return nil
}
