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

package cli

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualclaude"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualgemini"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualmistral"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualollama"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualopenai"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualshared"
)

// ------------------------------
// Provider builders
// ------------------------------

func DefaultOpenAIBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if len(strings.TrimSpace(getenv("OPENAI_API_KEY"))) < 10 {
		return nil, errors.New("missing or invalid OPENAI_API_KEY (required for OpenAI provider)")
	}

	procPtr, err := textualopenai.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualopenai.Line)
	default:
		proc = proc.WithAggregateType(textualopenai.Word)
	}

	// Role.
	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualopenai.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualopenai.RoleAssistant)
		case "system":
			proc = proc.WithRole(textualopenai.RoleSystem)
		case "developer":
			proc = proc.WithRole(textualopenai.RoleDeveloper)
		}
	}

	// Instructions.
	if cfg.Instructions != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	// Sampling.
	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithMaxOutputTokens(cfg.MaxTokens.Value())
	}

	// Structured output.
	if outputSchema != nil {
		strict := true
		proc = proc.WithTextFormatJSONSchema(textualopenai.JSONSchemaFormat{
			Type: "json_schema",
			JSONSchema: textualopenai.JSONSchema{
				Name:   "response",
				Schema: outputSchema,
				Strict: &strict,
			},
		})
	} else {
		proc = proc.WithTextFormatText()
	}

	return proc, nil
}

func DefaultClaudeBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if len(strings.TrimSpace(getenv("ANTHROPIC_API_KEY"))) < 10 {
		return nil, errors.New("missing or invalid ANTHROPIC_API_KEY (required for Claude provider)")
	}

	procPtr, err := textualclaude.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualshared.Line)
	default:
		proc = proc.WithAggregateType(textualshared.Word)
	}

	// Role (Claude messages roles: user|assistant).
	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualclaude.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualclaude.RoleAssistant)
		}
	}

	// Overrides.
	if cfg.ClaudeBaseURL != "" {
		proc = proc.WithBaseURL(cfg.ClaudeBaseURL)
	}
	if cfg.ClaudeAPIVersion != "" {
		proc = proc.WithAPIVersion(cfg.ClaudeAPIVersion)
	}
	if cfg.ClaudeStream.IsSet() {
		proc = proc.WithStream(cfg.ClaudeStream.Value())
	}

	// Instructions (system prompt).
	if cfg.Instructions != "" {
		proc = proc.WithSystem(cfg.Instructions)
	}

	// Sampling.
	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithMaxTokens(cfg.MaxTokens.Value())
	}

	// Structured output via tool-use (portable approach).
	if outputSchema != nil {
		toolName := "response"
		proc = proc.WithTools(textualclaude.Tool{
			Name:        toolName,
			Description: "Return a response that matches the provided JSON Schema.",
			InputSchema: outputSchema,
		})
		proc = proc.WithToolChoice(textualclaude.ToolChoice{
			Type: "tool",
			Name: toolName,
		})
		proc = proc.WithEmitToolUse(true)
	}

	return proc, nil
}

func DefaultGeminiBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if len(strings.TrimSpace(getenv("GEMINI_API_KEY"))) < 10 {
		return nil, errors.New("missing or invalid GEMINI_API_KEY (required for Gemini provider)")
	}

	procPtr, err := textualgemini.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualgemini.Line)
	default:
		proc = proc.WithAggregateType(textualgemini.Word)
	}

	// Role (Gemini roles: user|model).
	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualgemini.RoleUser)
		case "assistant", "model":
			proc = proc.WithRole(textualgemini.RoleModel)
		}
	}

	// Overrides.
	if cfg.GeminiBaseURL != "" {
		proc = proc.WithBaseURL(cfg.GeminiBaseURL)
	}
	if cfg.GeminiAPIVersion != "" {
		proc = proc.WithAPIVersion(cfg.GeminiAPIVersion)
	}
	if cfg.GeminiStream.IsSet() {
		proc = proc.WithStream(cfg.GeminiStream.Value())
	}

	// Instructions.
	if cfg.Instructions != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	// Sampling.
	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithMaxOutputTokens(cfg.MaxTokens.Value())
	}

	// Structured output.
	if outputSchema != nil {
		proc = proc.WithResponseMIMEType("application/json")
		proc = proc.WithResponseJSONSchema(outputSchema)
	}

	return proc, nil
}

func DefaultMistralBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if len(strings.TrimSpace(getenv("MISTRAL_API_KEY"))) < 10 {
		return nil, errors.New("missing or invalid MISTRAL_API_KEY (required for Mistral provider)")
	}

	procPtr, err := textualmistral.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualmistral.Line)
	default:
		proc = proc.WithAggregateType(textualmistral.Word)
	}

	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualmistral.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualmistral.RoleAssistant)
		case "system":
			proc = proc.WithRole(textualmistral.RoleSystem)
		case "tool":
			proc = proc.WithRole(textualmistral.RoleTool)
		}
	}

	if cfg.MistralBaseURL != "" {
		proc = proc.WithBaseURL(cfg.MistralBaseURL)
	}
	if cfg.MistralStream.IsSet() {
		proc = proc.WithStream(cfg.MistralStream.Value())
	}

	if cfg.Instructions != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithMaxTokens(cfg.MaxTokens.Value())
	}

	if outputSchema != nil {
		strict := true
		proc = proc.WithResponseFormatJSONSchema(textualmistral.JSONSchemaFormat{
			Type: "json_schema",
			JSONSchema: textualmistral.JSONSchema{
				Name:   "response",
				Schema: outputSchema,
				Strict: &strict,
			},
		})
	}

	return proc, nil
}

func DefaultOllamaBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	_ func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	procPtr, err := textualollama.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualollama.Line)
	default:
		proc = proc.WithAggregateType(textualollama.Word)
	}

	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualollama.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualollama.RoleAssistant)
		case "system":
			proc = proc.WithRole(textualollama.RoleSystem)
		}
	}

	if cfg.OllamaHost != "" {
		proc = proc.WithBaseURL(cfg.OllamaHost)
	}

	switch strings.ToLower(cfg.OllamaEndpoint) {
	case "", "chat":
		proc = proc.WithChatEndpoint()
	case "generate":
		proc = proc.WithGenerateEndpoint()
	default:
		proc = proc.WithChatEndpoint()
	}

	if cfg.Instructions != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	if cfg.OllamaStream.IsSet() {
		proc = proc.WithStream(cfg.OllamaStream.Value())
	}

	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithNumPredict(cfg.MaxTokens.Value())
	}

	if outputSchema != nil {
		proc = proc.WithFormat(outputSchema)
	}

	return proc, nil
}
