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

package models

import "strings"

// Compile-time check: XAIModel implements Model.
var _ Model = XAIModel{}

func (m XAIModel) ProviderInfo() ProviderInfo { p, _ := Providers[ProviderXAI]; return p }
func (m XAIModel) Identifier() ModelID        { return m.ID }
func (m XAIModel) DisplayName() string        { return m.Name }
func (m XAIModel) Kind() string               { return m.Flavour }
func (m XAIModel) TagList() []Tag             { return m.Tags }
func (m XAIModel) Summary() string            { return m.Description }
func (m XAIModel) KnownSnapshots() []string   { return m.Snapshots }
func (m XAIModel) IsDeprecated() bool         { return m.Deprecated }
func (m XAIModel) KnownSizes() []string       { return nil }
func (m XAIModel) LicenseText() string        { return "" }

func (m XAIModel) SupportsTools() bool     { return supportsTools(m.Tags) }
func (m XAIModel) SupportsThinking() bool  { return supportsThinking(m.Flavour, m.Tags) }
func (m XAIModel) SupportsVision() bool    { return supportsVision(m.Flavour, m.Tags) }
func (m XAIModel) SupportsEmbedding() bool { return supportsEmbedding(m.Flavour, m.Tags) }

// XAIModel contains metadata about an xAI (OpenAI-compatible) model.
//
// The IDs map directly to the `model` parameter used across OpenAI-compatible endpoints.
type XAIModel struct {
	// ID is the stable alias used in the API, e.g. "grok-4".
	ID ModelID

	// Name is a human-friendly display name.
	Name string

	// Flavour is a loose category used for UI/filters (e.g. "thinking", "instruct",
	// "embedding", "image", "audio", "moderation", "tools").
	Flavour string

	// Tags is a list of cross-cutting capabilities (vision, tools, thinking, etc.).
	Tags []Tag

	// Description is a short, UI-friendly summary of the model.
	Description string

	// Snapshots optionally lists known pinned model versions (if any).
	Snapshots []string

	// Deprecated indicates the model is listed as deprecated.
	Deprecated bool
}

// XAIModels holds a collection of XAIModel metadata.
type XAIModels struct {
	All []XAIModel
}

// xAI model identifiers (curated).
//
// Note: This is intentionally a small, best-effort list. The resolver can still accept
// unlisted IDs for ProviderXAI (see Resolve in lookup.go).
const (
	Grok4                  ModelID = "grok-4"
	Grok4Fast              ModelID = "grok-4-fast"
	Grok41Fast             ModelID = "grok-4-1-fast"
	GrokCodeFast1          ModelID = "grok-code-fast-1"
	Grok4FastNonReasoning  ModelID = "grok-4-fast-non-reasoning"
	Grok41FastNonReasoning ModelID = "grok-4-1-fast-non-reasoning"
)

// AllXAIModels is a curated list of xAI models.
//
// Tags and flavours are best-effort UI hints; the authoritative capabilities are determined by the provider.
var AllXAIModels = XAIModels{All: []XAIModel{
	{
		ID:          Grok4,
		Name:        "Grok 4",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagVision, TagTools},
		Description: "xAI flagship Grok model (OpenAI-compatible).",
	},
	{
		ID:          Grok4Fast,
		Name:        "Grok 4 Fast",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Lower-latency Grok 4 tier (OpenAI-compatible).",
	},
	{
		ID:          Grok41Fast,
		Name:        "Grok 4.1 Fast",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Fast Grok tier commonly used in function-calling examples (OpenAI-compatible).",
	},
	{
		ID:          GrokCodeFast1,
		Name:        "Grok Code Fast 1",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Coding-focused Grok model (OpenAI-compatible).",
	},
	{
		ID:          Grok4FastNonReasoning,
		Name:        "Grok 4 Fast (Non-Reasoning)",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Non-reasoning variant (OpenAI-compatible).",
	},
	{
		ID:          Grok41FastNonReasoning,
		Name:        "Grok 4.1 Fast (Non-Reasoning)",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Non-reasoning fast Grok tier (OpenAI-compatible).",
	},
}}

// Search finds models whose Name, ID, or Tags contain the query substring (case-insensitive).
func (m *XAIModels) Search(query string) []XAIModel {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil
	}

	var results []XAIModel
	for _, model := range m.All {
		if strings.Contains(strings.ToLower(model.Name), q) ||
			strings.Contains(strings.ToLower(string(model.ID)), q) {
			results = append(results, model)
			continue
		}
		for _, tag := range model.Tags {
			if strings.Contains(strings.ToLower(string(tag)), q) {
				results = append(results, model)
				break
			}
		}
	}
	return results
}
