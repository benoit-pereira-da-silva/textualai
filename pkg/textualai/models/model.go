// Copyright 2026 Benoit Pereira da Silva
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package models

import "strings"

// ProviderName identifies a model provider in ModelString descriptors.
//
// Examples:
//   - "openai"  -> OpenAI Platform
//   - "ollama"  -> Ollama (OpenAI-compatible /v1 endpoints)
type ProviderName string

// ModelID is a unique identifier for a model (as used in Ollama).
type ModelID string

const (
	ProviderOpenAI ProviderName = "openai"
	ProviderOllama ProviderName = "ollama"
)

// ProviderInfo contains provider-level capabilities and defaults.
//
// This is intentionally provider-scoped (not model-scoped): e.g. OpenAI supports
// the Responses "conversation" feature for persisted conversations, while many
// OpenAI-compatible providers do not.
type ProviderInfo struct {
	// Name is the canonical provider identifier used in ModelString descriptors.
	Name ProviderName `json:"name"`

	// DisplayName is a UI-friendly provider name.
	DisplayName string `json:"display_name"`

	// DefaultBaseURL is the default OpenAI-compatible base URL for this provider.
	// It MUST NOT have a trailing slash.
	DefaultBaseURL string `json:"default_base_url"`

	// APIKeyRequired indicates whether this provider requires an API key for typical use.
	// This is a convenience signal used by samples / CLI; callers may override.
	APIKeyRequired bool `json:"api_key_required"`

	// SupportsConversation indicates whether the provider supports conversation persistency
	// using the OpenAI Responses API "conversation" feature.
	SupportsConversation bool `json:"supports_conversation"`

	// SupportsStrictFunctionTools indicates whether the provider supports the `strict`
	// attribute for function tools (JSON Schema strict mode).
	SupportsStrictFunctionTools bool `json:"supports_strict_function_tools"`
}

// Providers is a small provider registry used for model parsing and capability gating.
// Add new providers here as the framework expands.
var Providers = map[ProviderName]ProviderInfo{
	ProviderOpenAI: {
		Name:                        ProviderOpenAI,
		DisplayName:                 "OpenAI",
		DefaultBaseURL:              "https://api.openai.com/v1",
		APIKeyRequired:              true,
		SupportsConversation:        true,
		SupportsStrictFunctionTools: true,
	},
	ProviderOllama: {
		Name:                        ProviderOllama,
		DisplayName:                 "Ollama",
		DefaultBaseURL:              "http://localhost:11434/v1",
		APIKeyRequired:              false,
		SupportsConversation:        false,
		SupportsStrictFunctionTools: false,
	},
}

// Provider returns provider metadata if the provider is registered.
func Provider(name ProviderName) (ProviderInfo, bool) {
	p, ok := Providers[name]
	return p, ok
}

// NormalizeProviderName canonicalizes a provider name for lookups.
// It returns (provider, true) when the provider is known, otherwise ("", false).
func NormalizeProviderName(s string) (ProviderName, bool) {
	s = strings.ToLower(strings.TrimSpace(s))
	if s == "" {
		return "", false
	}
	for name := range Providers {
		if s == string(name) {
			return name, true
		}
	}
	return "", false
}

// Model is a provider-agnostic view of an LLM/AI model definition.
// Provider-specific structs (OpenAIModel, OllamaModel, ...) implement this interface.
//
// The method names are intentionally chosen to avoid clashing with common struct field
// names (ID, Name, Tags, ...), allowing existing provider metadata structs to implement
// the interface without structural changes.
type Model interface {
	// ProviderName returns the model provider.
	ProviderName() ProviderName

	// Identifier returns the provider model identifier (the string used in API calls).
	Identifier() ModelID

	// DisplayName is a human-friendly name for UI.
	DisplayName() string

	// Kind is a loose category used for UI/filters (e.g. "thinking", "instruct",
	// "embedding", "image", "audio", "moderation", "tools").
	Kind() string

	// TagList is a list of cross-cutting capabilities (vision, tools, thinking, etc.).
	TagList() []Tag

	// Summary is a short UI-friendly description.
	Summary() string

	// KnownSnapshots returns known pinned snapshot IDs for this model, if any.
	KnownSnapshots() []string

	// IsDeprecated indicates the model is listed as deprecated.
	IsDeprecated() bool

	// KnownSizes returns model size variants when applicable (e.g. Ollama sizes).
	KnownSizes() []string

	// LicenseText returns the model license information when applicable.
	LicenseText() string

	// Derived capabilities (convenience helpers).
	SupportsTools() bool
	SupportsThinking() bool
	SupportsVision() bool
	SupportsEmbedding() bool
}

// UniversalModel is a lightweight implementation of Model used when:
//   - the provider/model ID is unknown to the curated lists, OR
//   - we derive a model from a base entry (e.g. Ollama "qwen3:32b" from "qwen3").
//
// It is intentionally a superset of commonly useful metadata across providers.
type UniversalModel struct {
	Provider    ProviderName
	ID          ModelID
	Name        string
	Flavour     string
	Tags        []Tag
	Description string
	Snapshots   []string
	Deprecated  bool
	Sizes       []string
	License     string
}

func (m UniversalModel) ProviderName() ProviderName { return m.Provider }
func (m UniversalModel) Identifier() ModelID        { return m.ID }
func (m UniversalModel) DisplayName() string        { return m.Name }
func (m UniversalModel) Kind() string               { return m.Flavour }
func (m UniversalModel) TagList() []Tag             { return m.Tags }
func (m UniversalModel) Summary() string            { return m.Description }
func (m UniversalModel) KnownSnapshots() []string   { return m.Snapshots }
func (m UniversalModel) IsDeprecated() bool         { return m.Deprecated }
func (m UniversalModel) KnownSizes() []string       { return m.Sizes }
func (m UniversalModel) LicenseText() string        { return m.License }

func (m UniversalModel) SupportsTools() bool     { return supportsTools(m.Tags) }
func (m UniversalModel) SupportsThinking() bool  { return supportsThinking(m.Flavour, m.Tags) }
func (m UniversalModel) SupportsVision() bool    { return supportsVision(m.Flavour, m.Tags) }
func (m UniversalModel) SupportsEmbedding() bool { return supportsEmbedding(m.Flavour, m.Tags) }

func supportsTools(tags []Tag) bool {
	return hasTag(tags, TagTools)
}

func supportsThinking(flavour string, tags []Tag) bool {
	if hasTag(tags, TagThinking) {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(flavour), "thinking")
}

func supportsVision(flavour string, tags []Tag) bool {
	if hasTag(tags, TagVision) {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(flavour), "vision")
}

func supportsEmbedding(flavour string, tags []Tag) bool {
	if hasTag(tags, TagEmbedding) {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(flavour), "embedding")
}

func hasTag(tags []Tag, want Tag) bool {
	w := strings.TrimSpace(string(want))
	if w == "" {
		return false
	}
	for _, t := range tags {
		if strings.EqualFold(strings.TrimSpace(string(t)), w) {
			return true
		}
	}
	return false
}

// stringSliceContains reports whether slice contains s (case-sensitive).
func stringSliceContains(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}
