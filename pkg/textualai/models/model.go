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

// ModelID is a unique identifier for a model (as used in Ollama).
type ModelID string

// Model is a provider-agnostic view of an LLM/AI model definition.
// Provider-specific structs (OpenAIModel, OllamaModel, ...) implement this interface.
//
// The method names are intentionally chosen to avoid clashing with common struct field
// names (ID, Name, Tags, ...), allowing existing provider metadata structs to implement
// the interface without structural changes.
type Model interface {

	// ProviderInfo returns the model provider.
	ProviderInfo() ProviderInfo

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
