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

// ModelID is a unique identifier for a model.
type ModelID string

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

// stringSliceContains reports whether a slice contains s (case-sensitive).
func stringSliceContains(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}

type Models []Model

type Model struct {

	// ProviderName is injected by the Providers init func.
	ProviderName ProviderName

	// id is the stable alias used in the API, e.g. "grok-4".
	ID ModelID

	// name is a human-friendly display name.
	Name string

	// flavour is a loose category used for UI/filters (e.g. "thinking", "instruct",
	// "embedding", "image", "audio", "moderation", "tools").
	Flavour string

	// tags is a list of cross-cutting capabilities (vision, tools, thinking, etc.).
	Tags []Tag

	// description is a short, UI-friendly summary of the model.
	Description string

	// sizes optionally list known sizes variants (ollama)
	Sizes []string

	// snapshots optionally list known pinned model versions (if any).
	Snapshots []string

	// license the licence of the model if applicable.
	License string

	// deprecated indicates the model is listed as deprecated.
	Deprecated bool
}

func (m Model) ProviderInfo() ProviderInfo {
	pi, _ := m.ProviderName.ProviderInfo()
	return pi
}
func (m Model) SupportsTools() bool     { return supportsTools(m.Tags) }
func (m Model) SupportsThinking() bool  { return supportsThinking(m.Flavour, m.Tags) }
func (m Model) SupportsVision() bool    { return supportsVision(m.Flavour, m.Tags) }
func (m Model) SupportsEmbedding() bool { return supportsEmbedding(m.Flavour, m.Tags) }
