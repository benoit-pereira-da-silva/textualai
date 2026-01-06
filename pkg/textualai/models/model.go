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

func supportsThinking(flavor string, tags []Tag) bool {
	if hasTag(tags, TagThinking) {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(flavor), "thinking")
}

func supportsVision(flavor string, tags []Tag) bool {
	if hasTag(tags, TagVision) {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(flavor), "vision")
}

func supportsEmbedding(flavor string, tags []Tag) bool {
	if hasTag(tags, TagEmbedding) {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(flavor), "embedding")
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
	ProviderName ProviderName `json:"providerName"`

	// ID is the stable alias used in the API, e.g. "grok-4".
	ID ModelID `json:"id"`

	// Name is a human-friendly display name.
	Name string `json:"name"`

	// Flavor is a loose category used for UI/filters (e.g. "thinking", "instruct",
	// "embedding", "image", "audio", "moderation", "tools").
	Flavor string `json:"flavor"`

	// Tags is a list of cross-cutting capabilities (vision, tools, thinking, etc.).
	Tags []Tag `json:"tags"`

	// Description is a short, UI-friendly summary of the model.
	Description string `json:"description"`

	// Sizes optionally list known sizes variants (ollama)
	Sizes []string `json:"sizes"`

	// Snapshots optionally list known pinned model versions (if any).
	Snapshots []string `json:"snapshots"`

	// License the license of the model if applicable.
	License string `json:"license"`

	// Deprecated indicates the model is listed as deprecated.
	Deprecated bool `json:"deprecated"`
}

func (m Model) ProviderInfo() ProviderInfo {
	pi, _ := m.ProviderName.ProviderInfo()
	return pi
}
func (m Model) SupportsTools() bool     { return supportsTools(m.Tags) }
func (m Model) SupportsThinking() bool  { return supportsThinking(m.Flavor, m.Tags) }
func (m Model) SupportsVision() bool    { return supportsVision(m.Flavor, m.Tags) }
func (m Model) SupportsEmbedding() bool { return supportsEmbedding(m.Flavor, m.Tags) }
