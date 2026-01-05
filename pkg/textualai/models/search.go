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

// Search returns models whose provider, display name, identifier, kind, or tags
// contain the query substring (case-insensitive).
//
// It searches across all curated providers (currently OpenAI + Ollama + xAI).
func Search(query string) []Model {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil
	}

	var results []Model

	for _, m := range AllOpenAIModels.All {
		if modelMatches(m, q) {
			results = append(results, m)
		}
	}
	for _, m := range AllOllamaModels.All {
		if modelMatches(m, q) {
			results = append(results, m)
		}
	}
	for _, m := range AllXAIModels.All {
		if modelMatches(m, q) {
			results = append(results, m)
		}
	}

	return results
}

// SearchProvider is like Search but restricted to a single provider.
func SearchProvider(provider ProviderName, query string) []Model {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil
	}
	var results []Model
	switch provider {
	case ProviderOpenAI:
		for _, m := range AllOpenAIModels.All {
			if modelMatches(m, q) {
				results = append(results, m)
			}
		}
	case ProviderOllama:
		for _, m := range AllOllamaModels.All {
			if modelMatches(m, q) {
				results = append(results, m)
			}
		}
	case ProviderXAI:
		for _, m := range AllXAIModels.All {
			if modelMatches(m, q) {
				results = append(results, m)
			}
		}
	default:
		// Unknown provider: nothing to search in curated catalogs.
		return nil
	}

	return results
}

func modelMatches(m Model, q string) bool {
	if m == nil {
		return false
	}
	pi := m.ProviderInfo()
	if strings.Contains(strings.ToLower(string(pi.Name)), q) {
		return true
	}
	if strings.Contains(strings.ToLower(strings.TrimSpace(m.DisplayName())), q) {
		return true
	}
	if strings.Contains(strings.ToLower(string(m.Identifier())), q) {
		return true
	}
	if strings.Contains(strings.ToLower(strings.TrimSpace(m.Kind())), q) {
		return true
	}

	for _, tag := range m.TagList() {
		if strings.Contains(strings.ToLower(string(tag)), q) {
			return true
		}
	}

	// Snapshots are often searched directly.
	for _, snap := range m.KnownSnapshots() {
		if strings.Contains(strings.ToLower(snap), q) {
			return true
		}
	}

	// Sizes are relevant for some providers.
	for _, sz := range m.KnownSizes() {
		if strings.Contains(strings.ToLower(sz), q) {
			return true
		}
	}

	return false
}
