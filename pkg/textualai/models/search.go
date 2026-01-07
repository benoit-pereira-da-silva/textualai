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
func Search(query string) Models {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil
	}
	var results Models
	for _, m := range AllOpenAIModels {
		if modelMatches(m, q) {
			results = append(results, m)
		}
	}
	for _, m := range AllOllamaModels {
		if modelMatches(m, q) {
			results = append(results, m)
		}
	}
	for _, m := range AllXAIModels {
		if modelMatches(m, q) {
			results = append(results, m)
		}
	}
	return results
}

// SearchProvider is like Search but restricted to a single provider.
func SearchProvider(p ProviderName, query string) Models {
	var results Models = make(Models, 0)
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return results
	}
	for providerName, provider := range providers {
		if p == providerName {
			for _, model := range provider.Models {
				if modelMatches(model, q) {
					results = append(results, model)
				}
			}
		}
	}
	return results
}

func modelMatches(m Model, q string) bool {
	if strings.Contains(strings.ToLower(strings.TrimSpace(m.Name)), q) {
		return true
	}
	if strings.Contains(strings.ToLower(string(m.ID)), q) {
		return true
	}
	if strings.Contains(strings.ToLower(strings.TrimSpace(m.Flavor)), q) {
		return true
	}

	for _, tag := range m.Tags {
		if strings.Contains(strings.ToLower(string(tag)), q) {
			return true
		}
	}
	// Snapshots are often searched directly.
	for _, snap := range m.Snapshots {
		if strings.Contains(strings.ToLower(snap), q) {
			return true
		}
	}
	// Sizes are relevant for some providers.
	for _, sz := range m.Sizes {
		if strings.Contains(strings.ToLower(sz), q) {
			return true
		}
	}
	return false
}

// Search finds models whose Name, ID, or Tags contain the query substring (case-insensitive).
func (m Models) Search(query string) Models {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return nil
	}
	var results Models
	for _, model := range m {
		if strings.Contains(strings.ToLower(string(model.ID)), q) ||
			strings.Contains(strings.ToLower(string(model.Name)), q) {
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
