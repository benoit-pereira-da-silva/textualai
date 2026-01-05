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

import (
	"errors"
	"fmt"
	"strings"
)

func ModelFromString(s string) (Model, error) {
	ms := ModelString(s)
	prv, mid, err := ms.Split()
	if err != nil {
		return nil, err
	}
	return Resolve(prv, mid)
}

// Resolve returns the best-effort Model metadata for (provider, id).
//
// It uses curated lists when possible, and falls back to UniversalModel when unknown.
func Resolve(provider ProviderName, id ModelID) (Model, error) {
	p := provider
	if strings.TrimSpace(string(p)) == "" {
		return OpenAIModel{}, errors.New("provider name is empty")
	}
	if strings.TrimSpace(string(id)) == "" {
		return OpenAIModel{}, errors.New("model identifier is empty")
	}
	mid := ModelID(strings.TrimSpace(string(id)))
	switch p {
	case ProviderOpenAI:
		if m, ok := LookupOpenAIModel(mid); ok {
			return m, nil
		}
		return OpenAIModel{}, fmt.Errorf("openai model not found %s", mid)
	case ProviderOllama:
		if m, ok := LookupOllamaModel(mid); ok {
			return m, nil
		}
		return OpenAIModel{}, fmt.Errorf("ollama model not found %s", mid)
	default:
		return OpenAIModel{}, errors.New("provider name is invalid")
	}
}

// LookupOpenAIModel tries to find a curated OpenAI model entry by:
//  1. exact ID match (base model IDs), then
//  2. snapshot ID match (pinned versions listed in Snapshots).
//
// When a snapshot match is found, the returned model is a copy of the base entry
// with ID set to the snapshot value.
func LookupOpenAIModel(id ModelID) (OpenAIModel, bool) {
	idStr := strings.TrimSpace(string(id))
	if idStr == "" {
		return OpenAIModel{}, false
	}

	// 1) Exact match on base ID.
	for _, m := range AllOpenAIModels.All {
		if strings.TrimSpace(string(m.ID)) == idStr {
			return m, true
		}
	}

	// 2) Snapshot match.
	for _, m := range AllOpenAIModels.All {
		for _, snap := range m.Snapshots {
			if strings.TrimSpace(snap) == idStr {
				mm := m
				mm.ID = ModelID(idStr)
				return mm, true
			}
		}
	}

	return OpenAIModel{}, false
}

// LookupOllamaModel tries to find a curated Ollama model entry by:
//  1. exact ID match, then
//  2. "base:variant" match where base is a curated ID and variant is a size/tag.
//
// For example, "qwen3:32b" will be resolved from the "qwen3" curated entry when present.
func LookupOllamaModel(id ModelID) (OllamaModel, bool) {
	idStr := strings.TrimSpace(string(id))
	if idStr == "" {
		return OllamaModel{}, false
	}

	// 1) Exact match.
	for _, m := range AllOllamaModels.All {
		if strings.TrimSpace(string(m.ID)) == idStr {
			return m, true
		}
	}

	// 2) Variant match: base:variant
	parts := strings.SplitN(idStr, ":", 2)
	if len(parts) != 2 {
		return OllamaModel{}, false
	}
	base := strings.TrimSpace(parts[0])
	variant := strings.TrimSpace(parts[1])
	if base == "" {
		return OllamaModel{}, false
	}

	for _, m := range AllOllamaModels.All {
		if strings.TrimSpace(string(m.ID)) != base {
			continue
		}
		mm := m
		mm.ID = ModelID(idStr)

		// Make the derived model name more informative while keeping it compact.
		if variant != "" && !strings.Contains(mm.Name, variant) {
			mm.Name = mm.Name + " (" + variant + ")"
		}

		// Keep sizes as a hint for UIs. If base entry doesn't list sizes, we at least include the variant.
		if variant != "" {
			if mm.Sizes == nil {
				mm.Sizes = []string{variant}
			} else if !stringSliceContains(mm.Sizes, variant) {
				mm.Sizes = append([]string{variant}, mm.Sizes...)
			}
		}

		return mm, true
	}

	return OllamaModel{}, false
}
