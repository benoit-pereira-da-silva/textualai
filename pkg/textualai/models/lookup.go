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
		return Model{}, err
	}
	return Resolve(prv, mid)
}

// Resolve returns the best-effort Model metadata for (provider, id).
//
// It uses curated lists when possible.
// For some providers (currently xAI), it falls back to a best-effort model record when unknown.
func Resolve(providerName ProviderName, id ModelID) (Model, error) {
	if strings.TrimSpace(string(providerName)) == "" {
		return Model{}, errors.New("provider name is empty")
	}
	if strings.TrimSpace(string(id)) == "" {
		return Model{}, errors.New("model identifier is empty")
	}
	mid := ModelID(strings.TrimSpace(string(id)))
	provider, ok := providers[providerName]
	if !ok {
		return Model{}, fmt.Errorf("provider named \"%s\" not found", providerName)
	}
	models := provider.Models
	idStr := strings.TrimSpace(string(mid))
	// 1) Exact match on base ID.
	for _, m := range models {
		if strings.TrimSpace(string(m.ID)) == idStr {
			mm := m
			return mm, nil
		}
	}
	// 2) Snapshot match.
	for _, m := range models {
		for _, snap := range m.Snapshots {
			if strings.TrimSpace(snap) == idStr {
				mm := m
				mm.ID = ModelID(idStr)
				return mm, nil
			}
		}
	}
	// 3) Variant match: base:variant  (ollama syntax)
	parts := strings.SplitN(idStr, ":", 2)
	if len(parts) == 2 {
		base := strings.TrimSpace(parts[0])
		variant := strings.TrimSpace(parts[1])
		if base != "" {
			// Iterate on the models
			for _, m := range models {
				if strings.TrimSpace(string(m.ID)) != base {
					continue
				}
				mm := m
				mm.ID = ModelID(idStr)
				// Make the derived model name more informative while keeping it compact.
				if variant != "" && !strings.Contains(mm.Name, variant) {
					mm.Name = mm.Name + " (" + variant + ")"
				}
				// Keep sizes as a hint for UIs. If the base entry doesn't list sizes, we at least include the variant.
				if variant != "" {
					if mm.Sizes == nil {
						mm.Sizes = []string{variant}
					} else if !stringSliceContains(mm.Sizes, variant) {
						mm.Sizes = append([]string{variant}, mm.Sizes...)
					}
				}
				return mm, nil
			}
		}
	}
	return Model{}, fmt.Errorf("model not found provider name: %s modelId: %s", providerName, id)
}
