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
var AllXAIModels = Models{
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
}
