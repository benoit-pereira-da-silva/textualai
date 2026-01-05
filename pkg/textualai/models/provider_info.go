package models

import "strings"

// ProviderName identifies a model provider in ModelString descriptors.
//
// Examples:
//   - "openai"  -> OpenAI Platform
//   - "ollama"  -> Ollama (OpenAI-compatible /v1 endpoints)
//   - "xai"     -> xAI (OpenAI-compatible /v1 endpoints)
type ProviderName string

const (
	ProviderOpenAI ProviderName = "openai"
	ProviderOllama ProviderName = "ollama"
	ProviderXAI    ProviderName = "xai"
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

// ProviderInfo returns provider metadata if the provider is registered.
func (p ProviderName) ProviderInfo() (ProviderInfo, bool) {
	pi, ok := Providers[p]
	return pi, ok
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
