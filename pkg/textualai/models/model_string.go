package models

import (
	"fmt"
	"strings"
)

// ModelString is a compact provider-qualified model descriptor.
//
// Examples:
//   - "openai:gpt-4.1"
//   - "ollama:qwen3:32b"
//
// If no explicit provider prefix is present, the provider defaults to "openai"
// and the entire string is treated as the model id.
type ModelString string

func (s ModelString) String() string { return string(s) }

// FromModelString validates and canonicalizes a model descriptor.
//
// Validation rules:
//   - string must be non-empty after trimming spaces
//   - if an explicit provider prefix is used, it must be a known provider
//   - if an explicit provider prefix is used, the model id portion must be non-empty
//
// Canonicalization rules:
//   - explicit provider names are lower-cased (e.g. "OpenAI:" -> "openai:")
func FromModelString(s string) (ModelString, error) {
	raw := strings.TrimSpace(s)
	if raw == "" {
		return "", fmt.Errorf("models: empty model string")
	}

	if p, id, ok := splitExplicitProvider(raw); ok {
		if strings.TrimSpace(id) == "" {
			return "", fmt.Errorf("models: invalid model string %q: missing model id after %q", raw, p)
		}
		// Canonicalize provider casing.
		return ModelString(string(p) + ":" + strings.TrimSpace(id)), nil
	}
	return "", fmt.Errorf("models: invalid model string: %q", raw)
}

// Validate checks whether the ModelString is structurally valid.
func (s ModelString) Validate() error {
	_, _, err := s.Split()
	return err
}

// Provider returns the resolved provider name.
//
// If the ModelString contains an explicit known provider prefix, that prefix is used.
// Otherwise, ProviderOpenAI is returned for backward compatibility with pre-descriptor
// model ids (e.g. "gpt-4.1").
func (s ModelString) Provider() (ProviderName, error) {
	raw := strings.TrimSpace(string(s))
	if raw == "" {
		return "", fmt.Errorf("models: empty model string")
	}

	if p, _, ok := splitExplicitProvider(raw); ok {
		return p, nil
	}
	return ProviderOpenAI, nil
}

// ProviderName returns the provider name as a string (compatibility helper).
func (s ModelString) ProviderName() (string, error) {
	p, err := s.Provider()
	if err != nil {
		return "", err
	}
	return string(p), nil
}

// ModelID returns the provider model identifier part.
//
// If the ModelString contains an explicit known provider prefix, the id portion is returned.
// Otherwise, the entire string is treated as the id.
func (s ModelString) ModelID() (ModelID, error) {
	raw := strings.TrimSpace(string(s))
	if raw == "" {
		return "", fmt.Errorf("models: empty model string")
	}

	if _, id, ok := splitExplicitProvider(raw); ok {
		id = strings.TrimSpace(id)
		if id == "" {
			return "", fmt.Errorf("models: invalid model string %q: missing model id", raw)
		}
		return ModelID(id), nil
	}

	return ModelID(raw), nil
}

// Split returns (provider, model_id) using the same rules as Provider() and ModelID().
func (s ModelString) Split() (ProviderName, ModelID, error) {
	p, err := s.Provider()
	if err != nil {
		return "", "", err
	}
	id, err := s.ModelID()
	if err != nil {
		return "", "", err
	}
	return p, id, nil
}

// ProviderInfo returns provider-level metadata for the provider resolved by Provider().
func (s ModelString) ProviderInfo() (ProviderInfo, error) {
	p, err := s.Provider()
	if err != nil {
		return ProviderInfo{}, err
	}
	info, ok := p.ProviderInfo()
	if !ok {
		return ProviderInfo{}, fmt.Errorf("models: unknown provider: %q", p)
	}
	return info, nil
}

// Model resolves the ModelString into a provider-agnostic Model interface.
//
// The lookup is best-effort:
//   - if the model is found in curated lists (or can be inferred from a base entry),
//     the returned model will include known capabilities (tags, flavour, ...).
//   - otherwise, a UniversalModel with minimal metadata is returned.
func (s ModelString) Model() (Model, error) {
	p, id, err := s.Split()
	if err != nil {
		return Model{}, err
	}
	return Resolve(p, id)
}

// splitExplicitProvider returns (provider, remainder, true) when s starts with a known
// provider prefix "<provider>:".
func splitExplicitProvider(s string) (ProviderName, string, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", "", false
	}
	i := strings.IndexByte(s, ':')
	if i <= 0 {
		return "", "", false
	}

	prefix := s[:i]
	rest := s[i+1:]

	p, ok := NormalizeProviderName(prefix)
	if !ok {
		return "", "", false
	}
	return p, rest, true
}
