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

package textualopenai

import (
	"fmt"
	"net/url"
	"os"
	"strings"
)

// Model represents an OpenAI-compatible model identifier.
//
// While this package is named textualopenai, the HTTP client targets OpenAI-style
// /v1 endpoints and therefore works with any compliant provider (OpenAI, Ollama,
// self-hosted gateways, ...).
type Model string

const (
	// === GPT-4.x Family ===

	// ModelGpt41 GPT-4.1 – High-intelligence flagship GPT-4-class model
	ModelGpt41 Model = "gpt-4.1"

	// ModelGpt4o GPT-4o – Multimodal (text + vision) optimized GPT-4-class model
	ModelGpt4o Model = "gpt-4o"

	// ModelGpt4oMini GPT-4o Mini – Lightweight, faster, and cheaper GPT-4o variant
	ModelGpt4oMini Model = "gpt-4o-mini"

	// === GPT-5.x Family ===

	//ModelGpt52 GPT-5.2 – Latest generation flagship model
	ModelGpt52 Model = "gpt-5.2"

	// === Legacy / Compatibility Models ===

	//ModelGpt35Turbo GPT-3.5 Turbo – Legacy chat completion model (still supported for compatibility)
	ModelGpt35Turbo Model = "gpt-3.5-turbo"
)

// DefaultApiUrl Default OpenAI API endpoint
const (
	DefaultApiUrl = "https://api.openai.com/v1"
)

type Config struct {
	apiKey         string
	baseURL        string
	model          Model
	apiKeyRequired bool
}

func NewConfig(baseURL string, model Model) (Config, error) {
	config := Config{
		apiKey: strings.TrimSpace(firstNonEmpty(
			os.Getenv("OPENAI_API_KEY"),
			os.Getenv("TEXTUALAI_API_KEY"),
			os.Getenv("TERMCHAT_API_KEY"),
		)),
		baseURL: strings.TrimSpace(baseURL),
		model:   Model(strings.TrimSpace(string(model))),
	}

	if config.baseURL == "" {
		// Preserve historical behaviour: TEXTUALAI_API_URL remains the primary env var.
		config.baseURL = strings.TrimSpace(os.Getenv("TEXTUALAI_API_URL"))
		if config.baseURL == "" {
			config.baseURL = strings.TrimSpace(os.Getenv("TERMCHAT_API_URL"))
		}
		if config.baseURL == "" {
			config.baseURL = DefaultApiUrl
		}
	}

	if config.model == "" {
		// Termchat example uses TERMCHAT_MODEL; library users may prefer OPENAI_MODEL.
		m := strings.TrimSpace(os.Getenv("TERMCHAT_MODEL"))
		if m == "" {
			m = strings.TrimSpace(os.Getenv("OPENAI_MODEL"))
		}
		if m == "" {
			config.model = ModelGpt41
		} else {
			config.model = Model(m)
		}
	}

	// Default behaviour:
	//   - If the target looks like OpenAI's public API, an API key is required.
	//   - Otherwise, the key is optional (common for local OpenAI-compatible servers like Ollama).
	config.apiKeyRequired = isLikelyOpenAIBaseURL(config.baseURL)

	if strings.TrimSpace(config.baseURL) == "" {
		return Config{}, fmt.Errorf("textualopenai: missing base URL")
	}
	if strings.TrimSpace(string(config.model)) == "" {
		return Config{}, fmt.Errorf("textualopenai: missing model")
	}

	return config, nil
}

func (c Config) WithApiKey(apiKey string) Config {
	c.apiKey = strings.TrimSpace(apiKey)
	return c
}

// WithAPIKeyRequired controls whether the client enforces that an API key is present.
func (c Config) WithAPIKeyRequired(required bool) Config {
	c.apiKeyRequired = required
	return c
}

// BaseURL returns the resolved base URL used by the client.
func (c Config) BaseURL() string {
	return c.baseURL
}

// Model returns the resolved model identifier.
func (c Config) Model() Model {
	return c.model
}

// APIKeyRequired indicates whether the client enforces that an API key is present.
func (c Config) APIKeyRequired() bool {
	return c.apiKeyRequired
}

// HasAPIKey returns true when a non-empty API key is configured.
// This does not expose the key itself.
func (c Config) HasAPIKey() bool {
	return strings.TrimSpace(c.apiKey) != ""
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// isLikelyOpenAIBaseURL returns true when baseURL appears to target OpenAI's public API,
// in which case an API key is typically required.
func isLikelyOpenAIBaseURL(baseURL string) bool {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return false
	}
	host := strings.ToLower(strings.TrimSpace(u.Hostname()))
	if host == "" {
		return false
	}
	if host == "api.openai.com" {
		return true
	}
	return strings.HasSuffix(host, ".openai.com")
}
