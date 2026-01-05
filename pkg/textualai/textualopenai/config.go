package textualopenai

import (
	"fmt"
	"os"
)

// Model represents an OpenAI model identifier.
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
	apiKey  string
	baseURL string
	model   Model
}

func NewConfig(baseURL string, model Model) (Config, error) {
	config := Config{
		baseURL: baseURL,
		model:   model,
	}
	config.apiKey = os.Getenv("OPENAI_API_KEY")
	if config.baseURL == "" {
		config.baseURL = os.Getenv("TEXTUALAI_API_URL")
		if config.baseURL == "" {
			config.baseURL = DefaultApiUrl
		}
	}
	if config.model == "" {
		config.model = Model(os.Getenv("OPENAI_MODEL"))
		if config.model == "" {
			config.model = ModelGpt41
		}
	} else {
		switch config.model {
		case ModelGpt41:
		case ModelGpt4o:
		case ModelGpt4oMini:
		case ModelGpt52:
		case ModelGpt35Turbo:
		default:
			return config, nil //
			return Config{}, fmt.Errorf("textualopenai: invalid model type: %s", config.model)
		}
	}
	return config, nil
}

func (c Config) WithApiKey(apiKey string) Config {
	c.apiKey = apiKey
	return c
}
