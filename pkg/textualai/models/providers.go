package models

const (
	ProviderOpenAI ProviderName = "openai"
	ProviderOllama ProviderName = "ollama"
	ProviderXAI    ProviderName = "xai"
)

func init() {
	// Inject the provider names in the Models.
	for name, prov := range providers {
		models := make([]Model, len(prov.Models))
		for idx, model := range prov.Models {
			model.ProviderName = name
			models[idx] = model
		}
		providers[name] = Provider{
			Info:   prov.Info,
			Models: models,
		}
	}
}

func AllProviders() Providers {
	return providers
}

// providers is a small provider registry used for model parsing and capability gating.
// Add new providers here as the framework expands.
var providers = Providers{
	ProviderOpenAI: Provider{
		Info: ProviderInfo{
			Name:                        ProviderOpenAI,
			DisplayName:                 "OpenAI",
			DefaultBaseURL:              "https://api.openai.com/v1",
			APIKeyRequired:              true,
			SupportsConversation:        true,
			SupportsStrictFunctionTools: true,
		},
		Models: AllOpenAIModels,
	},
	ProviderOllama: Provider{
		Info: ProviderInfo{
			Name:                        ProviderOllama,
			DisplayName:                 "Ollama",
			DefaultBaseURL:              "http://localhost:11434/v1",
			APIKeyRequired:              false,
			SupportsConversation:        false,
			SupportsStrictFunctionTools: false,
		},
		Models: AllOllamaModels,
	},
	ProviderXAI: Provider{
		Info: ProviderInfo{
			Name:                        ProviderXAI,
			DisplayName:                 "xAI",
			DefaultBaseURL:              "https://api.x.ai/v1",
			APIKeyRequired:              true,
			SupportsConversation:        false,
			SupportsStrictFunctionTools: false,
		},
		Models: AllXAIModels,
	},
}
