package models

// Providers is a small provider registry used for model parsing and capability gating.
// Add new providers here as the framework expands.
var Providers = map[ProviderName]ProviderInfo{
	ProviderOpenAI: {
		Name:                        ProviderOpenAI,
		DisplayName:                 "OpenAI",
		DefaultBaseURL:              "https://api.openai.com/v1",
		APIKeyRequired:              true,
		SupportsConversation:        true,
		SupportsStrictFunctionTools: true,
	},
	ProviderOllama: {
		Name:                        ProviderOllama,
		DisplayName:                 "Ollama",
		DefaultBaseURL:              "http://localhost:11434/v1",
		APIKeyRequired:              false,
		SupportsConversation:        false,
		SupportsStrictFunctionTools: false,
	},
	ProviderXAI: {
		Name:                        ProviderXAI,
		DisplayName:                 "xAI",
		DefaultBaseURL:              "https://api.x.ai/v1",
		APIKeyRequired:              true,
		SupportsConversation:        false,
		SupportsStrictFunctionTools: false,
	},
}
