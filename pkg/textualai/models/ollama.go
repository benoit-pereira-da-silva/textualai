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

// Compile-time check: OllamaModel implements Model.
var _ Model = OllamaModel{}

func (m OllamaModel) ProviderInfo() ProviderInfo { p, _ := Providers[ProviderOllama]; return p }
func (m OllamaModel) Identifier() ModelID        { return m.ID }
func (m OllamaModel) DisplayName() string        { return m.Name }
func (m OllamaModel) Kind() string               { return m.Flavour }
func (m OllamaModel) TagList() []Tag             { return m.Tags }
func (m OllamaModel) Summary() string            { return m.Description }
func (m OllamaModel) KnownSnapshots() []string   { return nil }
func (m OllamaModel) IsDeprecated() bool         { return false }
func (m OllamaModel) KnownSizes() []string       { return m.Sizes }
func (m OllamaModel) LicenseText() string        { return m.License }

func (m OllamaModel) SupportsTools() bool     { return supportsTools(m.Tags) }
func (m OllamaModel) SupportsThinking() bool  { return supportsThinking(m.Flavour, m.Tags) }
func (m OllamaModel) SupportsVision() bool    { return supportsVision(m.Flavour, m.Tags) }
func (m OllamaModel) SupportsEmbedding() bool { return supportsEmbedding(m.Flavour, m.Tags) }

// OllamaModel contains metadata about an Ollama model.
type OllamaModel struct {
	ID          ModelID
	Name        string
	Flavour     string
	Tags        []Tag
	Description string
	Sizes       []string
	License     string
}

// Predefined model identifiers.
const (
	Nemotron3Nano          ModelID = "nemotron-3-nano"
	Functiongemma          ModelID = "functiongemma"
	Olmo3                  ModelID = "olmo-3"
	Gemini3FlashPreview    ModelID = "gemini-3-flash-preview"
	DevstralSmall2         ModelID = "devstral-small-2"
	Devstral2              ModelID = "devstral-2"
	Ministral3             ModelID = "ministral-3"
	Qwen3Vl                ModelID = "qwen3-vl"
	GptOss                 ModelID = "gpt-oss"
	DeepseekR1             ModelID = "deepseek-r1"
	Qwen3Coder             ModelID = "qwen3-coder"
	Gemma3                 ModelID = "gemma3"
	Llama31                ModelID = "llama3.1"
	Llama32                ModelID = "llama3.2"
	NomicEmbedText         ModelID = "nomic-embed-text"
	Mistral                ModelID = "mistral"
	Qwen25                 ModelID = "qwen2.5"
	Qwen3                  ModelID = "qwen3"
	Phi3                   ModelID = "phi3"
	Llama3                 ModelID = "llama3"
	Gemma2                 ModelID = "gemma2"
	Llava                  ModelID = "llava"
	Qwen25Coder            ModelID = "qwen2.5-coder"
	Phi4                   ModelID = "phi4"
	MxbaiEmbedLarge        ModelID = "mxbai-embed-large"
	Gemma                  ModelID = "gemma"
	Qwen                   ModelID = "qwen"
	Llama2                 ModelID = "llama2"
	Qwen2                  ModelID = "qwen2"
	MinicpmV               ModelID = "minicpm-v"
	Codellama              ModelID = "codellama"
	Dolphin3               ModelID = "dolphin3"
	Llama32Vision          ModelID = "llama3.2-vision"
	Olmo2                  ModelID = "olmo2"
	Tinyllama              ModelID = "tinyllama"
	MistralNemo            ModelID = "mistral-nemo"
	DeepseekV3             ModelID = "deepseek-v3"
	BgeM3                  ModelID = "bge-m3"
	Llama33                ModelID = "llama3.3"
	DeepseekCoder          ModelID = "deepseek-coder"
	Smollm2                ModelID = "smollm2"
	MistralSmall           ModelID = "mistral-small"
	AllMinilm              ModelID = "all-minilm"
	LlavaLlama3            ModelID = "llava-llama3"
	Qwq                    ModelID = "qwq"
	Codegemma              ModelID = "codegemma"
	Falcon3                ModelID = "falcon3"
	Granite31Moe           ModelID = "granite3.1-moe"
	Starcoder2             ModelID = "starcoder2"
	Mixtral                ModelID = "mixtral"
	SnowflakeArcticEmbed   ModelID = "snowflake-arctic-embed"
	Llama2Uncensored       ModelID = "llama2-uncensored"
	OrcaMini               ModelID = "orca-mini"
	DeepseekCoderV2        ModelID = "deepseek-coder-v2"
	Qwen25vl               ModelID = "qwen2.5vl"
	Cogito                 ModelID = "cogito"
	MistralSmall32         ModelID = "mistral-small3.2"
	Gemma3n                ModelID = "gemma3n"
	Llama4                 ModelID = "llama4"
	Deepscaler             ModelID = "deepscaler"
	DolphinPhi             ModelID = "dolphin-phi"
	Phi4Reasoning          ModelID = "phi4-reasoning"
	Magistral              ModelID = "magistral"
	Phi                    ModelID = "phi"
	DolphinMixtral         ModelID = "dolphin-mixtral"
	Granite33              ModelID = "granite3.3"
	DolphinLlama3          ModelID = "dolphin-llama3"
	Phi4Mini               ModelID = "phi4-mini"
	Openthinker            ModelID = "openthinker"
	Codestral              ModelID = "codestral"
	Smollm                 ModelID = "smollm"
	Granite32Vision        ModelID = "granite3.2-vision"
	Devstral               ModelID = "devstral"
	Wizardlm2              ModelID = "wizardlm2"
	DolphinMistral         ModelID = "dolphin-mistral"
	Deepcoder              ModelID = "deepcoder"
	Moondream              ModelID = "moondream"
	MistralSmall31         ModelID = "mistral-small3.1"
	CommandR               ModelID = "command-r"
	GraniteCode            ModelID = "granite-code"
	Hermes3                ModelID = "hermes3"
	Phi35                  ModelID = "phi3.5"
	Bakllava               ModelID = "bakllava"
	Granite4               ModelID = "granite4"
	Yi                     ModelID = "yi"
	Zephyr                 ModelID = "zephyr"
	Embeddinggemma         ModelID = "embeddinggemma"
	ExaoneDeep             ModelID = "exaone-deep"
	MistralLarge           ModelID = "mistral-large"
	WizardVicunaUncensored ModelID = "wizard-vicuna-uncensored"
	Opencoder              ModelID = "opencoder"
	Starcoder              ModelID = "starcoder"
	NousHermes             ModelID = "nous-hermes"
	Falcon                 ModelID = "falcon"
	DeepseekLlm            ModelID = "deepseek-llm"
	Openchat               ModelID = "openchat"
	Vicuna                 ModelID = "vicuna"
	DeepseekV2             ModelID = "deepseek-v2"
	Openhermes             ModelID = "openhermes"
	Codeqwen               ModelID = "codeqwen"
	ParaphraseMultilingual ModelID = "paraphrase-multilingual"
	Qwen2Math              ModelID = "qwen2-math"
	Codegeex4              ModelID = "codegeex4"
	DeepseekV31            ModelID = "deepseek-v3.1"
	MistralOpenorca        ModelID = "mistral-openorca"
	CommandRPlus           ModelID = "command-r-plus"
	Glm4                   ModelID = "glm4"
	Qwen3Embedding         ModelID = "qwen3-embedding"
	Aya                    ModelID = "aya"
	Llama2Chinese          ModelID = "llama2-chinese"
	Qwen3Next              ModelID = "qwen3-next"
	StableCode             ModelID = "stable-code"
	Tinydolphin            ModelID = "tinydolphin"
	NeuralChat             ModelID = "neural-chat"
	SnowflakeArcticEmbed2  ModelID = "snowflake-arctic-embed2"
	NousHermes2            ModelID = "nous-hermes2"
	Wizardcoder            ModelID = "wizardcoder"
	Sqlcoder               ModelID = "sqlcoder"
	Granite32              ModelID = "granite3.2"
	Stablelm2              ModelID = "stablelm2"
	YiCoder                ModelID = "yi-coder"
	Llama3Chatqa           ModelID = "llama3-chatqa"
	Granite3Dense          ModelID = "granite3-dense"
	Granite31Dense         ModelID = "granite3.1-dense"
	WizardMath             ModelID = "wizard-math"
	Llama3Gradient         ModelID = "llama3-gradient"
	Dolphincoder           ModelID = "dolphincoder"
	SamanthaMistral        ModelID = "samantha-mistral"
	R11776                 ModelID = "r1-1776"
	BgeLarge               ModelID = "bge-large"
	Internlm2              ModelID = "internlm2"
	Reflection             ModelID = "reflection"
	Exaone35               ModelID = "exaone3.5"
	Llama3GroqToolUse      ModelID = "llama3-groq-tool-use"
	StarlingLm             ModelID = "starling-lm"
	PhindCodellama         ModelID = "phind-codellama"
	LlavaPhi3              ModelID = "llava-phi3"
	Solar                  ModelID = "solar"
	Xwinlm                 ModelID = "xwinlm"
	LlamaGuard3            ModelID = "llama-guard3"
	NemotronMini           ModelID = "nemotron-mini"
	GraniteEmbedding       ModelID = "granite-embedding"
	AyaExpanse             ModelID = "aya-expanse"
	YarnLlama2             ModelID = "yarn-llama2"
	Granite3Moe            ModelID = "granite3-moe"
	AtheneV2               ModelID = "athene-v2"
	Meditron               ModelID = "meditron"
	Nemotron               ModelID = "nemotron"
	Dbrx                   ModelID = "dbrx"
	Tulu3                  ModelID = "tulu3"
	Orca2                  ModelID = "orca2"
	WizardlmUncensored     ModelID = "wizardlm-uncensored"
	StableBeluga           ModelID = "stable-beluga"
	ReaderLm               ModelID = "reader-lm"
	Medllama2              ModelID = "medllama2"
	Shieldgemma            ModelID = "shieldgemma"
	NousHermes2Mixtral     ModelID = "nous-hermes2-mixtral"
	LlamaPro               ModelID = "llama-pro"
	YarnMistral            ModelID = "yarn-mistral"
	Wizardlm               ModelID = "wizardlm"
	Smallthinker           ModelID = "smallthinker"
	Nexusraven             ModelID = "nexusraven"
	Phi4MiniReasoning      ModelID = "phi4-mini-reasoning"
	CommandR7b             ModelID = "command-r7b"
	Mathstral              ModelID = "mathstral"
	DeepseekV25            ModelID = "deepseek-v2.5"
	Codeup                 ModelID = "codeup"
	Everythinglm           ModelID = "everythinglm"
	StablelmZephyr         ModelID = "stablelm-zephyr"
	SolarPro               ModelID = "solar-pro"
	Falcon2                ModelID = "falcon2"
	DuckdbNsql             ModelID = "duckdb-nsql"
	CommandA               ModelID = "command-a"
	Magicoder              ModelID = "magicoder"
	Mistrallite            ModelID = "mistrallite"
	BespokeMinicheck       ModelID = "bespoke-minicheck"
	Nuextract              ModelID = "nuextract"
	Codebooga              ModelID = "codebooga"
	WizardVicuna           ModelID = "wizard-vicuna"
	Megadolphin            ModelID = "megadolphin"
	MarcoO1                ModelID = "marco-o1"
	DeepseekOcr            ModelID = "deepseek-ocr"
	FirefunctionV2         ModelID = "firefunction-v2"
	Notux                  ModelID = "notux"
	Notus                  ModelID = "notus"
	OpenOrcaPlatypus2      ModelID = "open-orca-platypus2"
	Goliath                ModelID = "goliath"
	Granite3Guardian       ModelID = "granite3-guardian"
	Sailor2                ModelID = "sailor2"
	Gemini3ProPreview      ModelID = "gemini-3-pro-preview"
	Alfred                 ModelID = "alfred"
	CommandR7bArabic       ModelID = "command-r7b-arabic"
	Glm46                  ModelID = "glm-4.6"
	GptOssSafeguard        ModelID = "gpt-oss-safeguard"
	MinimaxM2              ModelID = "minimax-m2"
	Cogito21               ModelID = "cogito-2.1"
	KimiK2                 ModelID = "kimi-k2"
	Rnj1                   ModelID = "rnj-1"
	Olmo31                 ModelID = "olmo-3.1"
	KimiK2Thinking         ModelID = "kimi-k2-thinking"
)

// OllamaModels holds a collection of OllamaModel metadata.
type OllamaModels []OllamaModel

// AllOllamaModels is the list of all available models.
var AllOllamaModels = OllamaModels{
	{
		ID:          Nemotron3Nano,
		Name:        "Nemotron 3 Nano",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud},
		Description: "A new Standard for Efficient, Open, and Intelligent Agentic OllamaModels",
		Sizes:       []string{"30b"},
		License:     "NVIDIA Open OllamaModel License",
	},
	{
		ID:          Functiongemma,
		Name:        "FunctionGemma",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "a specialized version of Google's Gemma 3 270M model fine-tuned explicitly for function calling",
		Sizes:       []string{"270m"},
		License:     "",
	},
	{
		ID:          Olmo3,
		Name:        "Olmo 3",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a series of Open language models designed to enable the science of language models. These models are pre-trained on the Dolma 3 dataset and post-trained on the Dolci datasets",
		Sizes:       []string{"7b", "32b"},
		License:     "",
	},
	{
		ID:          Gemini3FlashPreview,
		Name:        "Gemini 3 Flash",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud},
		Description: "offers frontier intelligence built for speed at a fraction of the cost",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          DevstralSmall2,
		Name:        "Devstral Small 2",
		Flavour:     "instruct",
		Tags:        []Tag{TagVision, TagTools, TagCloud},
		Description: "24B model that excels at using tools to explore codebases, editing multiple files and power software engineering agents",
		Sizes:       []string{"24b"},
		License:     "",
	},
	{
		ID:          Devstral2,
		Name:        "Devstral 2",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools, TagCloud},
		Description: "123B model that excels at using tools to explore codebases, editing multiple files and power software engineering agents",
		Sizes:       []string{"123b"},
		License:     "",
	},
	{
		ID:          Ministral3,
		Name:        "Ministral 3",
		Flavour:     "instruct",
		Tags:        []Tag{TagVision, TagTools, TagCloud},
		Description: "family is designed for edge deployment, capable of running on a wide range of hardware",
		Sizes:       []string{"3b", "8b", "14b"},
		License:     "",
	},
	{
		ID:          Qwen3Vl,
		Name:        "Qwen3-VL",
		Flavour:     "instruct",
		Tags:        []Tag{TagVision, TagTools, TagCloud},
		Description: "The most powerful vision-language model in the Qwen model family to date",
		Sizes:       []string{"2b", "4b", "8b", "30b", "32b", "235b"},
		License:     "",
	},
	{
		ID:          GptOss,
		Name:        "GPT-OSS",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools, TagThinking, TagCloud},
		Description: "OpenAI’s open-weight models designed for powerful reasoning, agentic tasks, and versatile developer use cases",
		Sizes:       []string{"20b", "120b"},
		License:     "",
	},
	{
		ID:          DeepseekR1,
		Name:        "DeepSeek-R1",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools, TagThinking},
		Description: "is a family of open reasoning models with performance approaching that of leading models, such as O3 and Gemini 2.5 Pro",
		Sizes:       []string{"1.5b", "7b", "8b", "14b", "32b", "70b", "671b"},
		License:     "",
	},
	{
		ID:          Qwen3Coder,
		Name:        "Qwen3 Coder",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools, TagCloud},
		Description: "Alibaba's performant long context models for agentic and coding tasks",
		Sizes:       []string{"30b", "480b"},
		License:     "",
	},
	{
		ID:          Gemma3,
		Name:        "Gemma 3",
		Flavour:     "instruct",
		Tags:        []Tag{TagVision, TagCloud},
		Description: "The current, most capable model that runs on a single GPU",
		Sizes:       []string{"270m", "1b", "4b", "12b", "27b"},
		License:     "",
	},
	{
		ID:          Llama31,
		Name:        "Llama 3.1",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "a new state-of-the-art model from Meta available in 8B, 70B and 405B parameter sizes",
		Sizes:       []string{"8b", "70b", "405b"},
		License:     "",
	},
	{
		ID:          Llama32,
		Name:        "Llama 3.2",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "Meta's Llama 3.2 goes small with 1B and 3B models",
		Sizes:       []string{"1b", "3b"},
		License:     "",
	},
	{
		ID:          NomicEmbedText,
		Name:        "Nomic Embed Text",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "A high-performing open embedding model with a large token context window",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          Mistral,
		Name:        "Mistral",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "The 7B model released by Mistral AI, updated to version 0.3",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Qwen25,
		Name:        "Qwen2.5",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "models are pretrained on Alibaba's latest large-scale dataset, encompassing up to 18 trillion tokens. The model supports up to 128K tokens and has multilingual support",
		Sizes:       []string{"0.5b", "1.5b", "3b", "7b", "14b", "32b", "72b"},
		License:     "",
	},
	{
		ID:          Qwen3,
		Name:        "Qwen3",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools, TagThinking},
		Description: "is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models",
		Sizes:       []string{"0.6b", "1.7b", "4b", "8b", "14b", "30b", "32b", "235b"},
		License:     "",
	},
	{
		ID:          Phi3,
		Name:        "Phi-3",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a family of lightweight 3B (Mini) and 14B (Medium) state-of-the-art open models by Microsoft",
		Sizes:       []string{"3.8b", "14b"},
		License:     "",
	},
	{
		ID:          Llama3,
		Name:        "Llama 3",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Meta Llama 3: The most capable openly available LLM to date",
		Sizes:       []string{"8b", "70b"},
		License:     "",
	},
	{
		ID:          Gemma2,
		Name:        "Gemma 2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Google Gemma 2 is a high-performing and efficient model available in three sizes: 2B, 9B, and 27B",
		Sizes:       []string{"2b", "9b", "27b"},
		License:     "",
	},
	{
		ID:          Llava,
		Name:        "LLaVA",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "is a novel end-to-end trained large multimodal model that combines a vision encoder and Vicuna for general-purpose visual and language understanding. Updated to version 1.6",
		Sizes:       []string{"7b", "13b", "34b"},
		License:     "",
	},
	{
		ID:          Qwen25Coder,
		Name:        "Qwen2.5 Coder",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "The latest series of Code-Specific Qwen models, with significant improvements in code generation, code reasoning, and code fixing",
		Sizes:       []string{"0.5b", "1.5b", "3b", "7b", "14b", "32b"},
		License:     "",
	},
	{
		ID:          Phi4,
		Name:        "Phi-4",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a 14B parameter, state-of-the-art open model from Microsoft",
		Sizes:       []string{"14b"},
		License:     "",
	},
	{
		ID:          MxbaiEmbedLarge,
		Name:        "MXBAI Embed Large",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "State-of-the-art large embedding model from mixedbread.ai",
		Sizes:       []string{"335m"},
		License:     "",
	},
	{
		ID:          Gemma,
		Name:        "Gemma",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Gemma is a family of lightweight, state-of-the-art open models built by Google DeepMind. Updated to version 1.1",
		Sizes:       []string{"2b", "7b"},
		License:     "",
	},
	{
		ID:          Qwen,
		Name:        "Qwen 1.5",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a series of large language models by Alibaba Cloud spanning from 0.5B to 110B parameters",
		Sizes:       []string{"0.5b", "1.8b", "4b", "7b", "14b", "32b", "72b", "110b"},
		License:     "",
	},
	{
		ID:          Llama2,
		Name:        "Llama 2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Llama 2 is a collection of foundation language models ranging from 7B to 70B parameters",
		Sizes:       []string{"7b", "13b", "70b"},
		License:     "",
	},
	{
		ID:          Qwen2,
		Name:        "Qwen2",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is a new series of large language models from Alibaba group",
		Sizes:       []string{"0.5b", "1.5b", "7b", "72b"},
		License:     "",
	},
	{
		ID:          MinicpmV,
		Name:        "MiniCPM-V",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "A series of multimodal LLMs (MLLMs) designed for vision-language understanding",
		Sizes:       []string{"8b"},
		License:     "",
	},
	{
		ID:          Codellama,
		Name:        "CodeLlama",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A large language model that can use text prompts to generate and discuss code",
		Sizes:       []string{"7b", "13b", "34b", "70b"},
		License:     "",
	},
	{
		ID:          Dolphin3,
		Name:        "Dolphin 3.0 (Llama 3.1 8B)",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is the next generation of the Dolphin series of instruct-tuned models designed to be the ultimate general purpose local model, enabling coding, math, agentic, function calling, and general use cases",
		Sizes:       []string{"8b"},
		License:     "",
	},
	{
		ID:          Llama32Vision,
		Name:        "Llama 3.2 Vision",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "is a collection of instruction-tuned image reasoning generative models in 11B and 90B sizes",
		Sizes:       []string{"11b", "90b"},
		License:     "",
	},
	{
		ID:          Olmo2,
		Name:        "OLMo 2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "OLMo 2 is a new family of 7B and 13B models trained on up to 5T tokens. These models are on par with or better than equivalently sized fully open models, and competitive with open-weight models such as Llama 3.1 on English academic benchmarks",
		Sizes:       []string{"7b", "13b"},
		License:     "",
	},
	{
		ID:          Tinyllama,
		Name:        "TinyLlama",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "The TinyLlama project is an open endeavor to train a compact 1.1B Llama model on 3 trillion tokens",
		Sizes:       []string{"1.1b"},
		License:     "",
	},
	{
		ID:          MistralNemo,
		Name:        "Mistral-Nemo",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "A state-of-the-art 12B model with 128k context length, built by Mistral AI in collaboration with NVIDIA",
		Sizes:       []string{"12b"},
		License:     "",
	},
	{
		ID:          DeepseekV3,
		Name:        "DeepSeek-V3",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token",
		Sizes:       []string{"671b"},
		License:     "",
	},
	{
		ID:          BgeM3,
		Name:        "BGE-M3",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "is a new model from BAAI distinguished for its versatility in Multi-Functionality, Multi-Linguality, and Multi-Granularity",
		Sizes:       []string{"567m"},
		License:     "",
	},
	{
		ID:          Llama33,
		Name:        "Llama 3.3",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "New state of the art 70B model. Llama 3.3 70B offers similar performance compared to the Llama 3.1 405B model",
		Sizes:       []string{"70b"},
		License:     "",
	},
	{
		ID:          DeepseekCoder,
		Name:        "DeepSeek Coder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a capable coding model trained on two trillion code and natural language tokens",
		Sizes:       []string{"1.3b", "6.7b", "33b"},
		License:     "",
	},
	{
		ID:          Smollm2,
		Name:        "SmolLM2",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters",
		Sizes:       []string{"135m", "360m", "1.7b"},
		License:     "",
	},
	{
		ID:          MistralSmall,
		Name:        "Mistral Small 3",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "sets a new benchmark in the “small” Large Language OllamaModels category below 70B",
		Sizes:       []string{"22b", "24b"},
		License:     "",
	},
	{
		ID:          AllMinilm,
		Name:        "all-MiniLM",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "Embedding models on very large sentence level datasets",
		Sizes:       []string{"22m", "33m"},
		License:     "",
	},
	{
		ID:          LlavaLlama3,
		Name:        "LLaVA-Llama3",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "A LLaVA model fine-tuned from Llama 3 Instruct with better scores in several benchmarks",
		Sizes:       []string{"8b"},
		License:     "",
	},
	{
		ID:          Qwq,
		Name:        "QwQ",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is the reasoning model of the Qwen series",
		Sizes:       []string{"32b"},
		License:     "",
	},
	{
		ID:          Codegemma,
		Name:        "CodeGemma",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a collection of powerful, lightweight models that can perform a variety of coding tasks like fill-in-the-middle code completion, code generation, natural language understanding, mathematical reasoning, and instruction following",
		Sizes:       []string{"2b", "7b"},
		License:     "",
	},
	{
		ID:          Falcon3,
		Name:        "Falcon3",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A family of efficient AI models under 10B parameters performant in science, math, and coding through innovative training techniques",
		Sizes:       []string{"1b", "3b", "7b", "10b"},
		License:     "",
	},
	{
		ID:          Granite31Moe,
		Name:        "Granite 3.1-MoE",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "The IBM Granite 1B and 3B models are long-context mixture of experts (MoE) Granite models from IBM designed for low latency usage",
		Sizes:       []string{"1b", "3b"},
		License:     "",
	},
	{
		ID:          Starcoder2,
		Name:        "StarCoder2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "StarCoder2 is the next generation of transparently trained open code LLMs that comes in three sizes: 3B, 7B and 15B parameters",
		Sizes:       []string{"3b", "7b", "15b"},
		License:     "",
	},
	{
		ID:          Mixtral,
		Name:        "Mixtral",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "A set of Mixture of Experts (MoE) model with open weights by Mistral AI in 8x7b and 8x22b parameter sizes",
		Sizes:       []string{"8x7b", "8x22b"},
		License:     "",
	},
	{
		ID:          SnowflakeArcticEmbed,
		Name:        "Snowflake Arctic Embed",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "A suite of text embedding models by Snowflake, optimized for performance",
		Sizes:       []string{"22m", "33m", "110m", "137m", "335m"},
		License:     "",
	},
	{
		ID:          Llama2Uncensored,
		Name:        "Llama2-Uncensored",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Uncensored Llama 2 model by George Sung and Jarrad Hope",
		Sizes:       []string{"7b", "70b"},
		License:     "",
	},
	{
		ID:          OrcaMini,
		Name:        "orca-mini",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A general-purpose model ranging from 3 billion parameters to 70 billion, suitable for entry-level hardware",
		Sizes:       []string{"3b", "7b", "13b", "70b"},
		License:     "",
	},
	{
		ID:          DeepseekCoderV2,
		Name:        "DeepSeek-Coder-v2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "An open-source Mixture-of-Experts code language model that achieves performance comparable to GPT4-Turbo in code-specific tasks",
		Sizes:       []string{"16b", "236b"},
		License:     "",
	},
	{
		ID:          Qwen25vl,
		Name:        "Qwen2.5-VL",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "Flagship vision-language model of Qwen and also a significant leap from the previous Qwen2-VL",
		Sizes:       []string{"3b", "7b", "32b", "72b"},
		License:     "",
	},
	{
		ID:          Cogito,
		Name:        "Cogito v1 Preview",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is a family of hybrid reasoning models by Deep Cogito that outperform the best available open models of the same size, including counterparts from LLaMA, DeepSeek, and Qwen across most standard benchmarks",
		Sizes:       []string{"3b", "8b", "14b", "32b", "70b"},
		License:     "",
	},
	{
		ID:          MistralSmall32,
		Name:        "Mistral Small 3.2",
		Flavour:     "instruct",
		Tags:        []Tag{TagVision, TagTools},
		Description: "An update to Mistral Small that improves on function calling, instruction following, and less repetition errors",
		Sizes:       []string{"24b"},
		License:     "",
	},
	{
		ID:          Gemma3n,
		Name:        "Gemma 3n",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Gemma 3n models are designed for efficient execution on everyday devices such as laptops, tablets or phones",
		Sizes:       []string{"e2b", "e4b"},
		License:     "",
	},
	{
		ID:          Llama4,
		Name:        "Llama 4",
		Flavour:     "instruct",
		Tags:        []Tag{TagVision, TagTools},
		Description: "Meta's latest collection of multimodal models",
		Sizes:       []string{"16x17b", "128x17b"},
		License:     "",
	},
	{
		ID:          Deepscaler,
		Name:        "Deepscaler",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A fine-tuned version of Deepseek-R1-Distilled-Qwen-1.5B that surpasses the performance of OpenAI’s o1-preview with just 1.5B parameters on popular math evaluations",
		Sizes:       []string{"1.5b"},
		License:     "",
	},
	{
		ID:          DolphinPhi,
		Name:        "Dolphin-Phi",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "2.7B uncensored Dolphin model by Eric Hartford, based on the Phi language model by Microsoft Research",
		Sizes:       []string{"2.7b"},
		License:     "",
	},
	{
		ID:          Phi4Reasoning,
		Name:        "Phi-4 (Reasoning)",
		Flavour:     "thinking",
		Tags:        nil,
		Description: "Phi 4 reasoning and reasoning plus are 14-billion parameter open-weight reasoning models that rival much larger models on complex reasoning tasks",
		Sizes:       []string{"14b"},
		License:     "",
	},
	{
		ID:          Magistral,
		Name:        "Magistral",
		Flavour:     "thinking",
		Tags:        []Tag{TagTools, TagThinking},
		Description: "is a small, efficient reasoning model with 24B parameters",
		Sizes:       []string{"24b"},
		License:     "",
	},
	{
		ID:          Phi,
		Name:        "Phi-2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "a 2.7B language model by Microsoft Research that demonstrates outstanding reasoning and language understanding capabilities",
		Sizes:       []string{"2.7b"},
		License:     "",
	},
	{
		ID:          DolphinMixtral,
		Name:        "Dolphin-Mixtral",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Uncensored, 8x7b and 8x22b fine-tuned models based on the Mixtral mixture of experts models that excels at coding tasks. Created by Eric Hartford",
		Sizes:       []string{"8x7b", "8x22b"},
		License:     "",
	},
	{
		ID:          Granite33,
		Name:        "Granite 3.3",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "IBM Granite 2B and 8B models are 128K context length language models that have been fine-tuned for improved reasoning and instruction-following capabilities",
		Sizes:       []string{"2b", "8b"},
		License:     "",
	},
	{
		ID:          DolphinLlama3,
		Name:        "Dolphin 2.9 (Llama 3)",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a new model with 8B and 70B sizes by Eric Hartford based on Llama 3 that has a variety of instruction, conversational, and coding skills",
		Sizes:       []string{"8b", "70b"},
		License:     "",
	},
	{
		ID:          Phi4Mini,
		Name:        "Phi-4-mini",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "brings significant enhancements in multilingual support, reasoning, and mathematics, and now, the long-awaited function calling feature is finally supported",
		Sizes:       []string{"3.8b"},
		License:     "",
	},
	{
		ID:          Openthinker,
		Name:        "OpenThinker",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A fully open-source family of reasoning models built using a dataset derived by distilling DeepSeek-R1",
		Sizes:       []string{"7b", "32b"},
		License:     "",
	},
	{
		ID:          Codestral,
		Name:        "Codestral",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is Mistral AI’s first-ever code model designed for code generation tasks",
		Sizes:       []string{"22b"},
		License:     "",
	},
	{
		ID:          Smollm,
		Name:        "SmolLM",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A family of small models with 135M, 360M, and 1.7B parameters, trained on a new high-quality dataset",
		Sizes:       []string{"135m", "360m", "1.7b"},
		License:     "",
	},
	{
		ID:          Granite32Vision,
		Name:        "Granite 3.2 Vision",
		Flavour:     "vision",
		Tags:        []Tag{TagVision, TagTools},
		Description: "A compact and efficient vision-language model, specifically designed for visual document understanding, enabling automated content extraction from tables, charts, infographics, plots, diagrams, and more",
		Sizes:       []string{"2b"},
		License:     "",
	},
	{
		ID:          Devstral,
		Name:        "Devstral",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "Devstral: the best open source model for coding agents",
		Sizes:       []string{"24b"},
		License:     "",
	},
	{
		ID:          Wizardlm2,
		Name:        "WizardLM2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "State of the art large language model from Microsoft AI with improved performance on complex chat, multilingual, reasoning and agent use cases",
		Sizes:       []string{"7b", "8x22b"},
		License:     "",
	},
	{
		ID:          DolphinMistral,
		Name:        "Dolphin-Mistral",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "The uncensored Dolphin model based on Mistral that excels at coding tasks. Updated to version 2.8",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Deepcoder,
		Name:        "DeepCoder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a fully open-Source 14B coder model at O3-mini level, with a 1.5B version also available",
		Sizes:       []string{"1.5b", "14b"},
		License:     "",
	},
	{
		ID:          Moondream,
		Name:        "moondream2",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "is a small vision language model designed to run efficiently on edge devices",
		Sizes:       []string{"1.8b"},
		License:     "",
	},
	{
		ID:          MistralSmall31,
		Name:        "Mistral Small 3.1",
		Flavour:     "vision",
		Tags:        []Tag{TagVision, TagTools},
		Description: "Building upon Mistral Small 3, Mistral Small 3.1 (2503) adds state-of-the-art vision understanding and enhances long context capabilities up to 128k tokens without compromising text performance",
		Sizes:       []string{"24b"},
		License:     "",
	},
	{
		ID:          CommandR,
		Name:        "Command R",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is a Large Language OllamaModel optimized for conversational interaction and long context tasks",
		Sizes:       []string{"35b"},
		License:     "",
	},
	{
		ID:          GraniteCode,
		Name:        "Granite-Code",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A family of open foundation models by IBM for Code Intelligence",
		Sizes:       []string{"3b", "8b", "20b", "34b"},
		License:     "",
	},
	{
		ID:          Hermes3,
		Name:        "Hermes 3",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is the latest version of the flagship Hermes series of LLMs by Nous Research",
		Sizes:       []string{"3b", "8b", "70b", "405b"},
		License:     "",
	},
	{
		ID:          Phi35,
		Name:        "Phi-3.5",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A lightweight AI model with 3.8 billion parameters with performance overtaking similarly and larger sized models",
		Sizes:       []string{"3.8b"},
		License:     "",
	},
	{
		ID:          Bakllava,
		Name:        "BakLLaVA",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "is a multimodal model consisting of the Mistral 7B base model augmented with the LLaVA architecture",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Granite4,
		Name:        "Granite 4",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "features improved instruction following (IF) and tool-calling capabilities, making them more effective in enterprise applications",
		Sizes:       []string{"350m", "1b", "3b"},
		License:     "",
	},
	{
		ID:          Yi,
		Name:        "Yi 1.5",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a high-performing, bilingual language model",
		Sizes:       []string{"6b", "9b", "34b"},
		License:     "",
	},
	{
		ID:          Zephyr,
		Name:        "Zephyr",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a series of fine-tuned versions of the Mistral and Mixtral models that are trained to act as helpful assistants",
		Sizes:       []string{"7b", "141b"},
		License:     "",
	},
	{
		ID:          Embeddinggemma,
		Name:        "EmbeddingGemma",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "is a 300M parameter embedding model from Google",
		Sizes:       []string{"300m"},
		License:     "",
	},
	{
		ID:          ExaoneDeep,
		Name:        "EXAONE Deep",
		Flavour:     "thinking",
		Tags:        nil,
		Description: "exhibits superior capabilities in various reasoning tasks including math and coding benchmarks, ranging from 2.4B to 32B parameters developed and released by LG AI Research",
		Sizes:       []string{"2.4b", "7.8b", "32b"},
		License:     "",
	},
	{
		ID:          MistralLarge,
		Name:        "Mistral Large 2",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is Mistral's new flagship model that is significantly more capable in code generation, mathematics, and reasoning with 128k context window and support for dozens of languages",
		Sizes:       []string{"123b"},
		License:     "",
	},
	{
		ID:          WizardVicunaUncensored,
		Name:        "Wizard Vicuna Uncensored",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a 7B, 13B, and 30B parameter model based on Llama 2 uncensored by Eric Hartford",
		Sizes:       []string{"7b", "13b", "30b"},
		License:     "",
	},
	{
		ID:          Opencoder,
		Name:        "OpenCoder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is an open and reproducible code LLM family which includes 1.5B and 8B models, supporting chat in English and Chinese languages",
		Sizes:       []string{"1.5b", "8b"},
		License:     "",
	},
	{
		ID:          Starcoder,
		Name:        "StarCoder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a code generation model trained on 80+ programming languages",
		Sizes:       []string{"1b", "3b", "7b", "15b"},
		License:     "",
	},
	{
		ID:          NousHermes,
		Name:        "Nous-Hermes",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "General use models based on Llama and Llama 2 from Nous Research",
		Sizes:       []string{"7b", "13b"},
		License:     "",
	},
	{
		ID:          Falcon,
		Name:        "Falcon",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A large language model built by the Technology Innovation Institute (TII) for use in summarization, text generation, and chat bots",
		Sizes:       []string{"7b", "40b", "180b"},
		License:     "",
	},
	{
		ID:          DeepseekLlm,
		Name:        "DeepSeek-LLM",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "An advanced language model crafted with 2 trillion bilingual tokens",
		Sizes:       []string{"7b", "67b"},
		License:     "",
	},
	{
		ID:          Openchat,
		Name:        "OpenChat",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A family of open-source models trained on a wide variety of data, surpassing ChatGPT on various benchmarks. Updated to version 3.5-0106",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Vicuna,
		Name:        "Vicuna",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "General use chat model based on Llama and Llama 2 with 2K to 16K context sizes",
		Sizes:       []string{"7b", "13b", "33b"},
		License:     "",
	},
	{
		ID:          DeepseekV2,
		Name:        "DeepSeek-V2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A strong, economical, and efficient Mixture-of-Experts language model",
		Sizes:       []string{"16b", "236b"},
		License:     "",
	},
	{
		ID:          Openhermes,
		Name:        "OpenHermes 2.5",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a 7B model fine-tuned by Teknium on Mistral with fully open datasets",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          Codeqwen,
		Name:        "CodeQwen1.5",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a large language model pretrained on a large amount of code data",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          ParaphraseMultilingual,
		Name:        "paraphrase-multilingual",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "Sentence-transformers model that can be used for tasks like clustering or semantic search",
		Sizes:       []string{"278m"},
		License:     "",
	},
	{
		ID:          Qwen2Math,
		Name:        "Qwen2 Math",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a series of specialized math language models built upon the Qwen2 LLMs, which significantly outperforms the mathematical capabilities of open-source models and even closed-source models (e.g., GPT4o)",
		Sizes:       []string{"1.5b", "7b", "72b"},
		License:     "",
	},
	{
		ID:          Codegeex4,
		Name:        "CodeGeeX4",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A versatile model for AI software development scenarios, including code completion",
		Sizes:       []string{"9b"},
		License:     "",
	},
	{
		ID:          DeepseekV31,
		Name:        "DeepSeek-V3.1-Terminus",
		Flavour:     "thinking",
		Tags:        []Tag{TagTools, TagThinking, TagCloud},
		Description: "is a hybrid model that supports both thinking mode and non-thinking mode",
		Sizes:       []string{"671b"},
		License:     "",
	},
	{
		ID:          MistralOpenorca,
		Name:        "Mistral OpenOrca",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a 7 billion parameter model, fine-tuned on top of the Mistral 7B model using the OpenOrca dataset",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          CommandRPlus,
		Name:        "Command R+",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is a powerful, scalable large language model purpose-built to excel at real-world enterprise use cases",
		Sizes:       []string{"104b"},
		License:     "",
	},
	{
		ID:          Glm4,
		Name:        "GLM4",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A strong multi-lingual general language model with competitive performance to Llama 3",
		Sizes:       []string{"9b"},
		License:     "",
	},
	{
		ID:          Qwen3Embedding,
		Name:        "Qwen3 Embedding",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "Building upon the foundational models of the Qwen3 series, Qwen3 Embedding provides a comprehensive range of text embeddings models in various sizes",
		Sizes:       []string{"0.6b", "4b", "8b"},
		License:     "",
	},
	{
		ID:          Aya,
		Name:        "Aya 23",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "released by Cohere, is a new family of state-of-the-art, multilingual models that support 23 languages",
		Sizes:       []string{"8b", "35b"},
		License:     "",
	},
	{
		ID:          Llama2Chinese,
		Name:        "Llama2-Chinese",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Llama 2 based model fine tuned to improve Chinese dialogue ability",
		Sizes:       []string{"7b", "13b"},
		License:     "",
	},
	{
		ID:          Qwen3Next,
		Name:        "Qwen3-Next",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools, TagThinking, TagCloud},
		Description: "The first installment in the Qwen3-Next series with strong performance in terms of both parameter efficiency and inference speed",
		Sizes:       []string{"80b"},
		License:     "",
	},
	{
		ID:          StableCode,
		Name:        "Stable Code 3B",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a coding model with instruct and code completion variants on par with models such as Code Llama 7B that are 2.5x larger",
		Sizes:       []string{"3b"},
		License:     "",
	},
	{
		ID:          Tinydolphin,
		Name:        "TinyDolphin",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "An experimental 1.1B parameter model trained on the new Dolphin 2.8 dataset by Eric Hartford and based on TinyLlama",
		Sizes:       []string{"1.1b"},
		License:     "",
	},
	{
		ID:          NeuralChat,
		Name:        "Neural-Chat",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A fine-tuned model based on Mistral with good coverage of domain and language",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          SnowflakeArcticEmbed2,
		Name:        "Snowflake Arctic Embed 2.0",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "Snowflake's frontier embedding model. Arctic Embed 2.0 adds multilingual support without sacrificing English performance or scalability",
		Sizes:       []string{"568m"},
		License:     "",
	},
	{
		ID:          NousHermes2,
		Name:        "Nous-Hermes 2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "The powerful family of models by Nous Research that excels at scientific discussion and coding tasks",
		Sizes:       []string{"10.7b", "34b"},
		License:     "",
	},
	{
		ID:          Wizardcoder,
		Name:        "WizardCoder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "State-of-the-art code generation model",
		Sizes:       []string{"33b"},
		License:     "",
	},
	{
		ID:          Sqlcoder,
		Name:        "SQLCoder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a code completion model fine-tuned on StarCoder for SQL generation tasks",
		Sizes:       []string{"7b", "15b"},
		License:     "",
	},
	{
		ID:          Granite32,
		Name:        "Granite-3.2",
		Flavour:     "thinking",
		Tags:        []Tag{TagTools},
		Description: "Granite-3.2 is a family of long-context AI models from IBM Granite fine-tuned for thinking capabilities",
		Sizes:       []string{"2b", "8b"},
		License:     "",
	},
	{
		ID:          Stablelm2,
		Name:        "Stable LM 2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a state-of-the-art 1.6B and 12B parameter language model trained on multilingual data in English, Spanish, German, Italian, French, Portuguese, and Dutch",
		Sizes:       []string{"1.6b", "12b"},
		License:     "",
	},
	{
		ID:          YiCoder,
		Name:        "Yi-Coder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a series of open-source code language models that delivers state-of-the-art coding performance with fewer than 10 billion parameters",
		Sizes:       []string{"1.5b", "9b"},
		License:     "",
	},
	{
		ID:          Llama3Chatqa,
		Name:        "Llama 3 ChatQA",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a model from NVIDIA based on Llama 3 that excels at conversational question answering (QA) and retrieval-augmented generation (RAG)",
		Sizes:       []string{"8b", "70b"},
		License:     "",
	},
	{
		ID:          Granite3Dense,
		Name:        "Granite 3 Dense",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "The IBM Granite 2B and 8B models are designed to support tool-based use cases and support for retrieval augmented generation (RAG), streamlining code generation, translation and bug fixing",
		Sizes:       []string{"2b", "8b"},
		License:     "",
	},
	{
		ID:          Granite31Dense,
		Name:        "Granite 3.1 Dense",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "The IBM Granite 2B and 8B models are text-only dense LLMs trained on over 12 trillion tokens of data, demonstrated significant improvements over their predecessors in performance and speed in IBM’s initial testing",
		Sizes:       []string{"2b", "8b"},
		License:     "",
	},
	{
		ID:          WizardMath,
		Name:        "Wizard-Math",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "OllamaModel focused on math and logic problems",
		Sizes:       []string{"7b", "13b", "70b"},
		License:     "",
	},
	{
		ID:          Llama3Gradient,
		Name:        "Llama 3 Gradient",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "This model extends LLaMA-3 8B's context length from 8k to over 1m tokens",
		Sizes:       []string{"8b", "70b"},
		License:     "",
	},
	{
		ID:          Dolphincoder,
		Name:        "DolphinCoder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A 7B and 15B uncensored variant of the Dolphin model family that excels at coding, based on StarCoder2",
		Sizes:       []string{"7b", "15b"},
		License:     "",
	},
	{
		ID:          SamanthaMistral,
		Name:        "Samantha-Mistral",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A companion assistant trained in philosophy, psychology, and personal relationships. Based on Mistral",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          R11776,
		Name:        "R1-1776",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A version of the DeepSeek-R1 model that has been post trained to provide unbiased, accurate, and factual information by Perplexity",
		Sizes:       []string{"70b", "671b"},
		License:     "",
	},
	{
		ID:          BgeLarge,
		Name:        "BGE-Large",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "Embedding model from BAAI mapping texts to vectors",
		Sizes:       []string{"335m"},
		License:     "",
	},
	{
		ID:          Internlm2,
		Name:        "InternLM2.5",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a 7B parameter model tailored for practical scenarios with outstanding reasoning capability",
		Sizes:       []string{"1m", "1.8b", "7b", "20b"},
		License:     "",
	},
	{
		ID:          Reflection,
		Name:        "Reflection",
		Flavour:     "thinking",
		Tags:        nil,
		Description: "A high-performing model trained with a new technique called Reflection-tuning that teaches a LLM to detect mistakes in its reasoning and correct course",
		Sizes:       []string{"70b"},
		License:     "",
	},
	{
		ID:          Exaone35,
		Name:        "EXAONE 3.5",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a collection of instruction-tuned bilingual (English and Korean) generative models ranging from 2.4B to 32B parameters, developed and released by LG AI Research",
		Sizes:       []string{"2.4b", "7.8b", "32b"},
		License:     "",
	},
	{
		ID:          Llama3GroqToolUse,
		Name:        "Llama3 Groq Tool-Use",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "A series of models from Groq that represent a significant advancement in open-source AI capabilities for tool use/function calling",
		Sizes:       []string{"8b", "70b"},
		License:     "",
	},
	{
		ID:          StarlingLm,
		Name:        "Starling LM",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Starling is a large language model trained by reinforcement learning from AI feedback focused on improving chatbot helpfulness",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          PhindCodellama,
		Name:        "Phind-CodeLlama",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Code generation model based on Code Llama",
		Sizes:       []string{"34b"},
		License:     "",
	},
	{
		ID:          LlavaPhi3,
		Name:        "LLaVA-Phi3",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "A new small LLaVA model fine-tuned from Phi 3 Mini",
		Sizes:       []string{"3.8b"},
		License:     "",
	},
	{
		ID:          Solar,
		Name:        "Solar",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A compact, yet powerful 10.7B large language model designed for single-turn conversation",
		Sizes:       []string{"10.7b"},
		License:     "",
	},
	{
		ID:          Xwinlm,
		Name:        "XWinLM",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Conversational model based on Llama 2 that performs competitively on various benchmarks",
		Sizes:       []string{"7b", "13b"},
		License:     "",
	},
	{
		ID:          LlamaGuard3,
		Name:        "Llama Guard 3",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a series of models fine-tuned for content safety classification of LLM inputs and responses",
		Sizes:       []string{"1b", "8b"},
		License:     "",
	},
	{
		ID:          NemotronMini,
		Name:        "Nemotron Mini",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "A commercial-friendly small language model by NVIDIA optimized for roleplay, RAG QA, and function calling",
		Sizes:       []string{"4b"},
		License:     "",
	},
	{
		ID:          GraniteEmbedding,
		Name:        "Granite Embedding",
		Flavour:     "embedding",
		Tags:        []Tag{TagEmbedding},
		Description: "The IBM Granite Embedding 30M and 278M models are text-only dense biencoder embedding models, with 30M available in English only and 278M serving multilingual use cases",
		Sizes:       []string{"30m", "278m"},
		License:     "",
	},
	{
		ID:          AyaExpanse,
		Name:        "Aya Expanse",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "Cohere For AI's language models trained to perform well across 23 different languages",
		Sizes:       []string{"8b", "32b"},
		License:     "",
	},
	{
		ID:          YarnLlama2,
		Name:        "Yarn-Llama2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "An extension of Llama 2 that supports a context of up to 128k tokens",
		Sizes:       []string{"7b", "13b"},
		License:     "",
	},
	{
		ID:          Granite3Moe,
		Name:        "Granite 3-MoE",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "The IBM Granite 1B and 3B models are the first mixture of experts (MoE) Granite models from IBM designed for low latency usage",
		Sizes:       []string{"1b", "3b"},
		License:     "",
	},
	{
		ID:          AtheneV2,
		Name:        "Athene-V2",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "is a 72B parameter model which excels at code completion, mathematics, and log extraction tasks",
		Sizes:       []string{"72b"},
		License:     "",
	},
	{
		ID:          Meditron,
		Name:        "Meditron",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Open-source medical large language model adapted from Llama 2 to the medical domain",
		Sizes:       []string{"7b", "70b"},
		License:     "",
	},
	{
		ID:          Nemotron,
		Name:        "Nemotron 70B Instruct",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "Llama-3.1-Nemotron-70B-Instruct is a large language model customized by NVIDIA to improve the helpfulness of LLM generated responses to user queries",
		Sizes:       []string{"70b"},
		License:     "",
	},
	{
		ID:          Dbrx,
		Name:        "DBRX",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is an open, general-purpose LLM created by Databricks",
		Sizes:       []string{"132b"},
		License:     "",
	},
	{
		ID:          Tulu3,
		Name:        "Tülu 3",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a leading instruction following model family, offering fully open-source data, code, and recipes by the The Allen Institute for AI",
		Sizes:       []string{"8b", "70b"},
		License:     "",
	},
	{
		ID:          Orca2,
		Name:        "Orca 2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is built by Microsoft research, and are a fine-tuned version of Meta's Llama 2 models. The model is designed to excel particularly in reasoning",
		Sizes:       []string{"7b", "13b"},
		License:     "",
	},
	{
		ID:          WizardlmUncensored,
		Name:        "WizardLM-Uncensored",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Uncensored version of Wizard LM model",
		Sizes:       []string{"13b"},
		License:     "",
	},
	{
		ID:          StableBeluga,
		Name:        "Stable Beluga",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Llama 2 based model fine tuned on an Orca-style dataset. Originally called Free Willy",
		Sizes:       []string{"7b", "13b", "70b"},
		License:     "",
	},
	{
		ID:          ReaderLm,
		Name:        "Reader-LM",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A series of models that convert HTML content to Markdown content, which is useful for content conversion tasks",
		Sizes:       []string{"0.5b", "1.5b"},
		License:     "",
	},
	{
		ID:          Medllama2,
		Name:        "MedLlama2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Fine-tuned Llama 2 model to answer medical questions based on an open source medical dataset",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Shieldgemma,
		Name:        "ShieldGemma",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is set of instruction tuned models for evaluating the safety of text prompt input and text output responses against a set of defined safety policies",
		Sizes:       []string{"2b", "9b", "27b"},
		License:     "",
	},
	{
		ID:          NousHermes2Mixtral,
		Name:        "Nous-Hermes 2 (Mixtral)",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "The Nous Hermes 2 model from Nous Research, now trained over Mixtral",
		Sizes:       []string{"8x7b"},
		License:     "",
	},
	{
		ID:          LlamaPro,
		Name:        "Llama-Pro",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "An expansion of Llama 2 that specializes in integrating both general language understanding and domain-specific knowledge, particularly in programming and mathematics",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          YarnMistral,
		Name:        "Yarn-Mistral",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "An extension of Mistral to support context windows of 64K or 128K",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Wizardlm,
		Name:        "WizardLM",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "General use model based on Llama 2",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          Smallthinker,
		Name:        "SmallThinker",
		Flavour:     "thinking",
		Tags:        []Tag{TagTools},
		Description: "A new small reasoning model fine-tuned from the Qwen 2.5 3B Instruct model",
		Sizes:       []string{"3b"},
		License:     "",
	},
	{
		ID:          Nexusraven,
		Name:        "Nexus Raven",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a 13B instruction tuned model for function calling tasks",
		Sizes:       []string{"13b"},
		License:     "",
	},
	{
		ID:          Phi4MiniReasoning,
		Name:        "Phi-4-mini (Reasoning)",
		Flavour:     "thinking",
		Tags:        nil,
		Description: "is a lightweight open model that balances efficiency with advanced reasoning ability",
		Sizes:       []string{"3.8b"},
		License:     "",
	},
	{
		ID:          CommandR7b,
		Name:        "Command R7B",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "The smallest model in Cohere's R series delivers top-tier speed, efficiency, and quality to build powerful AI applications on commodity GPUs and edge devices",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Mathstral,
		Name:        "MathΣtral",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "a 7B model designed for math reasoning and scientific discovery by Mistral AI",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          DeepseekV25,
		Name:        "DeepSeek-V2.5",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "An upgraded version of DeepSeek-V2 that integrates the general and coding abilities of both DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct",
		Sizes:       []string{"236b"},
		License:     "",
	},
	{
		ID:          Codeup,
		Name:        "CodeUp",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Great code generation model based on Llama2",
		Sizes:       []string{"13b"},
		License:     "",
	},
	{
		ID:          Everythinglm,
		Name:        "EverythingLM",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Uncensored Llama2 based model with support for a 16K context window",
		Sizes:       []string{"13b"},
		License:     "",
	},
	{
		ID:          StablelmZephyr,
		Name:        "StableLM-Zephyr",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A lightweight chat model allowing accurate, and responsive output without requiring high-end hardware",
		Sizes:       []string{"3b"},
		License:     "",
	},
	{
		ID:          SolarPro,
		Name:        "Solar Pro",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Solar Pro Preview: an advanced large language model (LLM) with 22 billion parameters designed to fit into a single GPU",
		Sizes:       []string{"22b"},
		License:     "",
	},
	{
		ID:          Falcon2,
		Name:        "Falcon2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is an 11B parameters causal decoder-only model built by TII and trained over 5T tokens",
		Sizes:       []string{"11b"},
		License:     "",
	},
	{
		ID:          DuckdbNsql,
		Name:        "DuckDB-SQL",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "7B parameter text-to-SQL model made by MotherDuck and Numbers Station",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          CommandA,
		Name:        "Command-A",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "111 billion parameter model optimized for demanding enterprises that require fast, secure, and high-quality AI tools",
		Sizes:       []string{"111b"},
		License:     "",
	},
	{
		ID:          Magicoder,
		Name:        "Magicoder",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a family of 7B parameter models trained on 75K synthetic instruction data using OSS-Instruct, a novel approach to enlightening LLMs with open-source code snippets",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Mistrallite,
		Name:        "MistralLite",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a fine-tuned model based on Mistral with enhanced capabilities of processing long contexts",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          BespokeMinicheck,
		Name:        "Bespoke-MiniCheck",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A state-of-the-art fact-checking model developed by Bespoke Labs",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Nuextract,
		Name:        "nuExtract",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A 3.8B model fine-tuned on a private high-quality synthetic dataset for information extraction, based on Phi-3",
		Sizes:       []string{"3.8b"},
		License:     "",
	},
	{
		ID:          Codebooga,
		Name:        "CodeBooga",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A high-performing code instruct model created by merging two existing code models",
		Sizes:       []string{"34b"},
		License:     "",
	},
	{
		ID:          WizardVicuna,
		Name:        "Wizard Vicuna",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a 13B parameter model based on Llama 2 trained by MelodysDreamj",
		Sizes:       []string{"13b"},
		License:     "",
	},
	{
		ID:          Megadolphin,
		Name:        "MegaDolphin-2.2-120b",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "is a transformation of Dolphin-2.2-70b created by interleaving the model with itself",
		Sizes:       []string{"120b"},
		License:     "",
	},
	{
		ID:          MarcoO1,
		Name:        "Marco-O1",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "An open large reasoning model for real-world solutions by the Alibaba International Digital Commerce Group (AIDC-AI)",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          DeepseekOcr,
		Name:        "DeepSeek-OCR",
		Flavour:     "vision",
		Tags:        []Tag{TagVision},
		Description: "is a vision-language model that can perform token-efficient OCR",
		Sizes:       []string{"3b"},
		License:     "",
	},
	{
		ID:          FirefunctionV2,
		Name:        "FireFunction-v2",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "An open weights function calling model based on Llama 3, competitive with GPT-4o function calling capabilities",
		Sizes:       []string{"70b"},
		License:     "",
	},
	{
		ID:          Notux,
		Name:        "NoTux",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A top-performing mixture of experts model, fine-tuned with high-quality data",
		Sizes:       []string{"8x7b"},
		License:     "",
	},
	{
		ID:          Notus,
		Name:        "Notus",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A 7B chat model fine-tuned with high-quality data and based on Zephyr",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          OpenOrcaPlatypus2,
		Name:        "Open-Orca-Platypus2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "Merge of the Open Orca OpenChat model and the Garage-bAInd Platypus 2 model. Designed for chat and code generation",
		Sizes:       []string{"13b"},
		License:     "",
	},
	{
		ID:          Goliath,
		Name:        "Goliath",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A language model created by combining two fine-tuned Llama 2 70B models into one",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          Granite3Guardian,
		Name:        "Granite 3 Guardian",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "The IBM Granite Guardian 3.0 2B and 8B models are designed to detect risks in prompts and/or responses",
		Sizes:       []string{"2b", "8b"},
		License:     "",
	},
	{
		ID:          Sailor2,
		Name:        "Sailor2",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "are multilingual language models made for South-East Asia. Available in 1B, 8B, and 20B parameter sizes",
		Sizes:       []string{"1b", "8b", "20b"},
		License:     "",
	},
	{
		ID:          Gemini3ProPreview,
		Name:        "Gemini 3 Pro (Preview)",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud},
		Description: "Google's most intelligent model with SOTA reasoning and multimodal understanding, and powerful agentic and vibe coding capabilities",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          Alfred,
		Name:        "Alfred",
		Flavour:     "instruct",
		Tags:        nil,
		Description: "A robust conversational model designed to be used for both chat and instruct use cases",
		Sizes:       []string{"40b"},
		License:     "",
	},
	{
		ID:          CommandR7bArabic,
		Name:        "Command R7B (Arabic)",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "A new state-of-the-art version of the lightweight Command R7B model that excels in advanced Arabic language capabilities for enterprises in the Middle East and Northern Africa",
		Sizes:       []string{"7b"},
		License:     "",
	},
	{
		ID:          Glm46,
		Name:        "GLM-4.6",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud},
		Description: "Advanced agentic, reasoning and coding capabilities",
		Sizes:       []string{"40b"},
		License:     "",
	},
	{
		ID:          GptOssSafeguard,
		Name:        "GPT-OSS-Safeguard",
		Flavour:     "thinking",
		Tags:        []Tag{TagTools, TagThinking},
		Description: "gpt-oss-safeguard-20b and gpt-oss-safeguard-120b are safety reasoning models built-upon gpt-oss",
		Sizes:       []string{"20b", "120b"},
		License:     "",
	},
	{
		ID:          MinimaxM2,
		Name:        "MiniMax M2",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud},
		Description: "MiniMax M2 is a high-efficiency large language model built for coding and agentic workflows",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          Cogito21,
		Name:        "Cogito v2.1",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud},
		Description: "The Cogito v2.1 LLMs are instruction tuned generative models. All models are released under MIT license for commercial use",
		Sizes:       []string{"671b"},
		License:     "MIT License",
	},
	{
		ID:          KimiK2,
		Name:        "Kimi K2",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud},
		Description: "A state-of-the-art mixture-of-experts (MoE) language model. Kimi K2-Instruct-0905 demonstrates significant improvements in performance on public benchmarks and real-world coding agent tasks",
		Sizes:       nil,
		License:     "",
	},
	{
		ID:          Rnj1,
		Name:        "Rnj-1",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools, TagCloud},
		Description: "is a family of 8B parameter open-weight, dense models trained from scratch by Essential AI, optimized for code and STEM with capabilities on par with SOTA open-weight models",
		Sizes:       []string{"8b"},
		License:     "",
	},
	{
		ID:          Olmo31,
		Name:        "Olmo 3.1",
		Flavour:     "instruct",
		Tags:        []Tag{TagTools},
		Description: "Olmo is a series of Open language models designed to enable the science of language models. These models are pre-trained on the Dolma 3 dataset and post-trained on the Dolci datasets",
		Sizes:       []string{"32b"},
		License:     "",
	},
	{
		ID:          KimiK2Thinking,
		Name:        "Kimi K2 Thinking",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud},
		Description: "Moonshot AI's best open-source thinking model",
		Sizes:       nil,
		License:     "",
	},
}
