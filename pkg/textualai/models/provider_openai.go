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

// OpenAI Platform model identifiers.
//
// NOTE: These are API model IDs (the string passed in the `model` parameter), not marketing names.
const (
	// Frontier reasoning (GPT-5 family).
	GPT52Pro ModelID = "gpt-5.2-pro"
	GPT52    ModelID = "gpt-5.2"
	GPT51    ModelID = "gpt-5.1"

	GPT5Pro  ModelID = "gpt-5-pro"
	GPT5     ModelID = "gpt-5"
	GPT5Mini ModelID = "gpt-5-mini"
	GPT5Nano ModelID = "gpt-5-nano"

	// Coding agents (Codex).
	GPT51CodexMini ModelID = "gpt-5.1-codex-mini"
	GPT51Codex     ModelID = "gpt-5.1-codex"
	GPT51CodexMax  ModelID = "gpt-5.1-codex-max"
	GPT5Codex      ModelID = "gpt-5-codex"

	// General purpose multimodal (GPT-4.x / GPT-4o).
	GPT41     ModelID = "gpt-4.1"
	GPT41Mini ModelID = "gpt-4.1-mini"
	GPT41Nano ModelID = "gpt-4.1-nano"

	GPT4o     ModelID = "gpt-4o"
	GPT4oMini ModelID = "gpt-4o-mini"

	// Reasoning (o-series).
	O3Pro     ModelID = "o3-pro"
	O3        ModelID = "o3"
	O4Mini    ModelID = "o4-mini"
	O3Mini    ModelID = "o3-mini"
	O1Pro     ModelID = "o1-pro"
	O1        ModelID = "o1"
	O1Mini    ModelID = "o1-mini"
	O1Preview ModelID = "o1-preview"

	// Tool-specialized.
	O3DeepResearch     ModelID = "o3-deep-research"
	O4MiniDeepResearch ModelID = "o4-mini-deep-research"

	GPT4oSearchPreview     ModelID = "gpt-4o-search-preview"
	GPT4oMiniSearchPreview ModelID = "gpt-4o-mini-search-preview"
	ComputerUsePreview     ModelID = "computer-use-preview"

	// Open-weight (served on OpenAI platform).
	GPTOss120B ModelID = "gpt-oss-120b"
	GPTOss20B  ModelID = "gpt-oss-20b"

	// Realtime & audio chat.
	GPTRealtime     ModelID = "gpt-realtime"
	GPTRealtimeMini ModelID = "gpt-realtime-mini"
	GPTAudio        ModelID = "gpt-audio"
	GPTAudioMini    ModelID = "gpt-audio-mini"

	// Legacy preview audio/realtime IDs (still documented).
	GPT4oAudioPreview        ModelID = "gpt-4o-audio-preview"
	GPT4oMiniAudioPreview    ModelID = "gpt-4o-mini-audio-preview"
	GPT4oRealtimePreview     ModelID = "gpt-4o-realtime-preview"
	GPT4oMiniRealtimePreview ModelID = "gpt-4o-mini-realtime-preview"

	// Speech & transcription.
	GPT4oMiniTTS           ModelID = "gpt-4o-mini-tts"
	GPT4oTranscribe        ModelID = "gpt-4o-transcribe"
	GPT4oMiniTranscribe    ModelID = "gpt-4o-mini-transcribe"
	GPT4oTranscribeDiarize ModelID = "gpt-4o-transcribe-diarize"

	Whisper1 ModelID = "whisper-1"
	TTS1     ModelID = "tts-1"
	TTS1HD   ModelID = "tts-1-hd"

	// Image generation & editing.
	GPTImage15         ModelID = "gpt-image-1.5"
	GPTImage1          ModelID = "gpt-image-1"
	GPTImage1Mini      ModelID = "gpt-image-1-mini"
	ChatGPTImageLatest ModelID = "chatgpt-image-latest"

	// Video generation.
	Sora2    ModelID = "sora-2"
	Sora2Pro ModelID = "sora-2-pro"

	// Embeddings.
	TextEmbedding3Large ModelID = "text-embedding-3-large"
	TextEmbedding3Small ModelID = "text-embedding-3-small"

	// Moderation.
	OmniModerationLatest ModelID = "omni-moderation-latest"

	// Deprecated image models (kept for compatibility).
	DallE2 ModelID = "dall-e-2"
	DallE3 ModelID = "dall-e-3"
)

// AllOpenAIModels is a curated list of OpenAI Platform models.
//
// Notes:
//   - This list intentionally focuses on developer-facing, documented model IDs.
//   - Some models are marked Deprecated when the docs label them as such.
//   - Snapshots are provided for frequently pinned models; when in doubt, use the base ID.
var AllOpenAIModels = Models{
	// Frontier reasoning (GPT-5 family).
	{
		ID:          GPT52Pro,
		Name:        "GPT-5.2 pro",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Flagship GPT-5.2 reasoning model (highest quality).",
		Snapshots:   []string{"gpt-5.2-pro-2025-12-11"},
	},
	{
		ID:          GPT52,
		Name:        "GPT-5.2",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "GPT-5.2 reasoning model balancing quality, latency, and cost.",
		Snapshots:   []string{"gpt-5.2-2025-12-11"},
	},
	{
		ID:          GPT51,
		Name:        "GPT-5.1",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Previous-generation GPT-5 reasoning model.",
		Snapshots:   []string{"gpt-5.1-2025-11-13"},
	},
	{
		ID:          GPT5,
		Name:        "GPT-5",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "General-purpose GPT-5 reasoning model.",
		Snapshots:   []string{"gpt-5-2025-08-07"},
	},
	{
		ID:          GPT5Pro,
		Name:        "GPT-5 pro",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Higher-capability GPT-5 tier for demanding reasoning tasks.",
	},
	{
		ID:          GPT5Mini,
		Name:        "GPT-5 mini",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Cost-efficient GPT-5 reasoning model.",
		Snapshots:   []string{"gpt-5-mini-2025-08-07"},
	},
	{
		ID:          GPT5Nano,
		Name:        "GPT-5 nano",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Lowest-latency GPT-5 reasoning model.",
		Snapshots:   []string{"gpt-5-nano-2025-08-07"},
	},

	// Coding agents (Codex).
	{
		ID:          GPT51CodexMini,
		Name:        "GPT-5.1 Codex mini",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Fast, cost-efficient model for coding agent workflows.",
	},
	{
		ID:          GPT51Codex,
		Name:        "GPT-5.1 Codex",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Coding-focused model tuned for agentic code editing and tool use.",
	},
	{
		ID:          GPT51CodexMax,
		Name:        "GPT-5.1 Codex max",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Highest-capability Codex model for complex repositories and long-running agents.",
	},
	{
		ID:          GPT5Codex,
		Name:        "GPT-5 Codex",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "GPT-5 Codex model for coding tasks and agents.",
	},

	// General-purpose multimodal (GPT-4.x / GPT-4o).
	{
		ID:          GPT41,
		Name:        "GPT-4.1",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools, TagVision},
		Description: "High-quality general model (text + image input).",
		Snapshots:   []string{"gpt-4.1-2025-04-14"},
	},
	{
		ID:          GPT41Mini,
		Name:        "GPT-4.1 mini",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools, TagVision},
		Description: "Smaller GPT-4.1 variant for lower latency and cost.",
		Snapshots:   []string{"gpt-4.1-mini-2025-04-14"},
	},
	{
		ID:          GPT41Nano,
		Name:        "GPT-4.1 nano",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools, TagVision},
		Description: "Smallest GPT-4.1 tier for very low latency.",
		Snapshots:   []string{"gpt-4.1-nano-2025-04-14"},
	},
	{
		ID:          GPT4o,
		Name:        "GPT-4o",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools, TagVision},
		Description: "Omni model (fast, multimodal) for general assistant workloads.",
	},
	{
		ID:          GPT4oMini,
		Name:        "GPT-4o mini",
		Flavour:     "instruct",
		Tags:        []Tag{TagCloud, TagTools, TagVision},
		Description: "Smaller GPT-4o for lower cost and latency.",
	},

	// Reasoning (o-series).
	{
		ID:          O3Pro,
		Name:        "o3-pro",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Highest-capability o-series reasoning model.",
		Snapshots:   []string{"o3-pro-2025-06-10"},
	},
	{
		ID:          O3,
		Name:        "o3",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Strong reasoning model for complex tasks.",
		Snapshots:   []string{"o3-2025-04-16"},
	},
	{
		ID:          O4Mini,
		Name:        "o4-mini",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Fast, cost-efficient reasoning model.",
		Snapshots:   []string{"o4-mini-2025-04-16"},
	},
	{
		ID:          O3Mini,
		Name:        "o3-mini",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Lightweight o-series reasoning model.",
		Snapshots:   []string{"o3-mini-2025-04-16"},
	},
	{
		ID:          O1Pro,
		Name:        "o1-pro",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Higher-capability o1 tier.",
		Snapshots:   []string{"o1-pro-2025-03-19"},
	},
	{
		ID:          O1,
		Name:        "o1",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Original o-series reasoning model.",
		Snapshots:   []string{"o1-2024-12-17"},
	},
	{
		ID:          O1Mini,
		Name:        "o1-mini (deprecated)",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Deprecated smaller o1 variant.",
		Snapshots:   []string{"o1-mini-2024-09-12"},
		Deprecated:  true,
	},
	{
		ID:          O1Preview,
		Name:        "o1-preview (deprecated)",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools, TagVision},
		Description: "Deprecated o1 preview model.",
		Deprecated:  true,
	},

	// Tool-specialized models.
	{
		ID:          GPT4oSearchPreview,
		Name:        "GPT-4o Search Preview",
		Flavour:     "tools",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Search-augmented preview model (tool-specific).",
	},
	{
		ID:          GPT4oMiniSearchPreview,
		Name:        "GPT-4o mini Search Preview",
		Flavour:     "tools",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Lower-cost search-augmented preview model.",
	},
	{
		ID:          ComputerUsePreview,
		Name:        "Computer Use Preview",
		Flavour:     "tools",
		Tags:        []Tag{TagCloud, TagTools, TagVision},
		Description: "Model specialized for computer-use style interactions.",
		Snapshots:   []string{"computer-use-preview-2025-03-11"},
	},
	{
		ID:          O3DeepResearch,
		Name:        "o3-deep-research",
		Flavour:     "tools",
		Tags:        []Tag{TagCloud, TagTools, TagThinking, TagVision},
		Description: "Deep research model for multi-step research tasks with internet search.",
		Snapshots:   []string{"o3-deep-research-2025-06-26"},
	},
	{
		ID:          O4MiniDeepResearch,
		Name:        "o4-mini-deep-research",
		Flavour:     "tools",
		Tags:        []Tag{TagCloud, TagTools, TagThinking, TagVision},
		Description: "Faster, more affordable deep research model.",
		Snapshots:   []string{"o4-mini-deep-research-2025-06-26"},
	},

	// Open-weight (served on OpenAI platform).
	{
		ID:          GPTOss120B,
		Name:        "gpt-oss-120b",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools},
		Description: "Open-weight reasoning model (120B).",
	},
	{
		ID:          GPTOss20B,
		Name:        "gpt-oss-20b",
		Flavour:     "thinking",
		Tags:        []Tag{TagCloud, TagThinking, TagTools},
		Description: "Open-weight reasoning model (20B).",
	},

	// Realtime & audio chat models.
	{
		ID:          GPTRealtime,
		Name:        "gpt-realtime",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud, TagTools, TagVision},
		Description: "General-availability realtime model (text + audio I/O, optional image input).",
		Snapshots:   []string{"gpt-realtime-2025-08-28"},
	},
	{
		ID:          GPTRealtimeMini,
		Name:        "gpt-realtime-mini",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud, TagTools, TagVision},
		Description: "Cost-efficient realtime model.",
		Snapshots:   []string{"gpt-realtime-mini-2025-10-06", "gpt-realtime-mini-2025-12-15"},
	},
	{
		ID:          GPTAudio,
		Name:        "gpt-audio",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Audio-capable chat model (text + audio I/O).",
		Snapshots:   []string{"gpt-audio-2025-08-28"},
	},
	{
		ID:          GPTAudioMini,
		Name:        "gpt-audio-mini",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Smaller audio-capable chat model.",
		Snapshots:   []string{"gpt-audio-mini-2025-10-06", "gpt-audio-mini-2025-12-15"},
	},

	// Legacy preview audio/realtime IDs (still documented).
	{
		ID:          GPT4oAudioPreview,
		Name:        "GPT-4o Audio (preview)",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Preview audio chat model (GPT-4o).",
		Snapshots:   []string{"gpt-4o-audio-preview-2025-06-03", "gpt-4o-audio-preview-2024-12-17", "gpt-4o-audio-preview-2024-10-01"},
	},
	{
		ID:          GPT4oMiniAudioPreview,
		Name:        "GPT-4o mini Audio (preview)",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Preview audio chat model (GPT-4o mini).",
		Snapshots:   []string{"gpt-4o-mini-audio-preview-2024-12-17"},
	},
	{
		ID:          GPT4oRealtimePreview,
		Name:        "GPT-4o Realtime (preview)",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Preview realtime model (GPT-4o).",
		Snapshots:   []string{"gpt-4o-realtime-preview-2025-06-03", "gpt-4o-realtime-preview-2024-12-17", "gpt-4o-realtime-preview-2024-10-01"},
	},
	{
		ID:          GPT4oMiniRealtimePreview,
		Name:        "GPT-4o mini Realtime (preview)",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud, TagTools},
		Description: "Preview realtime model (GPT-4o mini).",
		Snapshots:   []string{"gpt-4o-mini-realtime-preview-2024-12-17"},
	},

	// Speech & transcription.
	{
		ID:          GPT4oMiniTTS,
		Name:        "GPT-4o mini TTS",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud},
		Description: "Text-to-speech model.",
		Snapshots:   []string{"gpt-4o-mini-tts-2025-03-20", "gpt-4o-mini-tts-2025-12-15"},
	},
	{
		ID:          TTS1,
		Name:        "TTS-1",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud},
		Description: "Legacy text-to-speech model.",
	},
	{
		ID:          TTS1HD,
		Name:        "TTS-1 HD",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud},
		Description: "Higher quality legacy text-to-speech model.",
	},
	{
		ID:          GPT4oTranscribe,
		Name:        "GPT-4o Transcribe",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud},
		Description: "Speech-to-text transcription model.",
	},
	{
		ID:          GPT4oMiniTranscribe,
		Name:        "GPT-4o mini Transcribe",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud},
		Description: "Cost-efficient transcription model.",
		Snapshots:   []string{"gpt-4o-mini-transcribe-2025-03-20", "gpt-4o-mini-transcribe-2025-12-15"},
	},
	{
		ID:          GPT4oTranscribeDiarize,
		Name:        "GPT-4o Transcribe Diarize",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud},
		Description: "Transcription model with diarization support.",
	},
	{
		ID:          Whisper1,
		Name:        "Whisper",
		Flavour:     "audio",
		Tags:        []Tag{TagCloud},
		Description: "Legacy Whisper transcription model.",
	},

	// Image generation & editing.
	{
		ID:          GPTImage15,
		Name:        "GPT Image 1.5",
		Flavour:     "image",
		Tags:        []Tag{TagCloud, TagVision},
		Description: "Image generation and editing model.",
	},
	{
		ID:          GPTImage1,
		Name:        "GPT Image 1",
		Flavour:     "image",
		Tags:        []Tag{TagCloud, TagVision},
		Description: "Image generation and editing model.",
	},
	{
		ID:          GPTImage1Mini,
		Name:        "gpt-image-1-mini",
		Flavour:     "image",
		Tags:        []Tag{TagCloud, TagVision},
		Description: "Smaller image generation model.",
	},
	{
		ID:          ChatGPTImageLatest,
		Name:        "ChatGPT Image Latest",
		Flavour:     "image",
		Tags:        []Tag{TagCloud, TagVision},
		Description: "Alias pointing to the image snapshot currently used in ChatGPT.",
	},

	// Video generation.
	{
		ID:          Sora2,
		Name:        "Sora 2",
		Flavour:     "video",
		Tags:        []Tag{TagCloud, TagVision},
		Description: "Video generation model.",
	},
	{
		ID:          Sora2Pro,
		Name:        "Sora 2 Pro",
		Flavour:     "video",
		Tags:        []Tag{TagCloud, TagVision},
		Description: "Higher-tier video generation model.",
	},

	// Embeddings.
	{
		ID:          TextEmbedding3Large,
		Name:        "text-embedding-3-large",
		Flavour:     "embedding",
		Tags:        []Tag{TagCloud, TagEmbedding},
		Description: "High-quality text embedding model.",
	},
	{
		ID:          TextEmbedding3Small,
		Name:        "text-embedding-3-small",
		Flavour:     "embedding",
		Tags:        []Tag{TagCloud, TagEmbedding},
		Description: "Cost-efficient text embedding model.",
	},

	// Moderation.
	{
		ID:          OmniModerationLatest,
		Name:        "omni-moderation-latest",
		Flavour:     "moderation",
		Tags:        []Tag{TagCloud},
		Description: "Moderation model for text/image safety classification.",
	},

	// Deprecated image models (kept for compatibility).
	{
		ID:          DallE2,
		Name:        "DALL·E 2 (deprecated)",
		Flavour:     "image",
		Tags:        []Tag{TagCloud, TagVision},
		Description: "Deprecated image generation model.",
		Deprecated:  true,
	},
	{
		ID:          DallE3,
		Name:        "DALL·E 3 (deprecated)",
		Flavour:     "image",
		Tags:        []Tag{TagCloud, TagVision},
		Description: "Deprecated image generation model.",
		Deprecated:  true,
	},
}
