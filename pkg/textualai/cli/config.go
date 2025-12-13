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

package cli

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"
)

// ProviderKind is the resolved provider used by the CLI.
type ProviderKind int

const (
	ProviderAuto ProviderKind = iota
	ProviderOpenAI
	ProviderClaude
	ProviderGemini
	ProviderMistral
	ProviderOllama
)

func (p ProviderKind) String() string {
	switch p {
	case ProviderOpenAI:
		return "openai"
	case ProviderClaude:
		return "claude"
	case ProviderGemini:
		return "gemini"
	case ProviderMistral:
		return "mistral"
	case ProviderOllama:
		return "ollama"
	default:
		return "auto"
	}
}

// Config contains user-facing options.
//
// This CLI intentionally avoids preserving backward compatibility with older
// flag names: it focuses on provider interoperability, not historic flags.
type Config struct {
	Help    OptBool
	Version OptBool
	Verbose OptBool

	Provider string
	Model    string

	// -----------------
	// Interoperable input
	// -----------------

	// Exactly one of these is used in one-shot mode:
	//   - Message
	//   - Object / ObjectFile
	Message    string
	Object     string
	ObjectFile string

	// Template is optional. When set, it renders the prompt from the input.
	Template     string
	TemplateFile string

	// TemplateSchema validates the input JSON value (before templating).
	TemplateSchema string

	// -----------------
	// Interoperable output
	// -----------------

	// OutputSchema requests schema-based structured output when the provider supports it
	// and validates the final output against the schema locally.
	OutputSchema string

	// -----------------
	// Runtime controls
	// -----------------

	Loop          OptBool
	ExitCommands  string
	AggregateType string
	Role          string
	Instructions  string
	Timeout       OptDuration

	Temperature OptFloat64
	TopP        OptFloat64
	MaxTokens   OptInt

	// -----------------
	// Provider-specific overrides (kept minimal)
	// -----------------

	ClaudeBaseURL    string
	ClaudeAPIVersion string
	ClaudeStream     OptBool

	GeminiBaseURL    string
	GeminiAPIVersion string
	GeminiStream     OptBool

	MistralBaseURL string
	MistralStream  OptBool

	OllamaHost     string
	OllamaEndpoint string
	OllamaStream   OptBool
}

// OptBool is a flag.Value that tracks whether it has been explicitly set.
// It also supports bool flag shorthand `--flag` (implicit true) by implementing
// IsBoolFlag.
type OptBool struct {
	set bool
	val bool
}

func (b *OptBool) IsBoolFlag() bool { return true }

func (b *OptBool) Set(s string) error {
	b.set = true
	// `--flag` may call Set("").
	if strings.TrimSpace(s) == "" {
		b.val = true
		return nil
	}
	v, err := strconv.ParseBool(s)
	if err != nil {
		return fmt.Errorf("parse bool %q: %w", s, err)
	}
	b.val = v
	return nil
}

func (b *OptBool) String() string {
	if !b.set {
		return ""
	}
	return strconv.FormatBool(b.val)
}

// IsSet reports whether the flag has been explicitly set by the user.
func (b OptBool) IsSet() bool { return b.set }

// Value returns the parsed flag value.
func (b OptBool) Value() bool { return b.val }

// Enabled is a convenience shortcut for IsSet() && Value().
func (b OptBool) Enabled() bool { return b.set && b.val }

type OptInt struct {
	set bool
	val int
}

func (i *OptInt) Set(s string) error {
	i.set = true
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return fmt.Errorf("parse int %q: %w", s, err)
	}
	i.val = v
	return nil
}

func (i *OptInt) String() string {
	if !i.set {
		return ""
	}
	return strconv.Itoa(i.val)
}

func (i OptInt) IsSet() bool { return i.set }

func (i OptInt) Value() int { return i.val }

type OptFloat64 struct {
	set bool
	val float64
}

func (f *OptFloat64) Set(s string) error {
	f.set = true
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return fmt.Errorf("parse float %q: %w", s, err)
	}
	f.val = v
	return nil
}

func (f *OptFloat64) String() string {
	if !f.set {
		return ""
	}
	return strconv.FormatFloat(f.val, 'f', -1, 64)
}

func (f OptFloat64) IsSet() bool { return f.set }

func (f OptFloat64) Value() float64 { return f.val }

type OptDuration struct {
	set bool
	val time.Duration
}

func (d *OptDuration) Set(s string) error {
	d.set = true
	v, err := time.ParseDuration(strings.TrimSpace(s))
	if err != nil {
		return fmt.Errorf("parse duration %q: %w", s, err)
	}
	d.val = v
	return nil
}

func (d *OptDuration) String() string {
	if !d.set {
		return ""
	}
	return d.val.String()
}

func (d OptDuration) IsSet() bool { return d.set }

func (d OptDuration) Value() time.Duration { return d.val }

func parseCLI(args []string, getenv func(string) string) (Config, error) {
	if getenv == nil {
		getenv = os.Getenv
	}

	cfg := Config{
		Provider:      "auto",
		AggregateType: "word",
		Role:          "user",
		ExitCommands:  "exit,quit,/exit,/quit",
	}
	if v := strings.TrimSpace(getenv("TEXTUALAI_PROVIDER")); v != "" {
		cfg.Provider = v
	}

	fs := flag.NewFlagSet("textualai", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	// Meta.
	fs.Var(&cfg.Help, "help", "Show help.")
	fs.Var(&cfg.Help, "h", "Show help (shorthand).")
	fs.Var(&cfg.Version, "version", "Print version and exit.")
	fs.Var(&cfg.Verbose, "verbose", "Enable diagnostic output to stderr.")

	// Provider selection.
	fs.StringVar(&cfg.Provider, "provider", cfg.Provider, "Provider: auto|openai|claude|gemini|mistral|ollama. Can also be set via TEXTUALAI_PROVIDER.")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "Model name. Prefix with openai:, claude:, gemini:, mistral:, or ollama: to force provider.")

	// Interoperable input.
	fs.StringVar(&cfg.Message, "message", cfg.Message, "Raw message string (one-shot).")
	fs.StringVar(&cfg.Object, "object", cfg.Object, "JSON value as a string (one-shot).")
	fs.StringVar(&cfg.ObjectFile, "object-file", cfg.ObjectFile, "Path to a file containing a JSON value. Use '-' to read JSON from stdin (one-shot).")
	fs.StringVar(&cfg.Template, "template", cfg.Template, "Go text/template string. The input JSON value is available as {{.}}.")
	fs.StringVar(&cfg.TemplateFile, "template-file", cfg.TemplateFile, "Path to a Go text/template file. The input JSON value is available as {{.}}.")
	fs.StringVar(&cfg.TemplateSchema, "template-schema", cfg.TemplateSchema, "Path to a JSON Schema file used to validate the input JSON before template execution.")

	// Interoperable output.
	fs.StringVar(&cfg.OutputSchema, "output-schema", cfg.OutputSchema, "Path to a JSON Schema file for requesting/validating structured output.")

	// Runtime controls.
	fs.Var(&cfg.Loop, "loop", "Loop mode: after a response completes, prompt for a new message on stdin.")
	fs.StringVar(&cfg.ExitCommands, "exit-commands", cfg.ExitCommands, "Comma-separated commands that exit in --loop.")
	fs.StringVar(&cfg.AggregateType, "aggregate", cfg.AggregateType, "Streaming aggregation: word|line.")
	fs.StringVar(&cfg.Role, "role", cfg.Role, "Role for the user message when the provider builds a default message (default: user).")
	fs.StringVar(&cfg.Instructions, "instructions", cfg.Instructions, "System/developer instructions (provider-specific mapping).")
	fs.Var(&cfg.Timeout, "timeout", "Per-request timeout (e.g. 30s, 2m).")

	// Common model controls.
	fs.Var(&cfg.Temperature, "temperature", "Sampling temperature (provider-specific range).")
	fs.Var(&cfg.TopP, "top-p", "Nucleus sampling probability mass (0..1).")
	fs.Var(&cfg.MaxTokens, "max-tokens", "Max output tokens (OpenAI: max_output_tokens, Claude: max_tokens, Gemini: generationConfig.maxOutputTokens, Mistral: max_tokens, Ollama: num_predict).")

	// Provider-specific overrides (minimal).
	fs.StringVar(&cfg.ClaudeBaseURL, "claude-base-url", cfg.ClaudeBaseURL, "Claude/Anthropic: base URL (overrides ANTHROPIC_BASE_URL).")
	fs.StringVar(&cfg.ClaudeAPIVersion, "claude-api-version", cfg.ClaudeAPIVersion, "Claude/Anthropic: anthropic-version header (overrides ANTHROPIC_VERSION).")
	fs.Var(&cfg.ClaudeStream, "claude-stream", "Claude/Anthropic: stream mode (true/false). Default is true.")

	fs.StringVar(&cfg.GeminiBaseURL, "gemini-base-url", cfg.GeminiBaseURL, "Gemini: base URL (default: https://generativelanguage.googleapis.com).")
	fs.StringVar(&cfg.GeminiAPIVersion, "gemini-api-version", cfg.GeminiAPIVersion, "Gemini: REST API version (default: v1beta).")
	fs.Var(&cfg.GeminiStream, "gemini-stream", "Gemini: stream mode (true/false). Default is true.")

	fs.StringVar(&cfg.MistralBaseURL, "mistral-base-url", cfg.MistralBaseURL, "Mistral: base URL (overrides MISTRAL_BASE_URL).")
	fs.Var(&cfg.MistralStream, "mistral-stream", "Mistral: stream mode (true/false). Default is true.")

	fs.StringVar(&cfg.OllamaHost, "ollama-host", cfg.OllamaHost, "Ollama: base URL (overrides OLLAMA_HOST). Example: http://localhost:11434")
	fs.StringVar(&cfg.OllamaEndpoint, "ollama-endpoint", cfg.OllamaEndpoint, "Ollama: endpoint chat|generate (default: chat).")
	fs.Var(&cfg.OllamaStream, "ollama-stream", "Ollama: stream mode (true/false). Default is true.")

	if err := fs.Parse(args); err != nil {
		return cfg, err
	}

	// Sanitize.
	cfg.Provider = strings.TrimSpace(cfg.Provider)
	cfg.Model = strings.TrimSpace(cfg.Model)

	cfg.Message = strings.TrimSpace(cfg.Message)
	cfg.Object = strings.TrimSpace(cfg.Object)
	cfg.ObjectFile = strings.TrimSpace(cfg.ObjectFile)
	cfg.Template = strings.TrimSpace(cfg.Template)
	cfg.TemplateFile = strings.TrimSpace(cfg.TemplateFile)
	cfg.TemplateSchema = strings.TrimSpace(cfg.TemplateSchema)
	cfg.OutputSchema = strings.TrimSpace(cfg.OutputSchema)

	cfg.ExitCommands = strings.TrimSpace(cfg.ExitCommands)
	cfg.AggregateType = strings.TrimSpace(cfg.AggregateType)
	cfg.Role = strings.TrimSpace(cfg.Role)
	cfg.Instructions = strings.TrimSpace(cfg.Instructions)

	cfg.ClaudeBaseURL = strings.TrimSpace(cfg.ClaudeBaseURL)
	cfg.ClaudeAPIVersion = strings.TrimSpace(cfg.ClaudeAPIVersion)

	cfg.GeminiBaseURL = strings.TrimSpace(cfg.GeminiBaseURL)
	cfg.GeminiAPIVersion = strings.TrimSpace(cfg.GeminiAPIVersion)

	cfg.MistralBaseURL = strings.TrimSpace(cfg.MistralBaseURL)

	cfg.OllamaHost = strings.TrimSpace(cfg.OllamaHost)
	cfg.OllamaEndpoint = strings.TrimSpace(cfg.OllamaEndpoint)

	// Basic validation for mutually exclusive flags.
	if cfg.Template != "" && cfg.TemplateFile != "" {
		return cfg, fmt.Errorf("use either --template or --template-file (not both)")
	}
	if cfg.Object != "" && cfg.ObjectFile != "" {
		return cfg, fmt.Errorf("use either --object or --object-file (not both)")
	}

	return cfg, nil
}

func printUsage(w io.Writer) {
	fmt.Fprintln(w, "textualai - streaming CLI for OpenAI, Claude, Gemini, Mistral, and Ollama")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  textualai help")
	fmt.Fprintln(w, "  textualai --model <provider:model> [--message <text> | --object <json> | --object-file <path>] [flags]")
	fmt.Fprintln(w, "  textualai --model <provider:model> --loop [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Core:")
	fmt.Fprintln(w, "  --model <name>                 Model name. Prefix with openai:, claude:, gemini:, mistral:, or ollama: to force provider.")
	fmt.Fprintln(w, "  --provider <auto|openai|claude|gemini|mistral|ollama> Provider selection (default: auto).")
	fmt.Fprintln(w, "  --message <text>               Raw message string (one-shot).")
	fmt.Fprintln(w, "  --object <json>                JSON value as a string (one-shot).")
	fmt.Fprintln(w, "  --object-file <path|->         JSON value from file, or stdin when path is '-'.")
	fmt.Fprintln(w, "  --template <tmpl>              Go text/template string; input value is {{.}}.")
	fmt.Fprintln(w, "  --template-file <path>         Go text/template file; input value is {{.}}.")
	fmt.Fprintln(w, "  --template-schema <path>       JSON Schema to validate input before templating.")
	fmt.Fprintln(w, "  --output-schema <path>         JSON Schema to request & validate structured output.")
	fmt.Fprintln(w, "  --instructions <text>          System/developer instructions.")
	fmt.Fprintln(w, "  --aggregate <word|line>        Streaming aggregation (default: word).")
	fmt.Fprintln(w, "  --temperature <float>          Sampling temperature.")
	fmt.Fprintln(w, "  --top-p <float>                Nucleus sampling parameter.")
	fmt.Fprintln(w, "  --max-tokens <int>             Max output tokens.")
	fmt.Fprintln(w, "  --timeout <duration>           Per-request timeout (e.g. 30s, 2m).")
	fmt.Fprintln(w, "  --loop                         Interactive loop mode.")
	fmt.Fprintln(w, "  --exit-commands <csv>          Exit commands for loop mode.")
	fmt.Fprintln(w, "  --verbose                      Print diagnostics to stderr.")
	fmt.Fprintln(w, "  --version                      Print version.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Examples:")
	fmt.Fprintln(w, "  # Raw message")
	fmt.Fprintln(w, "  OPENAI_API_KEY=... textualai --model openai:gpt-4.1 --message \"Hello\"")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  # JSON + template")
	fmt.Fprintln(w, "  textualai --model mistral:mistral-small-latest --object '{\"name\":\"Ada\"}' --template 'Hello {{.name}}'")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  # Structured output")
	fmt.Fprintln(w, "  textualai --model openai:gpt-4.1 --output-schema ./schema.json --message \"Return JSON only.\"")
}

// ResolveProvider selects the provider and normalizes the model name.
// Prefix-based selection always wins.
func ResolveProvider(providerFlag string, model string) (ProviderKind, string, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return ProviderAuto, "", fmt.Errorf("model must not be empty")
	}

	lower := strings.ToLower(model)
	switch {
	case strings.HasPrefix(lower, "openai:"):
		return ProviderOpenAI, strings.TrimSpace(model[len("openai:"):]), nil
	case strings.HasPrefix(lower, "oa:"):
		return ProviderOpenAI, strings.TrimSpace(model[len("oa:"):]), nil
	case strings.HasPrefix(lower, "claude:"):
		return ProviderClaude, strings.TrimSpace(model[len("claude:"):]), nil
	case strings.HasPrefix(lower, "cl:"):
		return ProviderClaude, strings.TrimSpace(model[len("cl:"):]), nil
	case strings.HasPrefix(lower, "anthropic:"):
		return ProviderClaude, strings.TrimSpace(model[len("anthropic:"):]), nil
	case strings.HasPrefix(lower, "an:"):
		return ProviderClaude, strings.TrimSpace(model[len("an:"):]), nil
	case strings.HasPrefix(lower, "gemini:"):
		return ProviderGemini, strings.TrimSpace(model[len("gemini:"):]), nil
	case strings.HasPrefix(lower, "ge:"):
		return ProviderGemini, strings.TrimSpace(model[len("ge:"):]), nil
	case strings.HasPrefix(lower, "mistral:"):
		return ProviderMistral, strings.TrimSpace(model[len("mistral:"):]), nil
	case strings.HasPrefix(lower, "mi:"):
		return ProviderMistral, strings.TrimSpace(model[len("mi:"):]), nil
	case strings.HasPrefix(lower, "ollama:"):
		return ProviderOllama, strings.TrimSpace(model[len("ollama:"):]), nil
	}

	switch strings.ToLower(strings.TrimSpace(providerFlag)) {
	case "", "auto":
		// fallthrough to heuristics below
	case "openai":
		return ProviderOpenAI, model, nil
	case "claude", "anthropic":
		return ProviderClaude, model, nil
	case "gemini":
		return ProviderGemini, model, nil
	case "mistral":
		return ProviderMistral, model, nil
	case "ollama":
		return ProviderOllama, model, nil
	default:
		return ProviderAuto, "", fmt.Errorf("unknown provider %q (expected auto|openai|claude|gemini|mistral|ollama)", providerFlag)
	}

	// Simple heuristics for auto mode.
	if strings.HasPrefix(lower, "gpt") || strings.HasPrefix(lower, "o") {
		return ProviderOpenAI, model, nil
	}
	if strings.HasPrefix(lower, "claude") {
		return ProviderClaude, model, nil
	}
	if strings.HasPrefix(lower, "gemini") {
		return ProviderGemini, model, nil
	}
	if strings.HasPrefix(lower, "mistral") ||
		strings.HasPrefix(lower, "codestral") ||
		strings.HasPrefix(lower, "ministral") ||
		strings.HasPrefix(lower, "devstral") ||
		strings.HasPrefix(lower, "magistral") {
		return ProviderMistral, model, nil
	}
	return ProviderOllama, model, nil
}
