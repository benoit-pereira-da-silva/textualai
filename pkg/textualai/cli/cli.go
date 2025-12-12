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

// Package main implements the `textualai` command-line interface.
//
// `textualai` is a small streaming chat CLI that can target either:
//   - OpenAI (Responses API), or
//   - Ollama (local HTTP API: /api/chat or /api/generate)
//
// It is built on top of the textual pipeline concept used throughout this repo.
// Internally it creates a provider-specific ResponseProcessor[textual.String],
// feeds it one input message, then streams the processor outputs to stdout.
//
// # Provider selection
//
// The provider can be selected explicitly with --provider, or inferred from the
// --model value.
//
// Recommended and deterministic form:
//
//	textualai --model openai:gpt-4.1 ...
//	textualai --model ollama:llama3.1 ...
//
// When no prefix is used and --provider is "auto" (default), the provider is
// inferred using simple heuristics:
//
//   - model names starting with "gpt" or "o" => OpenAI
//   - everything else => Ollama
//
// # Environment variables
//
// OpenAI:
//   - OPENAI_API_KEY (required when using the OpenAI provider)
//
// Ollama:
//   - OLLAMA_HOST (optional; default: http://localhost:11434)
//
// textuali itself:
//   - TEXTUALAI_PROVIDER (optional; default provider when --provider is omitted)
//     values: auto|openai|ollama
//
// Usage
//
//	textualai help
//	textualai --model openai:gpt-4.1 --message "Hello!"
//	textualai --model ollama:llama3.1 --loop
//
// # Prompt templates
//
// The processors accept a Go text/template string with an {{.Input}} placeholder.
// If --prompt-template is omitted, the default is a minimal identity template:
//
//	{{.Input}}
//
// The template is executed for every input message.
//
// # Streaming output semantics
//
// The underlying ResponseProcessors emit aggregated text snapshots (prefixes).
// The CLI turns those snapshots back into a stream by printing only the delta
// between successive snapshots.
//
// Exit status
//
//	0 - success
//	2 - CLI usage / argument error
//	1 - runtime failure (HTTP error, missing API key, etc.)
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualollama"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualopenai"
)

// version is set at build time using:
//
//	go build -ldflags "-X main.version=v1.2.3"
//
// When not set, it defaults to "dev".
var version = "dev"

const defaultPromptTemplate = "{{.Input}}"

type providerKind int

const (
	providerAuto providerKind = iota
	providerOpenAI
	providerOllama
)

// cliConfig contains every user-facing option. Many fields are "optional"
// wrappers so we can preserve tri-state behaviour (unset vs explicitly set).
type cliConfig struct {
	Help    optBool
	Version optBool
	Verbose optBool

	Provider string
	Model    string

	PromptTemplatePath string

	Message         string
	FileMessagePath string
	FileEncoding    string

	Loop          optBool
	ExitCommands  string
	AggregateType string
	Role          string
	Instructions  string

	Timeout optDuration

	Temperature optFloat64
	TopP        optFloat64
	MaxTokens   optInt

	// Structured Outputs / JSON Schema.
	JSONSchemaPath   string
	JSONSchemaName   string
	JSONSchemaStrict optBool

	// -----------------
	// OpenAI-only flags
	// -----------------

	OpenAIServiceTier          string
	OpenAITruncation           string
	OpenAIStore                optBool
	OpenAIPromptCacheKey       string
	OpenAIPromptCacheRetention string
	OpenAISafetyIdentifier     string
	OpenAIMetadata             kvStringMap
	OpenAIInclude              string

	// -----------------
	// Ollama-only flags
	// -----------------

	OllamaHost      string
	OllamaEndpoint  string
	OllamaKeepAlive string
	OllamaThink     optBool
	OllamaStream    optBool
	OllamaFormat    string

	OllamaTopK     optInt
	OllamaNumCtx   optInt
	OllamaSeed     optInt
	OllamaStop     string
	OllamaRaw      optBool
	OllamaExtraOpt kvAnyMap
}

// optBool is a flag.Value that tracks whether it has been explicitly set.
// It also supports bool flag shorthand `--flag` (implicit true) by implementing
// IsBoolFlag.
type optBool struct {
	set bool
	val bool
}

func (b *optBool) IsBoolFlag() bool { return true }

func (b *optBool) Set(s string) error {
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

func (b *optBool) String() string {
	if !b.set {
		return ""
	}
	return strconv.FormatBool(b.val)
}

func (b optBool) Enabled() bool { return b.set && b.val }

type optInt struct {
	set bool
	val int
}

func (i *optInt) Set(s string) error {
	i.set = true
	v, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return fmt.Errorf("parse int %q: %w", s, err)
	}
	i.val = v
	return nil
}

func (i *optInt) String() string {
	if !i.set {
		return ""
	}
	return strconv.Itoa(i.val)
}

type optFloat64 struct {
	set bool
	val float64
}

func (f *optFloat64) Set(s string) error {
	f.set = true
	v, err := strconv.ParseFloat(strings.TrimSpace(s), 64)
	if err != nil {
		return fmt.Errorf("parse float %q: %w", s, err)
	}
	f.val = v
	return nil
}

func (f *optFloat64) String() string {
	if !f.set {
		return ""
	}
	// Keep it stable and readable.
	return strconv.FormatFloat(f.val, 'f', -1, 64)
}

type optDuration struct {
	set bool
	val time.Duration
}

func (d *optDuration) Set(s string) error {
	d.set = true
	v, err := time.ParseDuration(strings.TrimSpace(s))
	if err != nil {
		return fmt.Errorf("parse duration %q: %w", s, err)
	}
	d.val = v
	return nil
}

func (d *optDuration) String() string {
	if !d.set {
		return ""
	}
	return d.val.String()
}

// kvStringMap collects repeated key=value pairs into a map[string]string.
type kvStringMap map[string]string

func (m *kvStringMap) Set(s string) error {
	if *m == nil {
		*m = make(map[string]string)
	}
	key, val, ok := strings.Cut(s, "=")
	if !ok {
		return fmt.Errorf("expected key=value, got %q", s)
	}
	key = strings.TrimSpace(key)
	if key == "" {
		return fmt.Errorf("empty key in %q", s)
	}
	(*m)[key] = strings.TrimSpace(val)
	return nil
}

func (m *kvStringMap) String() string {
	if m == nil || *m == nil {
		return ""
	}
	// Non-deterministic ordering is fine for help output.
	var parts []string
	for k, v := range *m {
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, ",")
}

// kvAnyMap collects repeated key=value pairs into a map[string]any.
// Values are parsed as:
//   - JSON (objects/arrays) when value starts with "{" or "["
//   - bool, int, float when possible
//   - string otherwise
type kvAnyMap map[string]any

func (m *kvAnyMap) Set(s string) error {
	if *m == nil {
		*m = make(map[string]any)
	}
	key, raw, ok := strings.Cut(s, "=")
	if !ok {
		return fmt.Errorf("expected key=value, got %q", s)
	}
	key = strings.TrimSpace(key)
	if key == "" {
		return fmt.Errorf("empty key in %q", s)
	}
	raw = strings.TrimSpace(raw)
	(*m)[key] = parseScalarOrJSON(raw)
	return nil
}

func (m *kvAnyMap) String() string {
	if m == nil || *m == nil {
		return ""
	}
	var parts []string
	for k, v := range *m {
		parts = append(parts, fmt.Sprintf("%s=%v", k, v))
	}
	return strings.Join(parts, ",")
}

func parseScalarOrJSON(s string) any {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}

	// JSON object/array.
	if strings.HasPrefix(s, "{") || strings.HasPrefix(s, "[") {
		var v any
		if err := json.Unmarshal([]byte(s), &v); err == nil {
			return v
		}
		// Fall back to string if malformed.
		return s
	}

	// bool
	if b, err := strconv.ParseBool(s); err == nil {
		return b
	}

	// int
	if i, err := strconv.ParseInt(s, 10, 64); err == nil {
		return int(i)
	}

	// float
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f
	}

	return s
}

func main() {
	os.Exit(run(os.Args, os.Stdout, os.Stderr))
}

func run(argv []string, stdout io.Writer, stderr io.Writer) int {
	// "textualai help" convenience command.
	if len(argv) >= 2 {
		switch strings.TrimSpace(argv[1]) {
		case "help", "usage", "--help", "-help", "-h", "--h":
			printUsage(stdout)
			return 0
		}
	}

	cfg, err := parseCLI(argv[1:])
	if err != nil {
		// parseCLI already returns user-friendly errors; show usage too.
		fmt.Fprintln(stderr, "Error:", err)
		fmt.Fprintln(stderr)
		printUsage(stderr)
		return 2
	}

	if cfg.Help.Enabled() {
		printUsage(stdout)
		return 0
	}

	if cfg.Version.Enabled() {
		fmt.Fprintln(stdout, version)
		return 0
	}

	// Validate required fields.
	if strings.TrimSpace(cfg.Model) == "" {
		fmt.Fprintln(stderr, "Error: --model is required")
		fmt.Fprintln(stderr)
		printUsage(stderr)
		return 2
	}

	// Load prompt template (defaults to the built-in identity template).
	templateStr, err := loadPromptTemplate(cfg.PromptTemplatePath)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 2
	}

	// Load message file if provided. We do it once so --loop can still reuse it
	// as a first message without re-reading the file.
	fileMsg := ""
	if strings.TrimSpace(cfg.FileMessagePath) != "" {
		msg, err := readMessageFile(cfg.FileMessagePath, cfg.FileEncoding)
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			return 2
		}
		fileMsg = msg
	}

	// Root context reacts to Ctrl+C (SIGINT) and SIGTERM.
	rootCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Prepare optional JSON schema.
	var jsonSchema map[string]any
	if strings.TrimSpace(cfg.JSONSchemaPath) != "" {
		schema, err := loadJSONSchema(cfg.JSONSchemaPath)
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			return 2
		}
		jsonSchema = schema
	}

	// Resolve provider and normalize model (strip openai:/ollama: prefix).
	prov, modelName, err := resolveProvider(cfg.Provider, cfg.Model)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 2
	}

	if cfg.Verbose.Enabled() {
		fmt.Fprintf(stderr, "Provider: %s\n", providerName(prov))
		fmt.Fprintf(stderr, "Model: %s\n", modelName)
		if cfg.PromptTemplatePath == "" {
			fmt.Fprintf(stderr, "Prompt template: (default)\n")
		} else {
			fmt.Fprintf(stderr, "Prompt template: %s\n", cfg.PromptTemplatePath)
		}
		fmt.Fprintln(stderr)
	}

	// Prepare a buffered writer for smooth streaming.
	outw := bufio.NewWriter(stdout)
	defer outw.Flush()

	// Pick initial message (non-loop).
	initialMsg := combineMessage(cfg.Message, fileMsg)

	// If no message was provided, allow interactive input in loop mode.
	if !cfg.Loop.Enabled() && strings.TrimSpace(initialMsg) == "" {
		fmt.Fprintln(stderr, "Error: provide --message or --file-message, or use --loop")
		fmt.Fprintln(stderr)
		printUsage(stderr)
		return 2
	}

	// Build the provider processor and run once or in loop.
	switch prov {
	case providerOpenAI:
		return runOpenAI(rootCtx, cfg, modelName, templateStr, jsonSchema, initialMsg, outw, stderr)
	case providerOllama:
		return runOllama(rootCtx, cfg, modelName, templateStr, jsonSchema, initialMsg, outw, stderr)
	default:
		fmt.Fprintln(stderr, "Error: unable to resolve provider")
		return 2
	}
}

func parseCLI(args []string) (cliConfig, error) {
	cfg := cliConfig{
		Provider: "auto",
		// Sensible defaults for file reading.
		FileEncoding:   "UTF-8",
		AggregateType:  "word",
		Role:           "user",
		ExitCommands:   "exit,quit,/exit,/quit",
		JSONSchemaName: "response",
	}
	// Environment defaults (kept minimal and explicit).
	if v := strings.TrimSpace(os.Getenv("TEXTUALAI_PROVIDER")); v != "" {
		cfg.Provider = v
	}

	fs := flag.NewFlagSet("textualai", flag.ContinueOnError)
	// We handle error rendering ourselves.
	fs.SetOutput(io.Discard)

	// Standard meta flags.
	fs.Var(&cfg.Help, "help", "Show help.")
	fs.Var(&cfg.Help, "h", "Show help (shorthand).")
	fs.Var(&cfg.Version, "version", "Print version and exit.")
	fs.Var(&cfg.Verbose, "verbose", "Enable diagnostic output to stderr.")

	// Core selection flags.
	fs.StringVar(&cfg.Provider, "provider", cfg.Provider, "Provider: auto|openai|ollama. Can also be set via TEXTUALAI_PROVIDER.")
	fs.StringVar(&cfg.Model, "model", cfg.Model, "Model name. Prefix with openai: or ollama: to force provider (e.g. openai:gpt-4.1, ollama:llama3.1).")

	// Input / prompt shaping.
	fs.StringVar(&cfg.PromptTemplatePath, "prompt-template", cfg.PromptTemplatePath, "Path to a Go text/template file containing {{.Input}}. Default: identity template.")
	fs.StringVar(&cfg.Message, "message", cfg.Message, "Message to send (one-shot).")
	fs.StringVar(&cfg.FileMessagePath, "file-message", cfg.FileMessagePath, "Path to a file containing the message. Use '-' to read from stdin.")
	fs.StringVar(&cfg.FileEncoding, "file-encoding", cfg.FileEncoding, "Encoding used for --file-message (default: UTF-8). See textual.ParseEncoding supported names.")
	fs.Var(&cfg.Loop, "loop", "Loop mode: after a response completes, prompt for a new message on stdin.")
	fs.StringVar(&cfg.ExitCommands, "exit-commands", cfg.ExitCommands, "Comma-separated commands that exit in --loop (default: exit,quit,/exit,/quit).")
	fs.StringVar(&cfg.AggregateType, "aggregate", cfg.AggregateType, "Streaming aggregation: word|line.")
	fs.StringVar(&cfg.Role, "role", cfg.Role, "Role for the user message when the provider builds a default message (default: user).")
	fs.StringVar(&cfg.Instructions, "instructions", cfg.Instructions, "System/developer instructions (OpenAI 'instructions', Ollama 'system').")
	fs.Var(&cfg.Timeout, "timeout", "Per-request timeout (e.g. 30s, 2m).")

	// Common model controls.
	fs.Var(&cfg.Temperature, "temperature", "Sampling temperature (provider-specific range, typical: 0.0..2.0).")
	fs.Var(&cfg.TopP, "top-p", "Nucleus sampling probability mass (0..1).")
	fs.Var(&cfg.MaxTokens, "max-tokens", "Max output tokens (OpenAI: max_output_tokens, Ollama: num_predict).")

	// Structured outputs.
	fs.StringVar(&cfg.JSONSchemaPath, "json-schema", cfg.JSONSchemaPath, "Path to a JSON Schema file for Structured Outputs. Applied to OpenAI and Ollama when supported.")
	fs.StringVar(&cfg.JSONSchemaName, "json-schema-name", cfg.JSONSchemaName, "Name for the OpenAI JSON schema wrapper (default: response).")
	// Default strict=true if the flag is provided without explicit value.
	cfg.JSONSchemaStrict.val = true
	fs.Var(&cfg.JSONSchemaStrict, "json-schema-strict", "OpenAI-only: strict schema enforcement (default true when --json-schema is set).")

	// OpenAI-only.
	fs.StringVar(&cfg.OpenAIServiceTier, "openai-service-tier", cfg.OpenAIServiceTier, "OpenAI: service tier (auto|default|flex|priority).")
	fs.StringVar(&cfg.OpenAITruncation, "openai-truncation", cfg.OpenAITruncation, "OpenAI: truncation strategy (auto|disabled).")
	fs.Var(&cfg.OpenAIStore, "openai-store", "OpenAI: store the response (boolean).")
	fs.StringVar(&cfg.OpenAIPromptCacheKey, "openai-prompt-cache-key", cfg.OpenAIPromptCacheKey, "OpenAI: prompt_cache_key.")
	fs.StringVar(&cfg.OpenAIPromptCacheRetention, "openai-prompt-cache-retention", cfg.OpenAIPromptCacheRetention, "OpenAI: prompt_cache_retention (e.g. 24h).")
	fs.StringVar(&cfg.OpenAISafetyIdentifier, "openai-safety-identifier", cfg.OpenAISafetyIdentifier, "OpenAI: safety_identifier.")
	fs.Var(&cfg.OpenAIMetadata, "openai-metadata", "OpenAI: metadata key=value (repeatable).")
	fs.StringVar(&cfg.OpenAIInclude, "openai-include", cfg.OpenAIInclude, "OpenAI: include fields (comma-separated).")

	// Ollama-only.
	fs.StringVar(&cfg.OllamaHost, "ollama-host", cfg.OllamaHost, "Ollama: base URL (overrides OLLAMA_HOST). Example: http://localhost:11434")
	fs.StringVar(&cfg.OllamaEndpoint, "ollama-endpoint", cfg.OllamaEndpoint, "Ollama: endpoint chat|generate (default: chat).")
	fs.StringVar(&cfg.OllamaKeepAlive, "ollama-keep-alive", cfg.OllamaKeepAlive, "Ollama: keep_alive (e.g. 5m) or 0 to unload immediately.")
	fs.Var(&cfg.OllamaThink, "ollama-think", "Ollama: enable/disable thinking for thinking-capable models.")
	fs.Var(&cfg.OllamaStream, "ollama-stream", "Ollama: stream mode (true/false). Default is true.")
	fs.StringVar(&cfg.OllamaFormat, "ollama-format", cfg.OllamaFormat, "Ollama: format (e.g. json). For JSON schema, use --json-schema.")
	fs.Var(&cfg.OllamaTopK, "ollama-top-k", "Ollama: options.top_k.")
	fs.Var(&cfg.OllamaNumCtx, "ollama-num-ctx", "Ollama: options.num_ctx.")
	fs.Var(&cfg.OllamaSeed, "ollama-seed", "Ollama: options.seed.")
	fs.StringVar(&cfg.OllamaStop, "ollama-stop", cfg.OllamaStop, "Ollama: options.stop (comma-separated).")
	fs.Var(&cfg.OllamaRaw, "ollama-raw", "Ollama: /api/generate raw mode (disables template/system processing).")
	fs.Var(&cfg.OllamaExtraOpt, "ollama-option", "Ollama: arbitrary option key=value (repeatable). Values can be JSON, bool, int, float, or string.")

	if err := fs.Parse(args); err != nil {
		// flag returns errors like "flag provided but not defined: -x".
		return cfg, err
	}

	// Sanitize some defaults.
	cfg.Provider = strings.TrimSpace(cfg.Provider)
	cfg.Model = strings.TrimSpace(cfg.Model)
	cfg.PromptTemplatePath = strings.TrimSpace(cfg.PromptTemplatePath)
	cfg.Message = strings.TrimSpace(cfg.Message)
	cfg.FileMessagePath = strings.TrimSpace(cfg.FileMessagePath)
	cfg.FileEncoding = strings.TrimSpace(cfg.FileEncoding)
	cfg.AggregateType = strings.TrimSpace(cfg.AggregateType)
	cfg.Role = strings.TrimSpace(cfg.Role)
	cfg.Instructions = strings.TrimSpace(cfg.Instructions)
	cfg.ExitCommands = strings.TrimSpace(cfg.ExitCommands)

	cfg.OpenAIServiceTier = strings.TrimSpace(cfg.OpenAIServiceTier)
	cfg.OpenAITruncation = strings.TrimSpace(cfg.OpenAITruncation)
	cfg.OpenAIPromptCacheKey = strings.TrimSpace(cfg.OpenAIPromptCacheKey)
	cfg.OpenAIPromptCacheRetention = strings.TrimSpace(cfg.OpenAIPromptCacheRetention)
	cfg.OpenAISafetyIdentifier = strings.TrimSpace(cfg.OpenAISafetyIdentifier)
	cfg.OpenAIInclude = strings.TrimSpace(cfg.OpenAIInclude)

	cfg.OllamaHost = strings.TrimSpace(cfg.OllamaHost)
	cfg.OllamaEndpoint = strings.TrimSpace(cfg.OllamaEndpoint)
	cfg.OllamaKeepAlive = strings.TrimSpace(cfg.OllamaKeepAlive)
	cfg.OllamaFormat = strings.TrimSpace(cfg.OllamaFormat)
	cfg.OllamaStop = strings.TrimSpace(cfg.OllamaStop)

	// If the user didn't pass --json-schema-strict but did pass --json-schema,
	// keep strict=true by default.
	if strings.TrimSpace(cfg.JSONSchemaPath) != "" && !cfg.JSONSchemaStrict.set {
		cfg.JSONSchemaStrict.set = true
		cfg.JSONSchemaStrict.val = true
	}

	return cfg, nil
}

func printUsage(w io.Writer) {
	// Keep the help self-contained and copy/paste friendly.
	fmt.Fprintln(w, "textualai - streaming CLI chat for OpenAI or Ollama")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Usage:")
	fmt.Fprintln(w, "  textualai help")
	fmt.Fprintln(w, "  textualai --model <model> [--message <text> | --file-message <path>] [flags]")
	fmt.Fprintln(w, "  textualai --model <model> --loop [flags]")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Provider selection:")
	fmt.Fprintln(w, "  - Recommended: prefix the model with 'openai:' or 'ollama:'.")
	fmt.Fprintln(w, "    Examples: openai:gpt-4.1, ollama:llama3.1")
	fmt.Fprintln(w, "  - Alternatively set --provider openai|ollama.")
	fmt.Fprintln(w, "  - Or rely on --provider auto (default) which uses heuristics.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Environment:")
	fmt.Fprintln(w, "  OPENAI_API_KEY        OpenAI API key (required for OpenAI provider).")
	fmt.Fprintln(w, "  OLLAMA_HOST           Ollama host (optional, default http://localhost:11434).")
	fmt.Fprintln(w, "  TEXTUALAI_PROVIDER    Default provider (optional): auto|openai|ollama.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Core flags:")
	fmt.Fprintln(w, "  --model <name>                 Model name. Prefix with openai: or ollama: to force provider.")
	fmt.Fprintln(w, "  --provider <auto|openai|ollama> Provider selection (default: auto).")
	fmt.Fprintln(w, "  --prompt-template <path>       Go text/template file. Must contain {{.Input}}. Default: identity template.")
	fmt.Fprintln(w, "  --message <text>               Send a single message.")
	fmt.Fprintln(w, "  --file-message <path|->        Read message from file (or stdin when path is '-').")
	fmt.Fprintln(w, "  --file-encoding <name>         Encoding for --file-message (default UTF-8).")
	fmt.Fprintln(w, "  --loop                         Interactive loop (read messages from stdin).")
	fmt.Fprintln(w, "  --exit-commands <csv>          Exit commands in loop mode (default exit,quit,/exit,/quit).")
	fmt.Fprintln(w, "  --aggregate <word|line>        Streaming aggregation (default word).")
	fmt.Fprintln(w, "  --instructions <text>          System/developer instructions.")
	fmt.Fprintln(w, "  --temperature <float>          Sampling temperature.")
	fmt.Fprintln(w, "  --top-p <float>                Nucleus sampling parameter.")
	fmt.Fprintln(w, "  --max-tokens <int>             Max output tokens.")
	fmt.Fprintln(w, "  --timeout <duration>           Per-request timeout (e.g. 30s, 2m).")
	fmt.Fprintln(w, "  --json-schema <path>           JSON Schema file (Structured Outputs).")
	fmt.Fprintln(w, "  --json-schema-name <name>      OpenAI JSON schema wrapper name (default: response).")
	fmt.Fprintln(w, "  --json-schema-strict[=bool]    OpenAI strict schema enforcement (default true when --json-schema is set).")
	fmt.Fprintln(w, "  --verbose                      Print diagnostics to stderr.")
	fmt.Fprintln(w, "  --version                      Print version.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "OpenAI-only flags:")
	fmt.Fprintln(w, "  --openai-service-tier <tier>           auto|default|flex|priority")
	fmt.Fprintln(w, "  --openai-truncation <strategy>         auto|disabled")
	fmt.Fprintln(w, "  --openai-store[=bool]                 Store responses server-side (default false when set).")
	fmt.Fprintln(w, "  --openai-prompt-cache-key <key>        prompt_cache_key")
	fmt.Fprintln(w, "  --openai-prompt-cache-retention <dur>  prompt_cache_retention (e.g. 24h)")
	fmt.Fprintln(w, "  --openai-safety-identifier <id>        safety_identifier")
	fmt.Fprintln(w, "  --openai-metadata key=value            metadata (repeatable)")
	fmt.Fprintln(w, "  --openai-include <csv>                 include fields (comma-separated)")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Ollama-only flags:")
	fmt.Fprintln(w, "  --ollama-host <url>            Base URL (overrides OLLAMA_HOST).")
	fmt.Fprintln(w, "  --ollama-endpoint <chat|generate> Endpoint selection (default chat).")
	fmt.Fprintln(w, "  --ollama-keep-alive <v>        keep_alive value (e.g. 5m or 0).")
	fmt.Fprintln(w, "  --ollama-think[=bool]          Enable thinking (model-dependent).")
	fmt.Fprintln(w, "  --ollama-stream[=bool]         Enable/disable streaming (default true).")
	fmt.Fprintln(w, "  --ollama-format <json>         Request JSON output (Ollama format=\"json\").")
	fmt.Fprintln(w, "  --ollama-top-k <int>           options.top_k")
	fmt.Fprintln(w, "  --ollama-num-ctx <int>         options.num_ctx")
	fmt.Fprintln(w, "  --ollama-seed <int>            options.seed")
	fmt.Fprintln(w, "  --ollama-stop <csv>            options.stop (comma-separated)")
	fmt.Fprintln(w, "  --ollama-raw[=bool]            /api/generate raw mode")
	fmt.Fprintln(w, "  --ollama-option key=value      Arbitrary Ollama option (repeatable).")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Examples:")
	fmt.Fprintln(w, "  # OpenAI one-shot")
	fmt.Fprintln(w, "  OPENAI_API_KEY=... textualai --model openai:gpt-4.1 --message \"Write a haiku about terminals\"")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  # Ollama one-shot (defaults to http://localhost:11434)")
	fmt.Fprintln(w, "  textualai --model ollama:llama3.1 --message \"Explain monads\"")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  # Loop mode")
	fmt.Fprintln(w, "  textualai --model ollama:llama3.1 --loop")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "  # Template file")
	fmt.Fprintln(w, "  textualai --model openai:gpt-4.1 --prompt-template ./prompt.tmpl --message \"Hello\"")
}

func loadPromptTemplate(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return defaultPromptTemplate, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read prompt template %q: %w", path, err)
	}
	s := strings.TrimSpace(string(b))
	if s == "" {
		return "", fmt.Errorf("prompt template %q is empty", path)
	}
	return s, nil
}

func readMessageFile(path string, encodingName string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", nil
	}

	encodingName = strings.TrimSpace(encodingName)
	if encodingName == "" {
		encodingName = "UTF-8"
	}

	encID, err := textual.ParseEncoding(encodingName)
	if err != nil {
		return "", fmt.Errorf("invalid --file-encoding %q: %w", encodingName, err)
	}

	var r io.Reader
	if path == "-" {
		r = os.Stdin
	} else {
		f, err := os.Open(path)
		if err != nil {
			return "", fmt.Errorf("open message file %q: %w", path, err)
		}
		defer f.Close()
		r = f
	}

	utf8r, err := textual.NewUTF8Reader(r, encID)
	if err != nil {
		return "", fmt.Errorf("create UTF-8 reader: %w", err)
	}
	b, err := io.ReadAll(utf8r)
	if err != nil {
		return "", fmt.Errorf("read message: %w", err)
	}
	return strings.TrimSpace(string(b)), nil
}

func combineMessage(message string, fileMsg string) string {
	message = strings.TrimSpace(message)
	fileMsg = strings.TrimSpace(fileMsg)
	switch {
	case message == "" && fileMsg == "":
		return ""
	case message == "":
		return fileMsg
	case fileMsg == "":
		return message
	default:
		// Keep the explicit --message first so CLI users can prepend a short
		// instruction before the longer file content.
		return strings.TrimSpace(message + "\n\n" + fileMsg)
	}
}

func resolveProvider(providerFlag string, model string) (providerKind, string, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		return providerAuto, "", fmt.Errorf("model must not be empty")
	}

	// Prefix-based selection always wins.
	lower := strings.ToLower(model)
	switch {
	case strings.HasPrefix(lower, "openai:"):
		return providerOpenAI, strings.TrimSpace(model[len("openai:"):]), nil
	case strings.HasPrefix(lower, "oa:"):
		return providerOpenAI, strings.TrimSpace(model[len("oa:"):]), nil
	case strings.HasPrefix(lower, "ollama:"):
		return providerOllama, strings.TrimSpace(model[len("ollama:"):]), nil
	}

	// Explicit provider flag.
	switch strings.ToLower(strings.TrimSpace(providerFlag)) {
	case "", "auto":
		// fallthrough to heuristics below
	case "openai":
		return providerOpenAI, model, nil
	case "ollama":
		return providerOllama, model, nil
	default:
		return providerAuto, "", fmt.Errorf("unknown provider %q (expected auto|openai|ollama)", providerFlag)
	}

	// Heuristics for auto mode.
	// We keep this deliberately simple and deterministic:
	//   - OpenAI: models starting with gpt* or o* (o1, o3, etc.)
	//   - Ollama: everything else
	if strings.HasPrefix(lower, "gpt") || strings.HasPrefix(lower, "o") {
		return providerOpenAI, model, nil
	}
	return providerOllama, model, nil
}

func providerName(p providerKind) string {
	switch p {
	case providerOpenAI:
		return "openai"
	case providerOllama:
		return "ollama"
	default:
		return "auto"
	}
}

func loadJSONSchema(path string) (map[string]any, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read json schema %q: %w", path, err)
	}
	var schema map[string]any
	if err := json.Unmarshal(b, &schema); err != nil {
		return nil, fmt.Errorf("parse json schema %q: %w", path, err)
	}
	if len(schema) == 0 {
		return nil, fmt.Errorf("json schema %q is empty", path)
	}
	return schema, nil
}

func runOpenAI(
	rootCtx context.Context,
	cfg cliConfig,
	model string,
	templateStr string,
	jsonSchema map[string]any,
	initialMsg string,
	outw *bufio.Writer,
	stderr io.Writer,
) int {
	// OpenAI key check is performed again by the processor, but the CLI gives a
	// more actionable error message.
	if len(strings.TrimSpace(os.Getenv("OPENAI_API_KEY"))) < 10 {
		fmt.Fprintln(stderr, "Error: missing or invalid OPENAI_API_KEY (required for OpenAI provider)")
		return 1
	}

	procPtr, err := textualopenai.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 1
	}
	proc := *procPtr

	// Streaming aggregation.
	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualopenai.Line)
	default:
		proc = proc.WithAggregateType(textualopenai.Word)
	}

	// Role.
	if r := strings.ToLower(strings.TrimSpace(cfg.Role)); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualopenai.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualopenai.RoleAssistant)
		case "system":
			proc = proc.WithRole(textualopenai.RoleSystem)
		case "developer":
			proc = proc.WithRole(textualopenai.RoleDeveloper)
		default:
			fmt.Fprintf(stderr, "Warning: unsupported --role %q for OpenAI; using default\n", cfg.Role)
		}
	}

	// Instructions.
	if strings.TrimSpace(cfg.Instructions) != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	// Sampling.
	if cfg.Temperature.set {
		proc = proc.WithTemperature(cfg.Temperature.val)
	}
	if cfg.TopP.set {
		proc = proc.WithTopP(cfg.TopP.val)
	}
	if cfg.MaxTokens.set {
		proc = proc.WithMaxOutputTokens(cfg.MaxTokens.val)
	}

	// Structured outputs / JSON schema.
	if jsonSchema != nil {
		strict := cfg.JSONSchemaStrict.val
		proc = proc.WithTextFormatJSONSchema(textualopenai.JSONSchemaFormat{
			Type: "json_schema",
			JSONSchema: textualopenai.JSONSchema{
				Name:   strings.TrimSpace(cfg.JSONSchemaName),
				Schema: jsonSchema,
				Strict: &strict,
			},
		})
	} else {
		proc = proc.WithTextFormatText()
	}

	// OpenAI-only: Service tier, truncation, store, caching, metadata, include...
	if strings.TrimSpace(cfg.OpenAIServiceTier) != "" {
		proc = proc.WithServiceTier(textualopenai.ServiceTier(cfg.OpenAIServiceTier))
	}
	if strings.TrimSpace(cfg.OpenAITruncation) != "" {
		proc = proc.WithTruncation(textualopenai.TruncationStrategy(cfg.OpenAITruncation))
	}
	if cfg.OpenAIStore.set {
		proc = proc.WithStore(cfg.OpenAIStore.val)
	}
	if strings.TrimSpace(cfg.OpenAIPromptCacheKey) != "" {
		proc = proc.WithPromptCacheKey(cfg.OpenAIPromptCacheKey)
	}
	if strings.TrimSpace(cfg.OpenAIPromptCacheRetention) != "" {
		proc = proc.WithPromptCacheRetention(textualopenai.PromptCacheRetention(cfg.OpenAIPromptCacheRetention))
	}
	if strings.TrimSpace(cfg.OpenAISafetyIdentifier) != "" {
		proc = proc.WithSafetyIdentifier(cfg.OpenAISafetyIdentifier)
	}
	if cfg.OpenAIMetadata != nil && len(cfg.OpenAIMetadata) > 0 {
		proc = proc.WithMetadata(map[string]string(cfg.OpenAIMetadata))
	}
	if strings.TrimSpace(cfg.OpenAIInclude) != "" {
		proc = proc.WithInclude(splitCSV(cfg.OpenAIInclude)...)
	}

	// One-shot vs loop.
	if cfg.Loop.Enabled() {
		return interactiveLoop(rootCtx, cfg, func(ctx context.Context, msg string) (string, error) {
			return streamOnce(ctx, proc, msg, outw)
		}, initialMsg, outw, stderr)
	}

	ctx, cancel := withOptionalTimeout(rootCtx, cfg.Timeout)
	defer cancel()

	final, err := streamOnce(ctx, proc, initialMsg, outw)
	_ = final
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 1
	}
	fmt.Fprintln(outw) // final newline
	outw.Flush()
	return 0
}

func runOllama(
	rootCtx context.Context,
	cfg cliConfig,
	model string,
	templateStr string,
	jsonSchema map[string]any,
	initialMsg string,
	outw *bufio.Writer,
	stderr io.Writer,
) int {
	procPtr, err := textualollama.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 1
	}
	proc := *procPtr

	// Streaming aggregation.
	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualollama.Line)
	default:
		proc = proc.WithAggregateType(textualollama.Word)
	}

	// Role.
	if r := strings.ToLower(strings.TrimSpace(cfg.Role)); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualollama.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualollama.RoleAssistant)
		case "system":
			proc = proc.WithRole(textualollama.RoleSystem)
		default:
			fmt.Fprintf(stderr, "Warning: unsupported --role %q for Ollama; using default\n", cfg.Role)
		}
	}

	// Base URL override.
	if strings.TrimSpace(cfg.OllamaHost) != "" {
		proc = proc.WithBaseURL(cfg.OllamaHost)
	}

	// Endpoint.
	switch strings.ToLower(strings.TrimSpace(cfg.OllamaEndpoint)) {
	case "", "chat":
		proc = proc.WithChatEndpoint()
	case "generate":
		proc = proc.WithGenerateEndpoint()
	default:
		fmt.Fprintf(stderr, "Warning: unknown --ollama-endpoint %q; using chat\n", cfg.OllamaEndpoint)
		proc = proc.WithChatEndpoint()
	}

	// Instructions (system prompt).
	if strings.TrimSpace(cfg.Instructions) != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	// Streaming toggle (Ollama supports stream=false).
	if cfg.OllamaStream.set {
		proc = proc.WithStream(cfg.OllamaStream.val)
	}

	// Keep alive.
	if strings.TrimSpace(cfg.OllamaKeepAlive) != "" {
		proc = proc.WithKeepAlive(parseKeepAlive(cfg.OllamaKeepAlive))
	}

	// Thinking.
	if cfg.OllamaThink.set {
		proc = proc.WithThink(cfg.OllamaThink.val)
	}

	// Sampling / common controls.
	if cfg.Temperature.set {
		proc = proc.WithTemperature(cfg.Temperature.val)
	}
	if cfg.TopP.set {
		proc = proc.WithTopP(cfg.TopP.val)
	}
	if cfg.MaxTokens.set {
		proc = proc.WithNumPredict(cfg.MaxTokens.val)
	}

	// Ollama model options.
	if cfg.OllamaTopK.set {
		proc = proc.WithTopK(cfg.OllamaTopK.val)
	}
	if cfg.OllamaNumCtx.set {
		proc = proc.WithNumCtx(cfg.OllamaNumCtx.val)
	}
	if cfg.OllamaSeed.set {
		proc = proc.WithSeed(cfg.OllamaSeed.val)
	}
	if strings.TrimSpace(cfg.OllamaStop) != "" {
		proc = proc.WithStop(splitCSV(cfg.OllamaStop)...)
	}

	// Raw mode is generate-only; we set it regardless and Ollama will ignore
	// if using the chat endpoint.
	if cfg.OllamaRaw.set {
		proc = proc.WithRaw(cfg.OllamaRaw.val)
	}

	// Extra options.
	if cfg.OllamaExtraOpt != nil && len(cfg.OllamaExtraOpt) > 0 {
		for k, v := range cfg.OllamaExtraOpt {
			proc = proc.WithOption(k, v)
		}
	}

	// Output formatting.
	//  1) JSON schema file (Structured Outputs) wins
	//  2) --ollama-format=json can request a plain JSON response
	if jsonSchema != nil {
		proc = proc.WithFormat(jsonSchema)
	} else if strings.EqualFold(strings.TrimSpace(cfg.OllamaFormat), "json") {
		proc = proc.WithFormatJSON()
	}

	// One-shot vs loop.
	if cfg.Loop.Enabled() {
		return interactiveLoop(rootCtx, cfg, func(ctx context.Context, msg string) (string, error) {
			return streamOnce(ctx, proc, msg, outw)
		}, initialMsg, outw, stderr)
	}

	ctx, cancel := withOptionalTimeout(rootCtx, cfg.Timeout)
	defer cancel()

	final, err := streamOnce(ctx, proc, initialMsg, outw)
	_ = final
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 1
	}
	fmt.Fprintln(outw) // final newline
	outw.Flush()
	return 0
}

func withOptionalTimeout(parent context.Context, d optDuration) (context.Context, context.CancelFunc) {
	if !d.set || d.val <= 0 {
		return parent, func() {}
	}
	return context.WithTimeout(parent, d.val)
}

// streamOnce runs one processor invocation and streams its output to outw.
// It returns the final accumulated text (useful for higher-level loops if you
// want to keep conversation memory in the future).
func streamOnce[P textual.Processor[textual.String]](
	ctx context.Context,
	proc P,
	message string,
	outw *bufio.Writer,
) (string, error) {
	message = strings.TrimSpace(message)
	if message == "" {
		return "", errors.New("empty message")
	}

	in := make(chan textual.String, 1)
	// Use textual.String as a minimal carrier.
	in <- textual.String{}.FromUTF8String(message).WithIndex(0)
	close(in)

	out := proc.Apply(ctx, in)

	var last string
	for item := range out {
		if err := item.GetError(); err != nil {
			// Processor errors are carried as data; surface them as an error.
			return last, err
		}
		snapshot := item.UTF8String()
		delta := snapshot
		if strings.HasPrefix(snapshot, last) {
			delta = snapshot[len(last):]
		}
		if delta != "" {
			if _, err := outw.WriteString(delta); err != nil {
				return last, err
			}
			_ = outw.Flush()
		}
		last = snapshot
	}

	return last, nil
}

type streamFn func(ctx context.Context, msg string) (string, error)

func interactiveLoop(
	rootCtx context.Context,
	cfg cliConfig,
	fn streamFn,
	initialMsg string,
	outw *bufio.Writer,
	stderr io.Writer,
) int {
	exitCmds := make(map[string]struct{})
	for _, c := range splitCSV(cfg.ExitCommands) {
		if t := strings.TrimSpace(c); t != "" {
			exitCmds[strings.ToLower(t)] = struct{}{}
		}
	}

	inReader := bufio.NewReader(os.Stdin)

	// In loop mode we optionally run an initial message (from flags) before
	// prompting the user.
	if strings.TrimSpace(initialMsg) != "" {
		ctx, cancel := withOptionalTimeout(rootCtx, cfg.Timeout)
		_, err := fn(ctx, initialMsg)
		cancel()
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			fmt.Fprintln(outw)
			outw.Flush()
			return 1
		}
		fmt.Fprintln(outw)
		outw.Flush()
	}

	for {
		select {
		case <-rootCtx.Done():
			fmt.Fprintln(stderr, "\nCanceled.")
			return 1
		default:
		}

		// Prompt.
		fmt.Fprint(outw, "> ")
		outw.Flush()

		line, err := inReader.ReadString('\n')
		if err != nil {
			// EOF => normal exit.
			if errors.Is(err, io.EOF) {
				fmt.Fprintln(outw)
				outw.Flush()
				return 0
			}
			fmt.Fprintln(stderr, "Error reading stdin:", err)
			return 1
		}

		msg := strings.TrimSpace(line)
		if msg == "" {
			// Empty line: ignore.
			continue
		}

		if _, ok := exitCmds[strings.ToLower(msg)]; ok {
			return 0
		}

		ctx, cancel := withOptionalTimeout(rootCtx, cfg.Timeout)
		_, err = fn(ctx, msg)
		cancel()
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			fmt.Fprintln(outw)
			outw.Flush()
			// Keep looping after an error; this is a chat REPL, not a batch job.
			continue
		}
		fmt.Fprintln(outw)
		outw.Flush()
	}
}

func parseKeepAlive(s string) any {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	// Ollama accepts 0 as a number (unload immediately).
	if i, err := strconv.Atoi(s); err == nil {
		return i
	}
	return s
}

func splitCSV(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func init() {
	// Make sure examples that use relative schema/template paths can show a
	// helpful working directory in verbose mode if needed.
	_, _ = filepath.Abs(".")
}
