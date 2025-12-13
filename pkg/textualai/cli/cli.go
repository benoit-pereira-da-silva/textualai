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

// Package cli implements the `textualai` command-line interface runtime.
//
// textualai is a small streaming CLI that can target multiple providers with a
// single command line.
//
// Interoperability goals:
//
//   - Input: send either a raw string (--message) or a JSON value (--object /
//     --object-file). When using a template (--template / --template-file), the
//     JSON value becomes the Go template root ({{.}}). If the value is a string,
//     {{.}} is that string; otherwise {{.}} is the parsed JSON value.
//
//   - Output: stream plain text by default, or request and validate schema-based
//     structured output with --output-schema.
//
// Schema validation (input and output) uses github.com/google/jsonschema-go/jsonschema.
package cli

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"text/template"
	"time"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualclaude"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualgemini"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualmistral"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualollama"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualopenai"

	"github.com/google/jsonschema-go/jsonschema"
)

// version is set at build time using:
//
//	go build -ldflags "-X github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli.version=v1.2.3"
//
// When not set, it defaults to "dev".
var version = "dev"

// Version returns the build-time version string printed by --version.
func Version() string { return version }

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

// Run is the default textualai CLI entry point.
//
// It is intended to be called from a tiny main:
//
//	func main() { os.Exit(cli.Run(os.Args, os.Stdout, os.Stderr)) }
func Run(argv []string, stdout io.Writer, stderr io.Writer) int {
	return NewRunner().Run(argv, stdout, stderr)
}

// PrintUsage prints the CLI usage help.
func PrintUsage(w io.Writer) {
	printUsage(w)
}

// ProviderBuilder builds a provider-specific Processor based on the CLI config.
type ProviderBuilder func(
	ctx context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	stderr io.Writer,
) (textual.Processor[textual.String], error)

// Streamer runs one prompt through the processor and streams the output to
// outw. It returns the final accumulated text (last snapshot).
type Streamer func(
	ctx context.Context,
	proc textual.Processor[textual.String],
	prompt string,
	outw *bufio.Writer,
) (string, error)

// Runner is a configurable CLI runtime that can be embedded by third parties.
type Runner struct {
	Stdin  io.Reader
	Getenv func(string) string
	Usage  func(io.Writer)

	Parse            func(args []string) (Config, error)
	ProviderResolver func(providerFlag string, model string) (ProviderKind, string, error)

	OpenAIBuilder  ProviderBuilder
	ClaudeBuilder  ProviderBuilder
	GeminiBuilder  ProviderBuilder
	MistralBuilder ProviderBuilder
	OllamaBuilder  ProviderBuilder

	Streamer Streamer
}

// Option configures a Runner.
type Option func(*Runner)

// NewRunner constructs a Runner configured with the default textualai behavior.
// Use options to override specific extension points.
func NewRunner(opts ...Option) *Runner {
	r := &Runner{
		Stdin:            os.Stdin,
		Getenv:           os.Getenv,
		Usage:            PrintUsage,
		ProviderResolver: ResolveProvider,
		OpenAIBuilder:    DefaultOpenAIBuilder,
		ClaudeBuilder:    DefaultClaudeBuilder,
		GeminiBuilder:    DefaultGeminiBuilder,
		MistralBuilder:   DefaultMistralBuilder,
		OllamaBuilder:    DefaultOllamaBuilder,
		Streamer:         DefaultStreamer,
	}

	for _, opt := range opts {
		if opt != nil {
			opt(r)
		}
	}

	r.ensureDefaults()
	return r
}

func (r *Runner) ensureDefaults() {
	if r.Stdin == nil {
		r.Stdin = os.Stdin
	}
	if r.Getenv == nil {
		r.Getenv = os.Getenv
	}
	if r.Usage == nil {
		r.Usage = PrintUsage
	}
	if r.ProviderResolver == nil {
		r.ProviderResolver = ResolveProvider
	}
	if r.OpenAIBuilder == nil {
		r.OpenAIBuilder = DefaultOpenAIBuilder
	}
	if r.ClaudeBuilder == nil {
		r.ClaudeBuilder = DefaultClaudeBuilder
	}
	if r.GeminiBuilder == nil {
		r.GeminiBuilder = DefaultGeminiBuilder
	}
	if r.MistralBuilder == nil {
		r.MistralBuilder = DefaultMistralBuilder
	}
	if r.OllamaBuilder == nil {
		r.OllamaBuilder = DefaultOllamaBuilder
	}
	if r.Streamer == nil {
		r.Streamer = DefaultStreamer
	}
	if r.Parse == nil {
		get := r.Getenv
		r.Parse = func(args []string) (Config, error) { return parseCLI(args, get) }
	}
}

// Run executes the CLI against argv.
func (r *Runner) Run(argv []string, stdout io.Writer, stderr io.Writer) int {
	r.ensureDefaults()

	if len(argv) == 0 {
		argv = []string{"textualai"}
	}

	// "textualai help" convenience command.
	if len(argv) >= 2 {
		switch strings.TrimSpace(argv[1]) {
		case "help", "usage", "--help", "-help", "-h", "--h":
			r.Usage(stdout)
			return 0
		}
	}

	cfg, err := r.Parse(argv[1:])
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		fmt.Fprintln(stderr)
		r.Usage(stderr)
		return 2
	}

	if cfg.Help.Enabled() {
		r.Usage(stdout)
		return 0
	}
	if cfg.Version.Enabled() {
		fmt.Fprintln(stdout, version)
		return 0
	}

	if strings.TrimSpace(cfg.Model) == "" {
		fmt.Fprintln(stderr, "Error: --model is required")
		fmt.Fprintln(stderr)
		r.Usage(stderr)
		return 2
	}

	// Load and compile output schema (optional).
	var outputSchemaMap map[string]any
	var outputResolved *jsonschema.Resolved
	if strings.TrimSpace(cfg.OutputSchema) != "" {
		compiled, err := loadCompiledSchema(cfg.OutputSchema, r.Stdin)
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			return 2
		}
		outputSchemaMap = compiled.SchemaMap
		outputResolved = compiled.Resolved
	}

	// Parse and validate the optional template schema (input validation).
	var inputResolved *jsonschema.Resolved
	if strings.TrimSpace(cfg.TemplateSchema) != "" {
		compiled, err := loadCompiledSchema(cfg.TemplateSchema, r.Stdin)
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			return 2
		}
		inputResolved = compiled.Resolved
	}

	// Prepare renderer (optional template). We parse templates once per run.
	render, err := prepareRenderer(cfg, r.Stdin)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 2
	}

	// Root context reacts to Ctrl+C (SIGINT) and SIGTERM.
	rootCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Resolve provider and normalize model (strip openai:/claude:/... prefix).
	prov, modelName, err := r.ProviderResolver(cfg.Provider, cfg.Model)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 2
	}

	if cfg.Verbose.Enabled() {
		fmt.Fprintf(stderr, "Provider: %s\n", prov.String())
		fmt.Fprintf(stderr, "Model: %s\n", modelName)
		if cfg.Template != "" {
			fmt.Fprintf(stderr, "Template: (inline)\n")
		} else if cfg.TemplateFile != "" {
			fmt.Fprintf(stderr, "Template: %s\n", cfg.TemplateFile)
		} else {
			fmt.Fprintf(stderr, "Template: (none)\n")
		}
		if cfg.OutputSchema != "" {
			fmt.Fprintf(stderr, "Output schema: %s\n", cfg.OutputSchema)
		}
		if cfg.TemplateSchema != "" {
			fmt.Fprintf(stderr, "Template schema: %s\n", cfg.TemplateSchema)
		}
		fmt.Fprintln(stderr)
	}

	// Build the provider processor (default behavior).
	var proc textual.Processor[textual.String]
	switch prov {
	case ProviderOpenAI:
		proc, err = r.OpenAIBuilder(rootCtx, cfg, modelName, "{{.Input}}", outputSchemaMap, r.Getenv, stderr)
	case ProviderClaude:
		proc, err = r.ClaudeBuilder(rootCtx, cfg, modelName, "{{.Input}}", outputSchemaMap, r.Getenv, stderr)
	case ProviderGemini:
		proc, err = r.GeminiBuilder(rootCtx, cfg, modelName, "{{.Input}}", outputSchemaMap, r.Getenv, stderr)
	case ProviderMistral:
		proc, err = r.MistralBuilder(rootCtx, cfg, modelName, "{{.Input}}", outputSchemaMap, r.Getenv, stderr)
	case ProviderOllama:
		proc, err = r.OllamaBuilder(rootCtx, cfg, modelName, "{{.Input}}", outputSchemaMap, r.Getenv, stderr)
	default:
		err = fmt.Errorf("unable to resolve provider")
	}
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 1
	}
	if proc == nil {
		fmt.Fprintln(stderr, "Error: nil processor (provider builder returned nil)")
		return 1
	}

	// Prepare a buffered writer for smooth streaming.
	outw := bufio.NewWriter(stdout)
	defer outw.Flush()

	// One-shot vs loop.
	if cfg.Loop.Enabled() {
		return interactiveLoop(
			rootCtx,
			cfg,
			proc,
			render,
			inputResolved,
			outputResolved,
			r.Stdin,
			outw,
			stderr,
			r.Streamer,
		)
	}

	// One-shot: build input value, validate (optional), render prompt.
	inputVal, err := loadOneShotInput(cfg, r.Stdin)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		fmt.Fprintln(stderr)
		r.Usage(stderr)
		return 2
	}
	if inputVal == nil {
		fmt.Fprintln(stderr, "Error: provide --message or --object/--object-file, or use --loop")
		fmt.Fprintln(stderr)
		r.Usage(stderr)
		return 2
	}

	// Optional input validation.
	if inputResolved != nil {
		if err := validateAgainstSchema(inputResolved, inputVal); err != nil {
			fmt.Fprintln(stderr, "Error: input does not match --template-schema:", err)
			return 2
		}
	}

	prompt, err := render(inputVal)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 2
	}

	ctx, cancel := withOptionalTimeout(rootCtx, cfg.Timeout)
	defer cancel()

	final, err := r.Streamer(ctx, proc, prompt, outw)
	if err != nil {
		fmt.Fprintln(stderr, "Error:", err)
		return 1
	}
	fmt.Fprintln(outw) // final newline
	_ = outw.Flush()

	// Optional output validation.
	if outputResolved != nil {
		v, err := parseJSONAny(final)
		if err != nil {
			fmt.Fprintln(stderr, "Error: output is not valid JSON:", err)
			return 1
		}
		if err := validateAgainstSchema(outputResolved, v); err != nil {
			fmt.Fprintln(stderr, "Error: output does not match --output-schema:", err)
			return 1
		}
	}

	return 0
}

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

// ------------------------------
// Input loading + prompt rendering
// ------------------------------

func loadOneShotInput(cfg Config, stdin io.Reader) (any, error) {
	// One-shot: select exactly one input source.
	if strings.TrimSpace(cfg.Message) != "" {
		// Raw string input.
		if cfg.Object != "" || cfg.ObjectFile != "" {
			return nil, fmt.Errorf("use --message OR --object/--object-file (not both)")
		}
		return cfg.Message, nil
	}

	if cfg.Object != "" {
		v, err := parseJSONAny(cfg.Object)
		if err != nil {
			return nil, fmt.Errorf("parse --object: %w", err)
		}
		return v, nil
	}
	if cfg.ObjectFile != "" {
		raw, err := readFileOrStdin(cfg.ObjectFile, stdin)
		if err != nil {
			return nil, fmt.Errorf("read --object-file: %w", err)
		}
		v, err := parseJSONAny(raw)
		if err != nil {
			return nil, fmt.Errorf("parse --object-file JSON: %w", err)
		}
		return v, nil
	}

	return nil, nil
}

type renderer func(input any) (string, error)

func prepareRenderer(cfg Config, stdin io.Reader) (renderer, error) {
	// No template: identity for string inputs only.
	if cfg.Template == "" && cfg.TemplateFile == "" {
		return func(input any) (string, error) {
			switch v := input.(type) {
			case string:
				return strings.TrimSpace(v), nil
			default:
				return "", errors.New("non-string JSON input requires --template or --template-file")
			}
		}, nil
	}

	// Load template text.
	var tmplText string
	if cfg.TemplateFile != "" {
		b, err := readFileOrStdin(cfg.TemplateFile, stdin)
		if err != nil {
			return nil, fmt.Errorf("read template file: %w", err)
		}
		tmplText = b
	} else {
		tmplText = cfg.Template
	}

	tmplText = strings.TrimSpace(tmplText)
	if tmplText == "" {
		return nil, fmt.Errorf("template is empty")
	}

	// Parse template once and reuse.
	tmpl, err := template.New("textualai.template").Parse(tmplText)
	if err != nil {
		return nil, fmt.Errorf("parse template: %w", err)
	}

	return func(input any) (string, error) {
		var buf bytes.Buffer
		if err := tmpl.Execute(&buf, input); err != nil {
			return "", err
		}
		return strings.TrimSpace(buf.String()), nil
	}, nil
}

func readFileOrStdin(path string, stdin io.Reader) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", nil
	}
	if stdin == nil {
		stdin = os.Stdin
	}
	var r io.Reader
	if path == "-" {
		r = stdin
	} else {
		b, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
	b, err := io.ReadAll(r)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func parseJSONAny(s string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	// Ensure no trailing tokens.
	if dec.More() {
		return nil, fmt.Errorf("trailing JSON tokens")
	}
	// Try reading one more token.
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("trailing JSON value")
		}
		return nil, err
	}
	return v, nil
}

// ------------------------------
// Schema loading / validation
// ------------------------------

// CompiledSchema holds a JSON Schema in two forms:
//   - SchemaMap: for passing into provider request options
//   - Resolved: for local validation via Resolved.Validate
type CompiledSchema struct {
	Path      string
	SchemaMap map[string]any
	Resolved  *jsonschema.Resolved
}

// loadCompiledSchema reads a JSON schema file (or stdin when path is "-"),
// parses it into map form (for provider requests) and into jsonschema.Schema,
// resolves refs, and returns a CompiledSchema.
//
// For schema files on disk, it sets ResolveOptions.BaseURI to an absolute file:// URI
// so that relative $ref paths can be resolved.
func loadCompiledSchema(path string, stdin io.Reader) (*CompiledSchema, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, nil
	}

	raw, err := readFileOrStdin(path, stdin)
	if err != nil {
		return nil, err
	}

	// Provider schema map.
	var schemaMap map[string]any
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&schemaMap); err != nil {
		return nil, fmt.Errorf("parse schema JSON: %w", err)
	}
	if len(schemaMap) == 0 {
		return nil, fmt.Errorf("schema is empty")
	}

	// jsonschema.Schema for compilation/validation.
	var s jsonschema.Schema
	dec2 := json.NewDecoder(strings.NewReader(raw))
	dec2.UseNumber()
	if err := dec2.Decode(&s); err != nil {
		return nil, fmt.Errorf("parse schema (typed): %w", err)
	}

	// Resolve.
	opts := &jsonschema.ResolveOptions{
		ValidateDefaults: true,
	}
	if path != "-" {
		baseURI, baseDir, err := fileURIAndDir(path)
		if err != nil {
			return nil, err
		}
		opts.BaseURI = baseURI
		opts.Loader = fileSchemaLoader(baseDir, stdin)
	}

	resolved, err := s.Resolve(opts)
	if err != nil {
		return nil, err
	}

	return &CompiledSchema{
		Path:      path,
		SchemaMap: schemaMap,
		Resolved:  resolved,
	}, nil
}

func fileURIAndDir(path string) (baseURI string, baseDir string, err error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", "", fmt.Errorf("abs path %q: %w", path, err)
	}
	u := url.URL{Scheme: "file", Path: abs}
	return u.String(), filepath.Dir(abs), nil
}

// fileSchemaLoader loads schemas for remote references (anything not under the
// root schema). In practice, this supports file:// URIs (and bare/relative
// paths as a fallback).
//
// If you want to allow http(s) refs, implement a different Loader (or wrap this
// one) and pass it into Schema.Resolve.
func fileSchemaLoader(baseDir string, stdin io.Reader) jsonschema.Loader {
	return func(uri *url.URL) (*jsonschema.Schema, error) {
		if uri == nil {
			return nil, fmt.Errorf("nil schema URI")
		}

		// Ignore fragments when reading the file; fragment resolution is handled
		// by the jsonschema resolver.
		u := *uri
		u.Fragment = ""

		switch u.Scheme {
		case "", "file":
			p := u.Path
			// Fallback: if we got a relative path with an empty scheme, treat it as file under baseDir.
			if p == "" {
				p = u.String()
			}
			if p == "" {
				return nil, fmt.Errorf("empty schema path for URI %q", uri.String())
			}

			// If it's not absolute (possible with empty scheme), anchor it to baseDir.
			if !filepath.IsAbs(p) && strings.TrimSpace(baseDir) != "" {
				p = filepath.Join(baseDir, p)
			}

			// Special case: allow "-" only when the URI explicitly says so (rare).
			raw, err := readFileOrStdin(p, stdin)
			if err != nil {
				return nil, err
			}

			var s jsonschema.Schema
			dec := json.NewDecoder(strings.NewReader(raw))
			dec.UseNumber()
			if err := dec.Decode(&s); err != nil {
				return nil, fmt.Errorf("parse referenced schema %q: %w", uri.String(), err)
			}
			return &s, nil
		default:
			return nil, fmt.Errorf("unsupported schema URI scheme %q (uri=%q)", u.Scheme, uri.String())
		}
	}
}

func validateAgainstSchema(res *jsonschema.Resolved, v any) error {
	if res == nil {
		return nil
	}
	return res.Validate(v)
}

// ------------------------------
// Provider builders
// ------------------------------

func DefaultOpenAIBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if len(strings.TrimSpace(getenv("OPENAI_API_KEY"))) < 10 {
		return nil, errors.New("missing or invalid OPENAI_API_KEY (required for OpenAI provider)")
	}

	procPtr, err := textualopenai.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualopenai.Line)
	default:
		proc = proc.WithAggregateType(textualopenai.Word)
	}

	// Role.
	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualopenai.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualopenai.RoleAssistant)
		case "system":
			proc = proc.WithRole(textualopenai.RoleSystem)
		case "developer":
			proc = proc.WithRole(textualopenai.RoleDeveloper)
		}
	}

	// Instructions.
	if cfg.Instructions != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	// Sampling.
	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithMaxOutputTokens(cfg.MaxTokens.Value())
	}

	// Structured output.
	if outputSchema != nil {
		strict := true
		proc = proc.WithTextFormatJSONSchema(textualopenai.JSONSchemaFormat{
			Type: "json_schema",
			JSONSchema: textualopenai.JSONSchema{
				Name:   "response",
				Schema: outputSchema,
				Strict: &strict,
			},
		})
	} else {
		proc = proc.WithTextFormatText()
	}

	return proc, nil
}

func DefaultClaudeBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if len(strings.TrimSpace(getenv("ANTHROPIC_API_KEY"))) < 10 {
		return nil, errors.New("missing or invalid ANTHROPIC_API_KEY (required for Claude provider)")
	}

	procPtr, err := textualclaude.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualclaude.Line)
	default:
		proc = proc.WithAggregateType(textualclaude.Word)
	}

	// Role (Claude messages roles: user|assistant).
	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualclaude.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualclaude.RoleAssistant)
		}
	}

	// Overrides.
	if cfg.ClaudeBaseURL != "" {
		proc = proc.WithBaseURL(cfg.ClaudeBaseURL)
	}
	if cfg.ClaudeAPIVersion != "" {
		proc = proc.WithAPIVersion(cfg.ClaudeAPIVersion)
	}
	if cfg.ClaudeStream.IsSet() {
		proc = proc.WithStream(cfg.ClaudeStream.Value())
	}

	// Instructions (system prompt).
	if cfg.Instructions != "" {
		proc = proc.WithSystem(cfg.Instructions)
	}

	// Sampling.
	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithMaxTokens(cfg.MaxTokens.Value())
	}

	// Structured output via tool-use (portable approach).
	if outputSchema != nil {
		toolName := "response"
		proc = proc.WithTools(textualclaude.Tool{
			Name:        toolName,
			Description: "Return a response that matches the provided JSON Schema.",
			InputSchema: outputSchema,
		})
		proc = proc.WithToolChoice(textualclaude.ToolChoice{
			Type: "tool",
			Name: toolName,
		})
		proc = proc.WithEmitToolUse(true)
	}

	return proc, nil
}

func DefaultGeminiBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if len(strings.TrimSpace(getenv("GEMINI_API_KEY"))) < 10 {
		return nil, errors.New("missing or invalid GEMINI_API_KEY (required for Gemini provider)")
	}

	procPtr, err := textualgemini.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualgemini.Line)
	default:
		proc = proc.WithAggregateType(textualgemini.Word)
	}

	// Role (Gemini roles: user|model).
	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualgemini.RoleUser)
		case "assistant", "model":
			proc = proc.WithRole(textualgemini.RoleModel)
		}
	}

	// Overrides.
	if cfg.GeminiBaseURL != "" {
		proc = proc.WithBaseURL(cfg.GeminiBaseURL)
	}
	if cfg.GeminiAPIVersion != "" {
		proc = proc.WithAPIVersion(cfg.GeminiAPIVersion)
	}
	if cfg.GeminiStream.IsSet() {
		proc = proc.WithStream(cfg.GeminiStream.Value())
	}

	// Instructions.
	if cfg.Instructions != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	// Sampling.
	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithMaxOutputTokens(cfg.MaxTokens.Value())
	}

	// Structured output.
	if outputSchema != nil {
		proc = proc.WithResponseMIMEType("application/json")
		proc = proc.WithResponseJSONSchema(outputSchema)
	}

	return proc, nil
}

func DefaultMistralBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if len(strings.TrimSpace(getenv("MISTRAL_API_KEY"))) < 10 {
		return nil, errors.New("missing or invalid MISTRAL_API_KEY (required for Mistral provider)")
	}

	procPtr, err := textualmistral.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualmistral.Line)
	default:
		proc = proc.WithAggregateType(textualmistral.Word)
	}

	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualmistral.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualmistral.RoleAssistant)
		case "system":
			proc = proc.WithRole(textualmistral.RoleSystem)
		case "tool":
			proc = proc.WithRole(textualmistral.RoleTool)
		}
	}

	if cfg.MistralBaseURL != "" {
		proc = proc.WithBaseURL(cfg.MistralBaseURL)
	}
	if cfg.MistralStream.IsSet() {
		proc = proc.WithStream(cfg.MistralStream.Value())
	}

	if cfg.Instructions != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithMaxTokens(cfg.MaxTokens.Value())
	}

	if outputSchema != nil {
		strict := true
		proc = proc.WithResponseFormatJSONSchema(textualmistral.JSONSchemaFormat{
			Type: "json_schema",
			JSONSchema: textualmistral.JSONSchema{
				Name:   "response",
				Schema: outputSchema,
				Strict: &strict,
			},
		})
	}

	return proc, nil
}

func DefaultOllamaBuilder(
	_ context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	_ func(string) string,
	_ io.Writer,
) (textual.Processor[textual.String], error) {
	procPtr, err := textualollama.NewResponseProcessor[textual.String](model, templateStr)
	if err != nil {
		return nil, err
	}
	proc := *procPtr

	switch strings.ToLower(cfg.AggregateType) {
	case "line":
		proc = proc.WithAggregateType(textualollama.Line)
	default:
		proc = proc.WithAggregateType(textualollama.Word)
	}

	if r := strings.ToLower(cfg.Role); r != "" {
		switch r {
		case "user":
			proc = proc.WithRole(textualollama.RoleUser)
		case "assistant":
			proc = proc.WithRole(textualollama.RoleAssistant)
		case "system":
			proc = proc.WithRole(textualollama.RoleSystem)
		}
	}

	if cfg.OllamaHost != "" {
		proc = proc.WithBaseURL(cfg.OllamaHost)
	}

	switch strings.ToLower(cfg.OllamaEndpoint) {
	case "", "chat":
		proc = proc.WithChatEndpoint()
	case "generate":
		proc = proc.WithGenerateEndpoint()
	default:
		proc = proc.WithChatEndpoint()
	}

	if cfg.Instructions != "" {
		proc = proc.WithInstructions(cfg.Instructions)
	}

	if cfg.OllamaStream.IsSet() {
		proc = proc.WithStream(cfg.OllamaStream.Value())
	}

	if cfg.Temperature.IsSet() {
		proc = proc.WithTemperature(cfg.Temperature.Value())
	}
	if cfg.TopP.IsSet() {
		proc = proc.WithTopP(cfg.TopP.Value())
	}
	if cfg.MaxTokens.IsSet() {
		proc = proc.WithNumPredict(cfg.MaxTokens.Value())
	}

	if outputSchema != nil {
		proc = proc.WithFormat(outputSchema)
	}

	return proc, nil
}

// ------------------------------
// Streaming + loop
// ------------------------------

func withOptionalTimeout(parent context.Context, d OptDuration) (context.Context, context.CancelFunc) {
	if !d.IsSet() || d.Value() <= 0 {
		return parent, func() {}
	}
	return context.WithTimeout(parent, d.Value())
}

// DefaultStreamer runs one processor invocation and streams its output to outw.
//
// It expects the processor to emit aggregated "snapshots" and prints only the
// delta between successive snapshots. It returns the final accumulated text.
func DefaultStreamer(
	ctx context.Context,
	proc textual.Processor[textual.String],
	prompt string,
	outw *bufio.Writer,
) (string, error) {
	if proc == nil {
		return "", errors.New("nil processor")
	}

	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", errors.New("empty prompt")
	}

	in := make(chan textual.String, 1)
	in <- textual.String{}.FromUTF8String(prompt).WithIndex(0)
	close(in)

	out := proc.Apply(ctx, in)

	var last string
	for item := range out {
		if err := item.GetError(); err != nil {
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

func interactiveLoop(
	rootCtx context.Context,
	cfg Config,
	proc textual.Processor[textual.String],
	render renderer,
	inputSchema *jsonschema.Resolved,
	outputSchema *jsonschema.Resolved,
	stdin io.Reader,
	outw *bufio.Writer,
	stderr io.Writer,
	stream Streamer,
) int {
	if stdin == nil {
		stdin = os.Stdin
	}
	if stream == nil {
		stream = DefaultStreamer
	}

	exitCmds := make(map[string]struct{})
	for _, c := range splitCSV(cfg.ExitCommands) {
		if t := strings.TrimSpace(c); t != "" {
			exitCmds[strings.ToLower(t)] = struct{}{}
		}
	}

	inReader := bufio.NewReader(stdin)

	for {
		select {
		case <-rootCtx.Done():
			fmt.Fprintln(stderr, "\nCanceled.")
			return 1
		default:
		}

		// Prompt.
		fmt.Fprint(outw, "> ")
		_ = outw.Flush()

		line, err := inReader.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				fmt.Fprintln(outw)
				_ = outw.Flush()
				return 0
			}
			fmt.Fprintln(stderr, "Error reading stdin:", err)
			return 1
		}

		msg := strings.TrimSpace(line)
		if msg == "" {
			continue
		}
		if _, ok := exitCmds[strings.ToLower(msg)]; ok {
			return 0
		}

		// Loop input is always a raw string. Apply optional input schema validation.
		if inputSchema != nil {
			if err := validateAgainstSchema(inputSchema, msg); err != nil {
				fmt.Fprintln(stderr, "Error: input does not match --template-schema:", err)
				continue
			}
		}

		prompt, err := render(msg)
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			continue
		}

		ctx, cancel := withOptionalTimeout(rootCtx, cfg.Timeout)
		final, err := stream(ctx, proc, prompt, outw)
		cancel()
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			fmt.Fprintln(outw)
			_ = outw.Flush()
			continue
		}
		fmt.Fprintln(outw)
		_ = outw.Flush()

		if outputSchema != nil {
			v, err := parseJSONAny(final)
			if err != nil {
				fmt.Fprintln(stderr, "Error: output is not valid JSON:", err)
				continue
			}
			if err := validateAgainstSchema(outputSchema, v); err != nil {
				fmt.Fprintln(stderr, "Error: output does not match --output-schema:", err)
				continue
			}
		}
	}
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
