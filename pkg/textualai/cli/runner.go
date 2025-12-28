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
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/benoit-pereira-da-silva/textual/pkg/carrier"
	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/google/jsonschema-go/jsonschema"
)

// ProviderBuilder builds a provider-specific Processor based on the CLI config.
type ProviderBuilder func(
	ctx context.Context,
	cfg Config,
	model string,
	templateStr string,
	outputSchema map[string]any,
	getenv func(string) string,
	stderr io.Writer,
) (textual.Processor[carrier.String], error)

// Streamer runs one prompt through the processor and streams the output to
// outw. It returns the final accumulated text (last snapshot).
type Streamer func(
	ctx context.Context,
	proc textual.Processor[carrier.String],
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
	var proc textual.Processor[carrier.String]
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
