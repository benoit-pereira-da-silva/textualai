# textualai CLI package

This directory contains the reusable **`cli`** package that powers the `textualai` streaming chat CLI.

The package is designed for:

- **Default behavior out of the box** (OpenAI Responses API, Mistral Chat Completions, or Ollama HTTP API).
- **Embedding** in a tiny `main` (one-liner entry point).
- **Easy extension** by wrapping the provider processor into your own **textual processing graph** (e.g. `textual.Chain` / `textual.Router`).

---

## Default usage (same behavior as the historical CLI)

### OpenAI one-shot

```bash
OPENAI_API_KEY=... textualai --model openai:gpt-4.1 --message "Write a haiku about terminals"
```

### Mistral one-shot

```bash
MISTRAL_API_KEY=... textualai --model mistral:mistral-small-latest --message "Write a haiku about terminals"
```

### Ollama one-shot

```bash
textualai --model ollama:llama3.1 --message "Explain monads"
```

### Loop mode

```bash
textualai --model ollama:llama3.1 --loop
```

---

## Minimal `main` for the default CLI

Create a tiny `main.go` (this is exactly what the repo’s binary should do):

```go
package main

import (
	"os"

	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli"
)

func main() {
	os.Exit(cli.Run(os.Args, os.Stdout, os.Stderr))
}
```

---

## Embedding with customization

If you want to build a similar CLI but with your own processing graph, build a `Runner` and inject a `GraphComposer`.

### 1) Add a pre-processing chain

This example prepends a fixed instruction to every user input before it reaches the provider.

```go
package main

import (
	"context"
	"io"
	"os"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli"
)

func main() {
	r := cli.NewRunner(
		cli.WithGraphComposer(func(ctx context.Context, cfg cli.Config, p cli.ProviderKind, base textual.Processor[textual.String], stderr io.Writer) (textual.Processor[textual.String], error) {
			// Pre-processor: rewrite the incoming message (one carrier per message).
			pre := textual.ProcessorFunc[textual.String](func(ctx context.Context, in <-chan textual.String) <-chan textual.String {
				out := make(chan textual.String)
				go func() {
					defer close(out)
					for {
						select {
						case <-ctx.Done():
							return
						case item, ok := <-in:
							if !ok {
								return
							}
							msg := strings.TrimSpace(item.UTF8String())
							msg = "You are a concise assistant.\n\n" + msg
							out <- textual.String{}.FromUTF8String(msg).WithIndex(item.GetIndex())
						}
					}
				}()
				return out
			})

			// Chain: pre -> provider
			return textual.NewChain(pre, base), nil
		}),
	)

	os.Exit(r.Run(os.Args, os.Stdout, os.Stderr))
}
```

### 2) Add a local command router

This example intercepts `/ping` locally (no provider call) and forwards everything else to the provider.

```go
package main

import (
	"context"
	"io"
	"os"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli"
)

func main() {
	r := cli.NewRunner(
		cli.WithGraphComposer(func(ctx context.Context, cfg cli.Config, p cli.ProviderKind, base textual.Processor[textual.String], stderr io.Writer) (textual.Processor[textual.String], error) {
			local := textual.ProcessorFunc[textual.String](func(ctx context.Context, in <-chan textual.String) <-chan textual.String {
				out := make(chan textual.String)
				go func() {
					defer close(out)
					for {
						select {
						case <-ctx.Done():
							return
						case item, ok := <-in:
							if !ok {
								return
							}
							// Emit a single final value (a "snapshot") so the default
							// streamer prints it once.
							out <- textual.String{}.FromUTF8String("pong").WithIndex(item.GetIndex())
						}
					}
				}()
				return out
			})

			router := textual.NewRouter[textual.String](textual.RoutingStrategyFirstMatch)
			router.AddRoute(func(ctx context.Context, item textual.String) bool {
				return strings.TrimSpace(item.UTF8String()) == "/ping"
			}, local)

			// Catch-all route: the real provider (OpenAI/Mistral/Ollama).
			router.AddProcessor(base)

			return router, nil
		}),
	)

	os.Exit(r.Run(os.Args, os.Stdout, os.Stderr))
}
```

---

## Notes on streaming semantics

The default providers in this repo emit **aggregated snapshots** (prefixes) while streaming.

The default streamer (`cli.DefaultStreamer`) prints only the **delta** between successive snapshots.

If you add post-processing *after* the provider, your post-processor will see **snapshots**, not deltas.  
If you need different semantics, override the streamer with `cli.WithStreamer(...)`.

---

## Build-time version

`--version` prints a build-time variable defined in this package:

```bash
go build -ldflags "-X github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli.version=v1.2.3" ./...
```
