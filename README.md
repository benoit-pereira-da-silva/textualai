# textualai

**textualai** provides streaming building blocks (processors) for LLM providers **and** a small, provider-agnostic **CLI chat tool** built on top of the [`textual`](https://github.com/benoit-pereira-da-silva/textual) pipeline abstractions.

- Providers:
    - **OpenAI** via the **Responses API** (`POST /v1/responses`)
    - **Ollama** via the **local HTTP API** (`POST /api/chat` or `POST /api/generate`)
- A single CLI (`textualai`) can talk to either provider with the **same command**.
- The CLI runtime is implemented as a reusable Go package so third parties can ship **their own CLI** by composing a custom `textual` processing graph (chain/router) while keeping the default behavior as the baseline.

![](https://github.com/benoit-pereira-da-silva/textual/blob/main/assets/logo.png)

---

## What’s in this module?

### 1) The `textualai` CLI (binary)

A small streaming chat CLI with:

- one-shot mode (`--message` / `--file-message`)
- loop / REPL mode (`--loop`)
- provider auto-selection (or explicit provider)
- Go `text/template` prompt templates (`--prompt-template`)
- structured outputs / JSON Schema (`--json-schema`) when supported

➡️ Full CLI documentation: **[`doc/cli/COMMAND.md`](./doc/cli/COMMAND.md)**

### 2) A reusable CLI runtime package

The CLI implementation lives in **`pkg/textualai/cli`** as an embeddable `Runner`:

- default behavior: `os.Exit(cli.Run(os.Args, os.Stdout, os.Stderr))`
- extension point: wrap the provider processor into your own `textual.Chain` / `textual.Router` using `cli.WithGraphComposer(...)`

➡️ Embedding & extension guide: **[`doc/cli/PACKAGE.md`](./doc/cli/PACKAGE.md)**

### 3) Provider processors

Provider backends are exposed as `textual.Processor` implementations:

- OpenAI Responses API: `pkg/textualai/textualopenai`
- Ollama HTTP API: `pkg/textualai/textualollama`

➡️ OpenAI processor docs: **[`doc/openai/README.md`](./doc/openai/README.md)**  
➡️ Ollama processor docs: **[`doc/ollama/README.md`](./doc/ollama/README.md)**

---

## Quick start

### Build the CLI

From the repository root (example paths as used in this repo):

```bash
go build -o textualai ./textualai/pkg/textualai
```

Or run directly:

```bash
go run ./textualai/pkg/textualai --help
```

### OpenAI (one-shot)

```bash
export OPENAI_API_KEY="..."
./textualai --model openai:gpt-4.1 --message "Write a haiku about terminals."
```

### Ollama (one-shot)

Make sure Ollama is running (default: `http://localhost:11434`).

```bash
./textualai --model ollama:llama3.1 --message "Explain monads in simple terms."
```

### Loop / REPL mode

```bash
./textualai --model ollama:llama3.1 --loop
```

---

## Provider selection

There are 3 ways to choose the provider:

1) **Recommended: prefix the model**

```bash
./textualai --model openai:gpt-4.1 ...
./textualai --model ollama:llama3.1 ...
```

2) Use `--provider`

```bash
./textualai --provider openai --model gpt-4.1 ...
./textualai --provider ollama --model llama3.1 ...
```

3) Default: `--provider auto` (heuristics)

- model starts with `gpt` or `o` → OpenAI
- otherwise → Ollama

You can also set a default provider with:

```bash
export TEXTUALAI_PROVIDER="ollama"
```

---

## Environment variables

- `OPENAI_API_KEY` *(required for OpenAI provider)*
- `OLLAMA_HOST` *(optional; default: `http://localhost:11434`)*
- `TEXTUALAI_PROVIDER` *(optional; `auto|openai|ollama`)*

---

## Prompt templates

`--prompt-template` points to a file containing a Go `text/template`.

**Required:** it must contain an `{{.Input}}` placeholder.

If omitted, the CLI uses the identity template:

```gotemplate
{{.Input}}
```

---

## Structured Outputs (JSON Schema)

You can request schema-constrained JSON output with:

```bash
./textualai --model openai:gpt-4.1 --json-schema ./schema.json --message "Return JSON."
```

- OpenAI: uses `text.format = {type:"json_schema", ...}`
- Ollama: uses `format = <schema object>` when supported by the model

See **[`doc/cli/COMMAND.md`](./doc/cli/COMMAND.md)** for a complete example schema.

---

## Embedding the CLI

The binary in this repo is intentionally tiny. A minimal `main.go` looks like:

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

### Extending the processing graph

To customize behavior, build a `Runner` and inject a `GraphComposer` that wraps the base provider processor into your own graph (chain/router/etc.).

Examples you can copy/paste:

- Add a pre-processing chain
- Add a local command router (e.g. intercept `/ping` without calling a provider)

➡️ See **[`doc/cli/PACKAGE.md`](./doc/cli/PACKAGE.md)**.

---

## Notes on streaming semantics

The provider processors in this repo emit **snapshots** (prefixes of the full text so far) while streaming.

The default CLI streamer prints only the **delta** between successive snapshots, so your terminal output feels natural and incremental.

If you add post-processing stages *after* the provider, those stages will receive **snapshots**, not deltas. If you need different semantics, override the streamer via `cli.WithStreamer(...)`.

---

## Build-time version

`textualai --version` prints a build-time variable defined in the `cli` package.

Example:

```bash
go build -ldflags "-X github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli.version=v1.2.3" -o textualai ./textualai/pkg/textualai
```

---

## License

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.

# Built with textual

![textual](https://github.com/benoit-pereira-da-silva/textual/blob/main/assets/logo.png)
[textual](/benoit-pereira-da-silva/textual)

## License

Licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.