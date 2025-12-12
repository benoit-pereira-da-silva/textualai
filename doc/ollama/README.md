# textualollama.ResponseProcessor

`textualollama.ResponseProcessor` is a `textual.Processor` implementation that calls an **Ollama server** and streams back generated text as `textual.Carrier` values.

It is designed to be **API-ergonomic and consistent** with the OpenAI `ResponseProcessor` already present in this repository (`textualopenai/response_processor.go`):

- Go `text/template` prompt construction
- Stream-first processing model
- Aggregation of output into **Word** or **Line** snapshots
- Chainable `With*` configuration methods

---

## Prerequisites

- An Ollama server running (default URL: `http://localhost:11434`)
- Or set `OLLAMA_HOST` to target another server (for example `127.0.0.1:11434` or `http://127.0.0.1:11434`)

---

## Quick start (chat mode)

```go
package main

import (
  "context"
  "fmt"
  "strings"

  "github.com/benoit-pereira-da-silva/textual/pkg/textual"
  "github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualollama"
)

func main() {
  p, err := textualollama.NewResponseProcessor[textual.String](
    "llama3.1",
    "Rewrite in polite French: {{.Input}}",
  )
  if err != nil {
    panic(err)
  }

  // Optional tuning
  proc := p.
    WithChatEndpoint().
    WithAggregateType(textualollama.Word).
    WithTemperature(0.2)

  in := make(chan textual.String, 1)
  in <- textual.String{Value: "Hey, give me the file."}
  close(in)

  out := proc.Apply(context.Background(), in)
  for item := range out {
    fmt.Print(item.UTF8String())
  }
  fmt.Println()
}
```

---

## Endpoints

Ollama exposes two primary generation endpoints:

### `/api/chat` (default)

This is the default mode of `ResponseProcessor`. It sends a `messages` array and streams back `message.content` in response events.

When `Messages` is **not** explicitly set, `ResponseProcessor` builds:

1. An optional **system message** (from `WithInstructions` / `WithSystem`)
2. A single message containing the rendered template prompt (role from `WithRole`)

### `/api/generate`

Use `WithGenerateEndpoint()` to switch to the classic completion endpoint.

When `Prompt` is **not** explicitly set, `ResponseProcessor` uses the rendered template prompt as the `prompt` field.

---

## Prompt templating

The template is executed with:

```go
type templateData[S any] struct {
  Input string // input.UTF8String()
  Text  string // alias for Input
  Item  S      // full carrier value
}
```

So you can use:

- `{{.Input}}` / `{{.Text}}` for the incoming text
- `{{.Item}}` if your carrier stores metadata (e.g. `Parcel`)

---

## Streaming and aggregation

Ollama streams newline-delimited JSON objects. The processor consumes them and emits output using `AggregateType`:

- `Word`: emit snapshots when a whitespace/punctuation boundary is reached
- `Line`: emit snapshots when `\n` is reached

> Note: the current aggregation strategy emits **snapshots** (prefixes) rather than deltas.

---

## Request options (all exposed)

### Shared options (chat + generate)

- `WithModel(model string)`
- `WithStream(bool)`
- `WithFormat(any)` / `WithFormatJSON()`
- `WithOptions(*ModelOptions)` or granular helpers:
  - `WithTemperature`, `WithTopP`, `WithTopK`, `WithMinP`, `WithTypicalP`
  - `WithSeed`, `WithNumPredict`, `WithNumCtx`, `WithNumBatch`
  - `WithRepeatPenalty`, `WithRepeatLastN`, `WithPresencePenalty`, `WithFrequencyPenalty`
  - `WithStop(...)`, `WithPenalizeNewline`, `WithNumGPU`, `WithMainGPU`, `WithUseMMap`, `WithNumThread`, ...
- `WithKeepAlive(any)` (e.g. `"5m"` or `0`)
- `WithThink(bool)`
- `WithInstructions(string)` / `WithSystem(string)` (system prompt)

### Chat-only

- `WithChatEndpoint()`
- `WithMessages(...)` to override the entire `messages` list (**disables template injection**)
- `WithTools(...)` to declare tools (`type:"function"` tools supported)

### Generate-only

- `WithGenerateEndpoint()`
- `WithPrompt(string)` (**disables template injection**)
- `WithSuffix(string)`
- `WithImages(...)`
- `WithModelTemplate(string)` (Ollama `"template"` field)
- `WithRaw(bool)`
- `WithContext(...)` (deprecated by Ollama but still accepted)

---

## Structured outputs

Ollama supports `format:"json"` and JSON schema objects.

Example:

```go
proc := p.
  WithChatEndpoint().
  WithFormatJSON()
```

Or with a schema:

```go
schema := map[string]any{
  "type": "object",
  "properties": map[string]any{
    "name": map[string]any{"type": "string"},
    "age":  map[string]any{"type": "integer"},
  },
  "required": []string{"name", "age"},
}

proc := p.WithFormat(schema)
```

---

## Tool calling

Ollama can return `tool_calls` in chat responses.

This processor currently **emits only text** (`message.content` for chat, `response` for generate). Tool calls remain available in the raw JSON response, but are not emitted as textual output.

If you need tool-calling orchestration, implement a higher-level processor that:
- Parses tool calls,
- Executes tools,
- Feeds tool results back into `/api/chat`.

---

## File layout

Recommended placement in the repo:

- `textualai/pkg/textualai/textualollama/response_processor.go`
- `textualai/pkg/textualai/textualollama/README.md`
