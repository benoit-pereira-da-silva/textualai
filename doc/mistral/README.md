# textualmistral.ResponseProcessor

`textualmistral.ResponseProcessor` is a `textual.Processor` implementation that calls **Mistral AI’s Chat Completions API** and streams back generated text as `textual.Carrier` values.

It is designed to be **API-ergonomic and consistent** with the other providers already present in this repository:

- Go `text/template` prompt construction
- Stream-first processing model
- Aggregation of output into **Word** or **Line** snapshots
- Chainable `With*` configuration methods

> This processor targets **`POST /v1/chat/completions`** on the Mistral API base URL.

---

## Prerequisites

- A Mistral API key in `MISTRAL_API_KEY` (required).
- Optionally set `MISTRAL_BASE_URL` to override the API base URL (default: `https://api.mistral.ai`).

---

## Quick start

```go
package main

import (
  "context"
  "fmt"

  "github.com/benoit-pereira-da-silva/textual/pkg/textual"
  "github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualmistral"
)

func main() {
  p, err := textualmistral.NewResponseProcessor[textual.String](
    "mistral-small-latest",
    "Rewrite in polite French: {{.Input}}",
  )
  if err != nil {
    panic(err)
  }

  // Optional tuning
  proc := p.
    WithAggregateType(textualmistral.Word).
    WithTemperature(0.2).
    WithTopP(0.95)

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

**Important**: the template must contain `{{.Input}}` (or `{{ .Input }}`), otherwise the incoming text is never injected.

---

## Streaming and aggregation

Mistral streaming uses **SSE** (`data: ...` lines).

The processor aggregates streamed deltas into UI-friendly emissions using `AggregateType`:

- `Word`: emit snapshots when a whitespace/punctuation boundary is reached
- `Line`: emit snapshots when `\n` is reached

> Note: the current aggregation strategy emits **snapshots** (prefixes) rather than deltas. The CLI’s default streamer prints only the delta between successive snapshots.

---

## Request options (exposed)

This processor exposes the most commonly used request fields for `/v1/chat/completions` via chainable methods:

### Core

- `WithModel(model string)`
- `WithBaseURL(url string)` (defaults to `MISTRAL_BASE_URL` or `https://api.mistral.ai`)
- `WithStream(bool)` (default `true`)

### Prompt / messages

- `WithInstructions(string)` / `WithSystem(string)`:
  - When `Messages` is not explicitly set, `Instructions` is inserted as the first **system** message.
- `WithMessages(...)`:
  - Overrides the entire `messages` list (**disables template injection**).

### Sampling / output control

- `WithTemperature(float64)`
- `WithTopP(float64)`
- `WithMaxTokens(int)`
- `WithStop(...string)`

### Safety / determinism

- `WithSafePrompt(bool)`
- `WithRandomSeed(int)`

### Tools

- `WithTools(...)`
- `WithToolChoice(any)`
- `WithParallelToolCalls(bool)`

### Response formatting / Structured Outputs

- `WithResponseFormatText()`
- `WithResponseFormatJSONObject()`
- `WithResponseFormatJSONSchema(JSONSchemaFormat)`

Example JSON schema request:

```go
schema := map[string]any{
  "type": "object",
  "properties": map[string]any{
    "answer": map[string]any{"type": "string"},
    "confidence": map[string]any{"type": "number"},
  },
  "required": []any{"answer", "confidence"},
  "additionalProperties": false,
}

strict := true

proc := p.WithResponseFormatJSONSchema(textualmistral.JSONSchemaFormat{
  Type: "json_schema",
  JSONSchema: textualmistral.JSONSchema{
    Name: "response",
    Schema: schema,
    Strict: &strict,
  },
})
```

---

## Notes

- This processor only emits **text** output derived from `choices[0]` deltas (or `choices[0].message` when `stream=false`).
- Tool calls may be present in responses (under `tool_calls`), but this processor does not orchestrate tool execution.
  If you need tool execution, implement a higher-level processor that:
  1. Detects tool calls
  2. Executes the tools
  3. Sends tool results back as `role:"tool"` messages

---

## File layout

Recommended placement in the repo:

- `textualai/pkg/textualai/textualmistral/response_processor.go`
- `textualai/doc/mistral/README.md`
