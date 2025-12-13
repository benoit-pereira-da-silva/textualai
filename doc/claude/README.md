# textualclaude.ResponseProcessor

`textualclaude.ResponseProcessor` is a `textual.Processor` implementation that calls **Anthropic's Messages API** (Claude) and streams back generated text as `textual.Carrier` values.

It is designed to be **API-ergonomic and consistent** with the other providers already present in this repository:

- Go `text/template` prompt construction
- Stream-first processing model
- Aggregation of output into **Word** or **Line** snapshots
- Chainable `With*` configuration methods

> This processor targets **`POST /v1/messages`** on the Anthropic API base URL.

---

## Prerequisites

- An Anthropic API key in `ANTHROPIC_API_KEY` (required).
- Optionally override the API base URL with `WithBaseURL(...)` (default: `https://api.anthropic.com`).

---

## Quick start

```go
package main

import (
  "context"
  "fmt"

  "github.com/benoit-pereira-da-silva/textual/pkg/textual"
  "github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualclaude"
)

func main() {
  p, err := textualclaude.NewResponseProcessor[textual.String](
    "claude-3-5-sonnet-latest",
    "Rewrite in polite French: {{.Input}}",
  )
  if err != nil {
    panic(err)
  }

  // Optional tuning
  proc := p.
    WithAggregateType(textualclaude.Word).
    WithTemperature(0.2).
    WithMaxTokens(512)

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
- `{{.Item}}` if your carrier stores metadata

**Important**: the template must contain `{{.Input}}` (or `{{ .Input }}`), otherwise the incoming text is never injected.

---

## Streaming and aggregation

Anthropic streaming uses **SSE** (`data: ...` lines). The processor extracts:

- `content_block_delta` events with `text_delta` (text output)
- optionally (when enabled) `input_json_delta` (tool-use JSON streaming)

It aggregates deltas into UI-friendly emissions using `AggregateType`:

- `Word`: emit snapshots when a whitespace/punctuation boundary is reached
- `Line`: emit snapshots when `\n` is reached

> Note: the aggregation strategy emits **snapshots** (prefixes) rather than deltas. The CLI’s default streamer prints only the delta between successive snapshots.

---

## Request options (exposed)

This processor exposes the most commonly used request fields for `/v1/messages` via chainable methods:

### Core

- `WithModel(model string)`
- `WithBaseURL(url string)` (default: `https://api.anthropic.com`)
- `WithAPIVersion(version string)` (default: `2023-06-01`)
- `WithStream(bool)` (default: `true`)

### Prompt / messages

- `WithInstructions(string)` / `WithSystem(string)`:
  - sets the request `system` prompt.
- `WithMessages(...)`:
  - overrides the entire `messages` list (**disables template injection**).

### Sampling / output control

- `WithMaxTokens(int)` (maps to `max_tokens`)
- `WithTemperature(float64)`
- `WithTopP(float64)`
- `WithTopK(int)`
- `WithStopSequences(...string)`

### Tools (function calling)

- `WithTools(...)`
- `WithToolChoice(any)` (string or object)
- `WithEmitToolUse(bool)` to also emit tool-use JSON in the textual output stream.

---

## Structured outputs (JSON schema)

Anthropic does not provide an OpenAI-style "JSON schema output" field on the Messages API.

A practical strategy (used by the CLI integration in this repository) is to:

1) Declare a tool whose `input_schema` is your desired JSON schema.
2) Force the model to call that tool.

Then the tool-use block’s `input` will match your schema.

In the CLI:

```bash
export ANTHROPIC_API_KEY="..."
./textualai --model claude:claude-3-5-sonnet-latest --json-schema ./schema.json --message "Return JSON only."
```

---

## Notes

- This processor only emits **text** output derived from streamed text deltas (and optionally tool-use JSON deltas if enabled).
- Tool execution orchestration is not performed here. If you need to execute tools:
  1. detect tool-use blocks,
  2. run your tool,
  3. feed the tool result back to the model using the Messages API tool result mechanism.
