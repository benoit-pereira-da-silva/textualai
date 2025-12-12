# ResponseProcessor (Gemini API)

This [package](../../pkg/textualai/textualgemini) provides `ResponseProcessor`, a `textual.Processor` that turns an incoming stream of `textual.Carrier` values into requests to the **Gemini API** (Google AI for Developers) and re-emits the streamed **text output** as carrier values.

- Gemini API docs: https://ai.google.dev/gemini-api/docs
- REST Models API overview: https://ai.google.dev/gemini-api/docs/models
- Streaming endpoint reference: https://ai.google.dev/api/rest/v1beta/models/streamGenerateContent

---

## What it does

For every incoming item `S`:

1. Builds a prompt string by executing the configured Go template.
2. Sends a request to Gemini:
   - Streaming: `POST .../{model}:streamGenerateContent?alt=sse` (default)
   - Non-streaming: `POST .../{model}:generateContent`
3. Extracts text from `candidates[0].content.parts[*].text`.
4. Aggregates it into UI-friendly emissions (`word` or `line`).
5. Emits a new `S` value containing the **full accumulated output text so far**.

The processor only consumes and emits **text** output. Tool calling and other
non-text outputs are request-configurable but not streamed as `textual` output
unless you extend the processor.

---

## Construction

```go
p, err := NewResponseProcessor[textual.String](
    "gemini-2.5-flash",
    `{{.Input}}`,
)
if err != nil { /* handle */ }

// Optional options
p = p.WithInstructions("You are concise.")
p = p.WithAggregateType(Word)
p = p.WithTemperature(0.2)
p = p.WithMaxOutputTokens(512)
```

### Template data

Your template executes with:

```go
type templateData[S any] struct {
    Input  string // input.UTF8String()
    Text   string // alias of Input (kept for readability)
    Item   S      // the full carrier value
}
```

So you can use `{{.Input}}`, `{{.Text}}`, and (for rich carriers) `{{.Item}}`.

**Important**: The template must contain `{{.Input}}` (or `{{ .Input }}`), otherwise the incoming text is never injected.

---

## Authentication / environment

Set `GEMINI_API_KEY` in your environment.

The processor includes the key via the `x-goog-api-key` request header.

---

## Model names

You can pass:

- `gemini-2.5-flash` (short name) → normalized to `models/gemini-2.5-flash`
- `models/gemini-2.5-flash` (full resource)
- `tunedModels/...` (full resource for tuned models)

The processor only normalizes the common “short name” case; everything else is passed through.

---

## Streaming and aggregation

Gemini streaming returns SSE events (lines beginning with `data:`).

`AggregateType` controls emission strategy:

- `Word`: emit whenever we cross a whitespace or common punctuation boundary.
- `Line`: emit whenever we cross `\n`.

Each emitted item contains the **full output so far**, so a UI can simply render
the last item.

---

## Options coverage

`ResponseProcessor` exposes a subset of the request body in a stable, chainable API:

- `WithBaseURL(url)` (default `https://generativelanguage.googleapis.com`)
- `WithAPIVersion(v)` (default `v1beta`)
- `WithModel(model)`
- `WithStream(bool)` (streamGenerateContent vs generateContent)
- `WithRole(role)` (default: `user`)
- `WithInstructions(string)` → `systemInstruction`
- `WithSafetySettings(...)`
- `WithTools(...)`, `WithToolConfig(...)`
- `WithGenerationConfig(...)` and convenience helpers:
  - `WithTemperature`, `WithTopP`, `WithTopK`
  - `WithMaxOutputTokens`, `WithCandidateCount`
  - `WithStopSequences`
  - `WithResponseMIMEType`, `WithResponseJSONSchema`

For newly added Gemini fields, you can also set arbitrary config keys with:

- `WithGenerationConfigField(key, value)`

---

## Structured outputs (JSON schema)

Gemini supports requesting JSON output. A common strategy is:

1) Set JSON output:
- `responseMimeType = "application/json"`

2) Provide a schema:
- `responseSchema = <schema object>`

In this repository’s CLI integration, providing `--json-schema <file>` with a Gemini provider sets both.

---

## Notes

- Context cancellation is honored: the HTTP request is canceled and processing stops promptly.
- If the HTTP status is not 200, the processor returns an error containing up to 8 KiB of the response body (or the Google error message if present).
