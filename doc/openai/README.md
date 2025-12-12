# ResponseProcessor (OpenAI Responses API)

This [package](../../pkg/textualai/textualopenai) provides `ResponseProcessor`, a `textual.Processor` that turns an incoming stream of `textual.Carrier` values into requests to **OpenAI’s Responses API** (`POST /v1/responses`) and re-emits the streamed **output text** as carrier values.

It is designed as a drop-in building block for the `textual` pipeline types (`Chain`, `Router`, `IOReaderProcessor`, …).

- API reference: https://platform.openai.com/docs/api-reference/responses
- Streaming events reference: https://platform.openai.com/docs/api-reference/responses-streaming/response  

---

## What it does

For every incoming item `S`:

1. Builds a prompt string by executing the configured Go template.
2. Sends a streaming `POST https://api.openai.com/v1/responses`.
3. Parses SSE events and extracts **`response.output_text.delta`** chunks.
4. Aggregates deltas into UI-friendly emissions (`word` or `line`).
5. Emits a new `S` value containing the **full accumulated output text so far**.

The processor only consumes and emits **text** output (`response.output_text.*` events). Tool calls are supported by request options (e.g. `tools`, `tool_choice`), but their streaming events are ignored by this processor unless you extend it.

---

## Construction

```go
p, err := NewResponseProcessor[textual.String](
    "gpt-5-mini",
    `{{.Input}}`,
)
if err != nil { /* handle */ }

// Optional options
p = p.WithInstructions("You are concise.")
p = p.WithRole(RoleUser)
p = p.WithAggregateType(Word)
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

## Emission strategy: AggregateType

- `Word`: emit whenever we cross a whitespace or common punctuation boundary.
- `Line`: emit whenever we cross `\n`.

Each emitted item contains the **full output so far**, so a UI can simply render the last item.

---

## Options coverage (Responses API request body)

`ResponseProcessor` exposes (as fields + chainable `With*` methods) all request-body parameters shown in the Responses API reference, including:

- `background`, `conversation`, `include`, `input`, `instructions`
- `max_output_tokens`, `max_tool_calls`, `metadata`, `model`
- `parallel_tool_calls`, `previous_response_id`, `prompt`
- `prompt_cache_key`, `prompt_cache_retention`
- `reasoning`, `safety_identifier`, `service_tier`
- `store`, `stream`, `stream_options`
- `temperature`, `text`, `tool_choice`, `tools`
- `top_logprobs`, `top_p`, `truncation`, `user` (deprecated)

See: request body section in the API reference.

---

## Notes

- The processor always uses `stream: true` unless you override it (not recommended for this processor).
- If the HTTP status is not 200, the processor returns an error containing up to 4 KiB of the response body (for debugging).
- Context cancellation is honored: the request is canceled and processing stops promptly.

