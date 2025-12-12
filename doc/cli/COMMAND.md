# textualai

`textualai` is a small **streaming** CLI chat tool that can talk to **OpenAI** (Responses API), **Gemini** (GenerateContent API), **Mistral** (Chat Completions API), or **Ollama** (local HTTP API) with the **same command**.

It is built on top of the `textual` pipeline abstractions used in this repository.

---

## Features

- **One-shot** mode (`--message` / `--file-message`)
- **Loop / REPL** mode (`--loop`)
- **Provider-agnostic flags** (temperature, top-p, max tokens, instructions, templates…)
- **Prompt templates** (`--prompt-template`) using Go `text/template`
- **Structured Outputs** (JSON Schema):
  - OpenAI: `text.format = {type:"json_schema", ...}`
  - Gemini: `generationConfig.responseMimeType="application/json"` + `generationConfig.responseSchema=<schema>`
  - Mistral: `response_format = {type:"json_schema", ...}` (when supported by the model)
  - Ollama: `format = <schema object>` (when supported by your Ollama model)

---

## Installation / Build

From the repository root (example):

```bash
go build -o textualai ./textualai/pkg/textualai
```

Or run directly:

```bash
go run ./textualai/pkg/textualai --help
```

You can embed a version at build time:

```bash
go build -o textualai -ldflags "-X github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli.version=v0.1.0" ./textualai/pkg/textualai
```

---

## Quick start

### OpenAI (one-shot)

```bash
export OPENAI_API_KEY="..."
./textualai --model openai:gpt-4.1 --message "Write a haiku about terminals."
```

### Gemini (one-shot)

```bash
export GEMINI_API_KEY="..."
./textualai --model gemini:gemini-2.5-flash --message "Write a haiku about terminals."
```

### Mistral (one-shot)

```bash
export MISTRAL_API_KEY="..."
./textualai --model mistral:mistral-small-latest --message "Write a haiku about terminals."
```

### Ollama (one-shot)

Make sure Ollama is running (default: `http://localhost:11434`).

```bash
./textualai --model ollama:llama3.1 --message "Explain monads in simple terms."
```

If your Ollama server is elsewhere:

```bash
export OLLAMA_HOST="http://127.0.0.1:11434"
./textualai --model ollama:llama3.1 --message "Hello from another host."
```

---

## Help

```bash
./textualai help
# or
./textualai --help
```

---

## Provider selection

There are 3 ways to choose the provider:

1) **Recommended: prefix the model**

```bash
./textualai --model openai:gpt-4.1 ...
./textualai --model gemini:gemini-2.5-flash ...
./textualai --model mistral:mistral-small-latest ...
./textualai --model ollama:llama3.1 ...
```

2) Use `--provider`

```bash
./textualai --provider openai --model gpt-4.1 ...
./textualai --provider gemini --model gemini-2.5-flash ...
./textualai --provider mistral --model mistral-small-latest ...
./textualai --provider ollama --model llama3.1 ...
```

3) Default: `--provider auto` (heuristics)

- model starts with `gpt` or `o` → OpenAI
- model starts with `gemini` → Gemini
- model starts with `mistral`, `codestral`, `ministral`, `devstral`, `magistral` → Mistral
- otherwise → Ollama

You can also set a default provider:

```bash
export TEXTUALAI_PROVIDER="gemini"
```

---

## Prompt templates

`--prompt-template` points to a file containing a Go `text/template`.

**Required:** it must contain an `{{.Input}}` placeholder.

### Example template

`prompt.tmpl`:

```gotemplate
You are a helpful assistant.

User message:
{{.Input}}
```

Run:

```bash
./textualai --model openai:gpt-4.1 --prompt-template ./prompt.tmpl --message "Hello!"
```

If `--prompt-template` is omitted, a safe default is used:

```gotemplate
{{.Input}}
```

---

## Messages from a file

```bash
./textualai --model ollama:llama3.1 --file-message ./question.txt
```

Read from stdin:

```bash
cat ./question.txt | ./textualai --model openai:gpt-4.1 --file-message -
```

If your file isn’t UTF‑8, you can specify an encoding:

```bash
./textualai --model openai:gpt-4.1 --file-message ./latin1.txt --file-encoding ISO-8859-1
```

Supported encoding names come from `textual.ParseEncoding(...)`.

---

## Loop / REPL mode

```bash
./textualai --model ollama:llama3.1 --loop
```

Exit commands in loop mode:

- `exit`
- `quit`
- `/exit`
- `/quit`

You can override them:

```bash
./textualai --model ollama:llama3.1 --loop --exit-commands "bye,/bye"
```

---

## Structured Outputs (JSON Schema)

Create `schema.json`:

```json
{
  "type": "object",
  "properties": {
    "answer": { "type": "string" },
    "confidence": { "type": "number" }
  },
  "required": ["answer", "confidence"],
  "additionalProperties": false
}
```

### OpenAI JSON Schema

```bash
export OPENAI_API_KEY="..."
./textualai --model openai:gpt-4.1   --json-schema ./schema.json   --message "What is the capital of France? Return JSON."
```

### Gemini JSON Schema

When `--json-schema` is provided with the Gemini provider, the CLI sets:

- `generationConfig.responseMimeType = "application/json"` (unless you explicitly override it)
- `generationConfig.responseSchema = <schema object>`

```bash
export GEMINI_API_KEY="..."
./textualai --model gemini:gemini-2.5-flash   --json-schema ./schema.json   --message "What is the capital of France? Return JSON."
```

### Gemini JSON output (no schema)

Some use-cases only need “JSON mode” without schema validation. You can request JSON by setting the response MIME type:

```bash
export GEMINI_API_KEY="..."
./textualai --model gemini:gemini-2.5-flash   --gemini-response-mime-type application/json   --message "Return JSON with keys: a, b, c."
```

### Mistral JSON Schema

```bash
export MISTRAL_API_KEY="..."
./textualai --model mistral:mistral-small-latest   --json-schema ./schema.json   --message "What is the capital of France? Return JSON."
```

### Mistral JSON output (no schema)

Some models support requesting JSON output without a schema:

```bash
export MISTRAL_API_KEY="..."
./textualai --model mistral:mistral-small-latest   --mistral-response-format json_object   --message "Return JSON with keys: a, b, c."
```

### Ollama JSON output

Some models support requesting JSON output:

```bash
./textualai --model ollama:llama3.1 --ollama-format json --message "Return JSON with keys: a, b, c."
```

To request schema-based output (when supported):

```bash
./textualai --model ollama:llama3.1 --json-schema ./schema.json --message "Return JSON."
```

---

## Interesting flags

### Common flags

- `--instructions <text>`: system/developer instruction
- `--temperature <float>`
- `--top-p <float>`
- `--max-tokens <int>`
- `--timeout <duration>` (e.g. `30s`, `2m`)
- `--aggregate word|line`: how streaming snapshots are chunked

### OpenAI-only

- `--openai-service-tier auto|default|flex|priority`
- `--openai-truncation auto|disabled`
- `--openai-store[=bool]`
- `--openai-prompt-cache-key <key>`
- `--openai-prompt-cache-retention <dur>` (e.g. `24h`)
- `--openai-safety-identifier <id>`
- `--openai-metadata key=value` (repeatable)
- `--openai-include <csv>`

### Gemini-only

- `--gemini-base-url <url>` (default: `https://generativelanguage.googleapis.com`)
- `--gemini-api-version <v>` (default: `v1beta`)
- `--gemini-stream[=bool]`
- `--gemini-top-k <int>`
- `--gemini-stop <csv>`
- `--gemini-candidate-count <int>`
- `--gemini-response-mime-type <type>` (e.g. `application/json`)

### Mistral-only

- `--mistral-base-url <url>` (overrides `MISTRAL_BASE_URL`)
- `--mistral-stream[=bool]`
- `--mistral-safe-prompt[=bool]`
- `--mistral-random-seed <int>`
- `--mistral-prompt-mode <mode>` (e.g. `reasoning`)
- `--mistral-parallel-tool-calls[=bool]`
- `--mistral-frequency-penalty <float>`
- `--mistral-presence-penalty <float>`
- `--mistral-stop <csv>`
- `--mistral-n <int>`
- `--mistral-response-format <type>` (`text` or `json_object`)

### Ollama-only

- `--ollama-host <url>` (overrides `OLLAMA_HOST`)
- `--ollama-endpoint chat|generate`
- `--ollama-keep-alive <v>` (e.g. `5m` or `0`)
- `--ollama-think[=bool]`
- `--ollama-stream[=bool]`
- `--ollama-top-k <int>`
- `--ollama-num-ctx <int>`
- `--ollama-seed <int>`
- `--ollama-stop <csv>`
- `--ollama-option key=value` (repeatable; parses JSON/bool/int/float/string)

---

## Notes

### Streaming snapshots vs deltas

The underlying processors emit **snapshots** (prefixes of the full text so far).
The CLI prints only the **delta** between successive snapshots, so your terminal
shows a natural streaming experience.

### Conversation memory

This CLI currently sends a single message per request. It does not keep a full
conversation history between turns. If you want “stateful chat”, you can add a
history buffer and pass it through provider-specific message inputs (future work).

---

## License

Apache-2.0 (see repository license).
