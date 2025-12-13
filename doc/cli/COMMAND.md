# textualai

`textualai` is a small **streaming** CLI tool that can talk to multiple LLM providers with the **same command line**:

- OpenAI (Responses API)
- Claude / Anthropic (Messages API)
- Gemini (GenerateContent API)
- Mistral (Chat Completions API)
- Ollama (local HTTP API)

It is built on top of the `textual` pipeline abstractions used in this repository.

---

## Goals

- **Interoperable input**: send either a raw string or a JSON value rendered through a Go template
- **Interoperable output**: stream plain text, or request/validate **structured output** using a JSON Schema

The CLI does not attempt to preserve backward compatibility with older flag names.

---

## Provider selection

You can force the provider by prefixing the model:

```bash
textualai --model openai:gpt-4.1 ...
textualai --model claude:claude-3-5-sonnet-latest ...
textualai --model gemini:gemini-2.5-flash ...
textualai --model mistral:mistral-small-latest ...
textualai --model ollama:llama3.1 ...
```

Or set `--provider openai|claude|gemini|mistral|ollama` (or `auto`).

---

## Authentication

Set the provider API key in your environment:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export MISTRAL_API_KEY="..."
# Ollama does not require a key (defaults to http://localhost:11434)
```

---

## Input modes (one-shot)

Exactly one of these must be provided in one-shot mode:

### 1) Raw string

```bash
textualai --model openai:gpt-4.1 --message "Hello"
```

### 2) JSON value

Inline JSON:

```bash
textualai --model gemini:gemini-2.5-flash \
  --object '{"name":"Ada","lang":"Go"}'
```

From a file:

```bash
textualai --model gemini:gemini-2.5-flash \
  --object-file ./input.json
```

To read JSON from stdin:

```bash
cat ./input.json | textualai --model gemini:gemini-2.5-flash --object-file -
```

---

## Templates (Go text/template)

You can turn a JSON input into a prompt with `--template` or `--template-file`.

Template execution uses the JSON value as the root context (**dot**):

- if the input is a string → `{{.}}` is that string
- otherwise → `{{.}}` is the JSON value (usually an object)

Example:

```bash
textualai --model mistral:mistral-small-latest \
  --object '{"name":"Ada"}' \
  --template 'Write a greeting for {{.name}}.'
```

---

## Optional input validation: `--template-schema`

You can validate the input JSON before template execution:

```bash
textualai --model mistral:mistral-small-latest \
  --object-file ./input.json \
  --template-file ./prompt.tmpl \
  --template-schema ./input.schema.json
```

Validation uses `github.com/google/jsonschema-go/jsonschema`.

---

## Output: plain text stream (default)

If `--output-schema` is omitted, providers stream plain text:

```bash
textualai --model ollama:llama3.1 --message "Explain monads."
```

---

## Output: structured output with a schema

Provide `--output-schema <schema file>` to request structured output when the provider supports it, and to **validate the final output** locally.

```bash
textualai --model openai:gpt-4.1 \
  --output-schema ./schema.json \
  --message "Return JSON only."
```

### Provider mapping

- **OpenAI**: Structured Outputs via `text.format=json_schema`
- **Claude**: tool-use with `input_schema` and forced `tool_choice`; the tool input JSON is emitted
- **Gemini**: `generationConfig.responseMimeType=application/json` + `generationConfig.responseSchema`
- **Mistral**: `response_format=json_schema` (when supported by the model/API)
- **Ollama**: `format=<schema object>` (when supported by the model)

---

## Loop mode

Loop mode reads raw string lines from stdin:

```bash
textualai --model ollama:llama3.1 --loop
```

You can still use `--template`/`--template-file` in loop mode (the input is the line string).

Exit commands (default: `exit,quit,/exit,/quit`):

```bash
textualai --model ollama:llama3.1 --loop --exit-commands "bye,/bye"
```

---

## Notes on streaming semantics

Provider processors emit **snapshots** (prefixes of the full text so far).  
The CLI prints only the **delta** between successive snapshots, producing a natural stream.

---

## Build

From the repository root (example):

```bash
go build -o textualai ./textualai/pkg/textualai
```

Embed a version:

```bash
go build -o textualai -ldflags "-X github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli.version=v0.1.0" ./textualai/pkg/textualai
```
