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

package textualshared

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"text/template"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
)

// ResponseProcessor centralizes the shared mechanics used by provider-specific
// processors in this repository.
//
// It is meant to be embedded by provider ResponseProcessor types to avoid
// re-implementing:
//
//   - Go template prompt rendering (with TemplateData{Input/Text/Item})
//   - Standard Apply() loop semantics:
//   - ctx nil => context.Background()
//   - ctx cancellation drains upstream to avoid blocking senders
//   - any per-item processing error is attached via input.WithError(err)
//
// It intentionally does NOT implement textual.Processor by itself, because the
// provider-specific processing (request building + streaming parsing) differs
// across providers.
//
// Instead, provider processors call Apply(ctx, in, handler).
type ResponseProcessor[S textual.Carrier[S]] struct {

	// Model is the model identifier (e.g. "claude-3-5-sonnet-latest").
	Model string `json:"model,omitempty"`

	// Template is the prompt template used to build the message/prompt text.
	// It is a Go text/template executed with templateData (Input/Text/Item).
	Template template.Template `json:"template"`

	// Role is used when the processor constructs a default message.
	Role Role `json:"role,omitempty"`

	// AggregateType controls how streamed content is chunked into outputs.
	AggregateType AggregateType `json:"aggregateType"`
}

// templateData is the context passed to the shared Template execution.
// It matches the struct shape used historically across providers to avoid
// behavior changes.
type templateData[S any] struct {
	Input string // input.UTF8String()
	Text  string // alias for readability
	Item  S      // the full carrier value
}

// ParseTemplate parses a Go text/template and enforces the historical contract
// used by this project: templates must include an {{.Input}} injection point.
//
// This matches the previous per-provider behavior (string contains checks),
// including the intentionally narrow acceptance of:
//   - "{{.Input}}"
//   - "{{ .Input }}"
func ParseTemplate(name, templateStr string) (template.Template, error) {
	tmpl, err := template.New(name).Parse(templateStr)
	if err != nil {
		return template.Template{}, fmt.Errorf("parse template: %w", err)
	}

	// Ensure there is an injection point for the incoming text.
	if !strings.Contains(templateStr, "{{.Input}}") &&
		!strings.Contains(templateStr, "{{ .Input }}") {
		return template.Template{}, fmt.Errorf("template must contain an {{.Input}} placeholder")
	}

	return *tmpl, nil
}

// BuildPrompt executes the template against the incoming carrier value.
//
// Compatibility semantics:
//   - If Template is the zero value (Tree=nil), it returns input.UTF8String().
func (p ResponseProcessor[S]) BuildPrompt(input S) (string, error) {
	// Zero-valued Template has no parse tree; use the plain input text.
	if p.Template.Tree == nil {
		return input.UTF8String(), nil
	}

	var buf bytes.Buffer
	data := templateData[S]{
		Input: input.UTF8String(),
		Text:  input.UTF8String(),
		Item:  input,
	}
	if err := (&p.Template).Execute(&buf, data); err != nil {
		return "", err
	}
	return buf.String(), nil
}

// PromptHandler is the provider-specific callback invoked by the shared Apply loop
// once the prompt has been rendered.
type PromptHandler[S textual.Carrier[S]] func(
	ctx context.Context,
	input S,
	prompt string,
	out chan<- S,
) error

// Apply runs the shared Apply loop and delegates provider-specific processing
// to handler. Any error returned by BuildPrompt or handler is attached to the
// original input item via WithError(err) and emitted on out.
//
// Cancellation behavior is intentionally identical to the previous per-provider
// implementations:
//   - When ctx is done, it drains upstream `in` to avoid blocking senders.
func (p ResponseProcessor[S]) Apply(ctx context.Context, in <-chan S, handler PromptHandler[S]) <-chan S {
	if ctx == nil {
		ctx = context.Background()
	}

	out := make(chan S)

	go func() {
		defer close(out)

		for {
			select {
			case <-ctx.Done():
				// Stop processing on cancellation and drain upstream so we don't
				// block senders.
				for range in {
				}
				return

			case input, ok := <-in:
				if !ok {
					return
				}

				prompt, err := p.BuildPrompt(input)
				if err != nil {
					err = fmt.Errorf("build prompt: %w", err)
				} else if handler != nil {
					if herr := handler(ctx, input, prompt, out); herr != nil {
						err = herr
					}
				}

				if err != nil {
					// Attach the error to the item, keep stream alive.
					errRes := input.WithError(err)
					select {
					case <-ctx.Done():
						return
					case out <- errRes:
					}
				}
			}
		}
	}()

	return out
}
