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

package cli

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"text/template"
)

// ------------------------------
// Input loading + prompt rendering
// ------------------------------

func loadOneShotInput(cfg Config, stdin io.Reader) (any, error) {
	// One-shot: select exactly one input source.
	if strings.TrimSpace(cfg.Message) != "" {
		// Raw string input.
		if cfg.Object != "" || cfg.ObjectFile != "" {
			return nil, fmt.Errorf("use --message OR --object/--object-file (not both)")
		}
		return cfg.Message, nil
	}

	if cfg.Object != "" {
		v, err := parseJSONAny(cfg.Object)
		if err != nil {
			return nil, fmt.Errorf("parse --object: %w", err)
		}
		return v, nil
	}
	if cfg.ObjectFile != "" {
		raw, err := readFileOrStdin(cfg.ObjectFile, stdin)
		if err != nil {
			return nil, fmt.Errorf("read --object-file: %w", err)
		}
		v, err := parseJSONAny(raw)
		if err != nil {
			return nil, fmt.Errorf("parse --object-file JSON: %w", err)
		}
		return v, nil
	}

	return nil, nil
}

type renderer func(input any) (string, error)

func prepareRenderer(cfg Config, stdin io.Reader) (renderer, error) {
	// No template: identity for string inputs only.
	if cfg.Template == "" && cfg.TemplateFile == "" {
		return func(input any) (string, error) {
			switch v := input.(type) {
			case string:
				return strings.TrimSpace(v), nil
			default:
				return "", errors.New("non-string JSON input requires --template or --template-file")
			}
		}, nil
	}

	// Load template text.
	var tmplText string
	if cfg.TemplateFile != "" {
		b, err := readFileOrStdin(cfg.TemplateFile, stdin)
		if err != nil {
			return nil, fmt.Errorf("read template file: %w", err)
		}
		tmplText = b
	} else {
		tmplText = cfg.Template
	}

	tmplText = strings.TrimSpace(tmplText)
	if tmplText == "" {
		return nil, fmt.Errorf("template is empty")
	}

	// Parse template once and reuse.
	tmpl, err := template.New("textualai.template").Parse(tmplText)
	if err != nil {
		return nil, fmt.Errorf("parse template: %w", err)
	}

	return func(input any) (string, error) {
		var buf bytes.Buffer
		if err := tmpl.Execute(&buf, input); err != nil {
			return "", err
		}
		return strings.TrimSpace(buf.String()), nil
	}, nil
}

func readFileOrStdin(path string, stdin io.Reader) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", nil
	}
	if stdin == nil {
		stdin = os.Stdin
	}
	var r io.Reader
	if path == "-" {
		r = stdin
	} else {
		b, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return string(b), nil
	}
	b, err := io.ReadAll(r)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

func parseJSONAny(s string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	// Ensure no trailing tokens.
	if dec.More() {
		return nil, fmt.Errorf("trailing JSON tokens")
	}
	// Try reading one more token.
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("trailing JSON value")
		}
		return nil, err
	}
	return v, nil
}
