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
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

// ------------------------------
// Schema loading / validation
// ------------------------------

// CompiledSchema holds a JSON Schema in two forms:
//   - SchemaMap: for passing into provider request options
//   - Resolved: for local validation via Resolved.Validate
type CompiledSchema struct {
	Path      string
	SchemaMap map[string]any
	Resolved  *jsonschema.Resolved
}

// loadCompiledSchema reads a JSON schema file (or stdin when path is "-"),
// parses it into map form (for provider requests) and into jsonschema.Schema,
// resolves refs, and returns a CompiledSchema.
//
// For schema files on disk, it sets ResolveOptions.BaseURI to an absolute file:// URI
// so that relative $ref paths can be resolved.
func loadCompiledSchema(path string, stdin io.Reader) (*CompiledSchema, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, nil
	}

	raw, err := readFileOrStdin(path, stdin)
	if err != nil {
		return nil, err
	}

	// Provider schema map.
	var schemaMap map[string]any
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&schemaMap); err != nil {
		return nil, fmt.Errorf("parse schema JSON: %w", err)
	}
	if len(schemaMap) == 0 {
		return nil, fmt.Errorf("schema is empty")
	}

	// jsonschema.Schema for compilation/validation.
	var s jsonschema.Schema
	dec2 := json.NewDecoder(strings.NewReader(raw))
	dec2.UseNumber()
	if err := dec2.Decode(&s); err != nil {
		return nil, fmt.Errorf("parse schema (typed): %w", err)
	}

	// Resolve.
	opts := &jsonschema.ResolveOptions{
		ValidateDefaults: true,
	}
	if path != "-" {
		baseURI, baseDir, err := fileURIAndDir(path)
		if err != nil {
			return nil, err
		}
		opts.BaseURI = baseURI
		opts.Loader = fileSchemaLoader(baseDir, stdin)
	}

	resolved, err := s.Resolve(opts)
	if err != nil {
		return nil, err
	}

	return &CompiledSchema{
		Path:      path,
		SchemaMap: schemaMap,
		Resolved:  resolved,
	}, nil
}

func fileURIAndDir(path string) (baseURI string, baseDir string, err error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", "", fmt.Errorf("abs path %q: %w", path, err)
	}
	u := url.URL{Scheme: "file", Path: abs}
	return u.String(), filepath.Dir(abs), nil
}

// fileSchemaLoader loads schemas for remote references (anything not under the
// root schema). In practice, this supports file:// URIs (and bare/relative
// paths as a fallback).
//
// If you want to allow http(s) refs, implement a different Loader (or wrap this
// one) and pass it into Schema.Resolve.
func fileSchemaLoader(baseDir string, stdin io.Reader) jsonschema.Loader {
	return func(uri *url.URL) (*jsonschema.Schema, error) {
		if uri == nil {
			return nil, fmt.Errorf("nil schema URI")
		}

		// Ignore fragments when reading the file; fragment resolution is handled
		// by the jsonschema resolver.
		u := *uri
		u.Fragment = ""

		switch u.Scheme {
		case "", "file":
			p := u.Path
			// Fallback: if we got a relative path with an empty scheme, treat it as file under baseDir.
			if p == "" {
				p = u.String()
			}
			if p == "" {
				return nil, fmt.Errorf("empty schema path for URI %q", uri.String())
			}

			// If it's not absolute (possible with empty scheme), anchor it to baseDir.
			if !filepath.IsAbs(p) && strings.TrimSpace(baseDir) != "" {
				p = filepath.Join(baseDir, p)
			}

			// Special case: allow "-" only when the URI explicitly says so (rare).
			raw, err := readFileOrStdin(p, stdin)
			if err != nil {
				return nil, err
			}

			var s jsonschema.Schema
			dec := json.NewDecoder(strings.NewReader(raw))
			dec.UseNumber()
			if err := dec.Decode(&s); err != nil {
				return nil, fmt.Errorf("parse referenced schema %q: %w", uri.String(), err)
			}
			return &s, nil
		default:
			return nil, fmt.Errorf("unsupported schema URI scheme %q (uri=%q)", u.Scheme, uri.String())
		}
	}
}

func validateAgainstSchema(res *jsonschema.Resolved, v any) error {
	if res == nil {
		return nil
	}
	return res.Validate(v)
}
