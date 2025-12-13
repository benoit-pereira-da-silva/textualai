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

// Package cli implements the `textualai` command-line interface runtime.
//
// textualai is a small streaming CLI that can target multiple providers with a
// single command line.
//
// Interoperability goals:
//
//   - Input: send either a raw string (--message) or a JSON value (--object /
//     --object-file). When using a template (--template / --template-file), the
//     JSON value becomes the Go template root ({{.}}). If the value is a string,
//     {{.}} is that string; otherwise {{.}} is the parsed JSON value.
//
//   - Output: stream plain text by default, or request and validate schema-based
//     structured output with --output-schema.
//
// Schema validation (input and output) uses github.com/google/jsonschema-go/jsonschema.
package cli

import "io"

// version is set at build time using:
//
//	go build -ldflags "-X github.com/benoit-pereira-da-silva/textualai/pkg/textualai/cli.version=v1.2.3"
//
// When not set, it defaults to "dev".
var version = "dev"

// Version returns the build-time version string printed by --version.
func Version() string { return version }

// Run is the default textualai CLI entry point.
//
// It is intended to be called from a tiny main:
//
//	func main() { os.Exit(cli.Run(os.Args, os.Stdout, os.Stderr)) }
func Run(argv []string, stdout io.Writer, stderr io.Writer) int {
	return NewRunner().Run(argv, stdout, stderr)
}

// PrintUsage prints the CLI usage help.
func PrintUsage(w io.Writer) {
	printUsage(w)
}
