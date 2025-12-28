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
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/carrier"
	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/google/jsonschema-go/jsonschema"
)

// ------------------------------
// Streaming + loop
// ------------------------------

func withOptionalTimeout(parent context.Context, d OptDuration) (context.Context, context.CancelFunc) {
	if !d.IsSet() || d.Value() <= 0 {
		return parent, func() {}
	}
	return context.WithTimeout(parent, d.Value())
}

// DefaultStreamer runs one processor invocation and streams its output to outw.
//
// It expects the processor to emit aggregated "snapshots" and prints only the
// delta between successive snapshots. It returns the final accumulated text.
func DefaultStreamer(
	ctx context.Context,
	proc textual.Processor[carrier.String],
	prompt string,
	outw *bufio.Writer,
) (string, error) {
	if proc == nil {
		return "", errors.New("nil processor")
	}

	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", errors.New("empty prompt")
	}

	in := make(chan carrier.String, 1)
	in <- carrier.String{}.FromUTF8String(prompt).WithIndex(0)
	close(in)

	out := proc.Apply(ctx, in)

	var last string
	for item := range out {
		if err := item.GetError(); err != nil {
			return last, err
		}
		snapshot := item.UTF8String()
		delta := snapshot
		if strings.HasPrefix(snapshot, last) {
			delta = snapshot[len(last):]
		}
		if delta != "" {
			if _, err := outw.WriteString(delta); err != nil {
				return last, err
			}
			_ = outw.Flush()
		}
		last = snapshot
	}

	return last, nil
}

func interactiveLoop(
	rootCtx context.Context,
	cfg Config,
	proc textual.Processor[carrier.String],
	render renderer,
	inputSchema *jsonschema.Resolved,
	outputSchema *jsonschema.Resolved,
	stdin io.Reader,
	outw *bufio.Writer,
	stderr io.Writer,
	stream Streamer,
) int {
	if stdin == nil {
		stdin = os.Stdin
	}
	if stream == nil {
		stream = DefaultStreamer
	}

	exitCmds := make(map[string]struct{})
	for _, c := range splitCSV(cfg.ExitCommands) {
		if t := strings.TrimSpace(c); t != "" {
			exitCmds[strings.ToLower(t)] = struct{}{}
		}
	}

	inReader := bufio.NewReader(stdin)

	for {
		select {
		case <-rootCtx.Done():
			fmt.Fprintln(stderr, "\nCanceled.")
			return 1
		default:
		}

		// Prompt.
		fmt.Fprint(outw, "> ")
		_ = outw.Flush()

		line, err := inReader.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				fmt.Fprintln(outw)
				_ = outw.Flush()
				return 0
			}
			fmt.Fprintln(stderr, "Error reading stdin:", err)
			return 1
		}

		msg := strings.TrimSpace(line)
		if msg == "" {
			continue
		}
		if _, ok := exitCmds[strings.ToLower(msg)]; ok {
			return 0
		}

		// Loop input is always a raw string. Apply optional input schema validation.
		if inputSchema != nil {
			if err := validateAgainstSchema(inputSchema, msg); err != nil {
				fmt.Fprintln(stderr, "Error: input does not match --template-schema:", err)
				continue
			}
		}

		prompt, err := render(msg)
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			continue
		}

		ctx, cancel := withOptionalTimeout(rootCtx, cfg.Timeout)
		final, err := stream(ctx, proc, prompt, outw)
		cancel()
		if err != nil {
			fmt.Fprintln(stderr, "Error:", err)
			fmt.Fprintln(outw)
			_ = outw.Flush()
			continue
		}
		fmt.Fprintln(outw)
		_ = outw.Flush()

		if outputSchema != nil {
			v, err := parseJSONAny(final)
			if err != nil {
				fmt.Fprintln(stderr, "Error: output is not valid JSON:", err)
				continue
			}
			if err := validateAgainstSchema(outputSchema, v); err != nil {
				fmt.Fprintln(stderr, "Error: output does not match --output-schema:", err)
				continue
			}
		}
	}
}

func splitCSV(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if t := strings.TrimSpace(p); t != "" {
			out = append(out, t)
		}
	}
	return out
}
