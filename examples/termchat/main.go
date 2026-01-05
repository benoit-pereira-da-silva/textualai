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

package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/models"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualopenai"
)

type sessionOptions struct {
	Model              models.Model
	MaxOutputTokens    int
	Instructions       string
	Thinking           bool
	DisplayHeaderInfos bool
}

func main() {
	var (
		modelFlag            = flag.String("model", "", "model e.g. \"openai:gpt-4.1\" \"ollama:qwen3:32b\" (overrides TERMCHAT_MODEL)")
		baseURLFlag          = flag.String("base-url", "", "API base URL, e.g. https://api.openai.com/v1 or http://localhost:11434/v1 (overrides TERMCHAT_API_URL)")
		maxOutputTokensFlag  = flag.Int("max-output-tokens", 0, "Maximum output tokens (0 = omit)")
		instructionsFlag     = flag.String("instructions", "", "Optional assistant instructions (system prompt)")
		nonInteractivePrompt = flag.String("prompt", "", "If set, runs a single request and exits (otherwise starts a tiny REPL)")
		thinking             = flag.Bool("thinking", false, "If set, thinking mode is requested (only supported by reasoning models)")
		displayHeaderInfos   = flag.Bool("display-header-infos", false, "Display header infos")
	)
	flag.Parse()

	// Resolve model string name descriptor (flag > env > default).
	rawModel := resolveModel(*modelFlag)

	// Resolve model (best-effort).
	model, err := models.ModelFromString(rawModel)
	if err != nil {
		log.Fatal(err)
	}

	// Thinking compatibility gate: if thinking is explicitly requested, refuse early
	// for non-reasoning models (so users don't get confusing API errors).
	if *thinking && (model == nil || !model.SupportsThinking()) {
		log.Fatalf("termchat: -thinking requested but model %q does not advertise reasoning/thinking support", model.DisplayName())
	}

	client, err := textualopenai.ClientFrom(*baseURLFlag, model, context.Background())
	opts := sessionOptions{
		Model:              model,
		MaxOutputTokens:    *maxOutputTokensFlag,
		Instructions:       *instructionsFlag,
		Thinking:           *thinking,
		DisplayHeaderInfos: *displayHeaderInfos,
	}

	// Ctrl-C cancellation.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// One-shot mode.
	if strings.TrimSpace(*nonInteractivePrompt) != "" {
		if err := runOnce(ctx, client, opts, *nonInteractivePrompt); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	// If no -prompt was provided but args exist, treat them as a one-shot prompt.
	if argPrompt := strings.TrimSpace(strings.Join(flag.Args(), " ")); argPrompt != "" {
		if err := runOnce(ctx, client, opts, argPrompt); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}

	runRepl(ctx, client, opts)
}

func resolveModel(modelFlag string) string {
	if v := strings.TrimSpace(modelFlag); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("TERMCHAT_MODEL")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("OPENAI_MODEL")); v != "" {
		return v
	}
	return ""
}

// runRepl is a Minimal REP that keeps conversation history in memory.
func runRepl(ctx context.Context, client textualopenai.Client, opts sessionOptions) {
	_, _ = fmt.Fprintln(os.Stderr, "Enter a prompt and press Enter (Ctrl-D to quit, Ctrl-C to interrupt).")
	scanner := bufio.NewScanner(os.Stdin)
	history := make([]textualopenai.InputItem, 0, 16)
	for {
		select {
		case <-ctx.Done():
			_, _ = fmt.Fprintln(os.Stderr, "\ninterrupted")
			return
		default:
		}
		_, _ = fmt.Fprint(os.Stderr, "> ")
		if !scanner.Scan() {
			// EOF or error.
			if err := scanner.Err(); err != nil {
				_, _ = fmt.Fprintln(os.Stderr, "\nstdin error:", err)
			}
			return
		}
		content := strings.TrimSpace(scanner.Text())
		if content == "" {
			continue
		}

		// Add user turn.
		history = append(history, textualopenai.InputItem{Role: "user", Content: content})

		// Stream assistant response (with tool loop) and append it to history.
		assistantText, err := streamResponsesWithTools(ctx, client, opts, history)
		if err != nil {
			_, _ = fmt.Fprintln(os.Stderr, "\nerror:", err)
			continue
		}

		_, _ = fmt.Fprint(os.Stdout, "\n")
		history = append(history, textualopenai.InputItem{Role: "assistant", Content: assistantText})
	}
}

// runOnce just runs the request once (but will transparently perform extra API calls
// if the model invokes tools and needs function_call_output round-trips).
func runOnce(ctx context.Context, client textualopenai.Client, opts sessionOptions, prompt string) error {
	input := []textualopenai.InputItem{{Role: "user", Content: prompt}}
	_, err := streamResponsesWithTools(ctx, client, opts, input)
	return err
}

// streamResponsesWithTools performs a Responses request, streaming text to stdout,
// and automatically handles custom function tool calls by:
//  1. capturing tool call arguments during streaming,
//  2. executing registered handlers locally,
//  3. calling the Responses API again with function_call_output items using previous_response_id,
//  4. repeating until the model produces a response without new tool calls.
func streamResponsesWithTools(ctx context.Context, client textualopenai.Client, opts sessionOptions, initialInput any) (string, error) {
	var full strings.Builder

	input := initialInput
	previousResponseID := ""

	for {
		var responseID string
		req, err := buildRequest(ctx, opts, input, previousResponseID, &responseID)
		if err != nil {
			return full.String(), err
		}

		assistantText, headerInfos, stErr := client.StreamAndTranscodeResponses(ctx, req)
		if opts.DisplayHeaderInfos {
			_, _ = fmt.Fprintln(os.Stdout, "\n", headerInfos.ToString())
		}
		full.WriteString(assistantText)

		if stErr != nil {
			return full.String(), stErr
		}

		outs := req.FunctionCallOutputs()
		if len(outs) == 0 {
			return full.String(), nil
		}

		// We need the response id to continue the chain with previous_response_id.
		if strings.TrimSpace(responseID) == "" {
			return full.String(), fmt.Errorf("tool calls were executed but response id was not captured; cannot continue tool-calling chain")
		}

		previousResponseID = responseID
		input = outs
	}
}

// buildRequest creates and configures a textualopenai.ResponsesRequest with input data,
// optional instructions, maximum output tokens, thinking mode, and tool wiring. Returns the
// configured request or an error if listener/observer/tool registration fails.
func buildRequest(
	ctx context.Context,
	opts sessionOptions,
	input any,
	previousResponseID string,
	responseIDOut *string,
) (*textualopenai.ResponsesRequest, error) {

	req := textualopenai.NewResponsesRequest(ctx, opts.Model)
	req.Input = input
	req.Thinking = opts.Thinking
	req.Instructions = opts.Instructions
	req.MaxOutputTokens = opts.MaxOutputTokens
	req.PreviousResponseID = strings.TrimSpace(previousResponseID)

	// Register custom function tools (function calling / tool calling).
	// Only register tools when the model advertises tool support.
	if err := registerTools(req, opts); err != nil {
		return nil, err
	}

	// Capture the response id from the response.created event so we can chain calls
	// using previous_response_id when returning tool outputs.
	if responseIDOut != nil {
		obsErr := req.AddObservers(func(e textual.JsonGenericCarrier[textualopenai.StreamEvent]) {
			ev := e.Value
			if ev.Type != textualopenai.ResponseCreated {
				return
			}
			if strings.TrimSpace(*responseIDOut) != "" {
				return
			}
			if id := extractResponseIDFromCreatedEvent(ev); id != "" {
				*responseIDOut = id
			}
		}, textualopenai.ResponseCreated)
		if obsErr != nil {
			return nil, obsErr
		}
	}

	// Add listener for the event we wanna stream including error cases
	listErr := req.AddListeners(func(c textual.JsonGenericCarrier[textualopenai.StreamEvent]) textual.StringCarrier {
		str := textualopenai.StringCarrierFrom(c) // handles the normal stream delta + errors.
		if err := str.GetError(); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, "\nerror:", err)
		} else {
			_, _ = fmt.Fprint(os.Stdout, str.Value)
		}
		return str
	}, textualopenai.OutputTextDelta, textualopenai.RefusalDelta, textualopenai.RefusalDone, textualopenai.ResponseFailed, textualopenai.Error)
	if listErr != nil {
		return nil, listErr
	}

	// 	you can Add an observer for any event.
	/*
		last := time.Now()
		obsErr := req.AddObservers(func(e textual.JsonGenericCarrier[textualopenai.StreamEvent]) {
			// We can add a monitoring logic here
			ts := time.Now()
			_, _ = fmt.Fprint(os.Stderr, ts.Sub(last).Milliseconds(), "ms >>", e.Value.ToJson(), "\n")
			last = ts
		}, textualopenai.AllEvent)
		if obsErr != nil {
			return nil, obsErr
		}*/
	return req, nil
}

type responseIDEnvelope struct {
	ID string `json:"id,omitempty"`
}

func extractResponseIDFromCreatedEvent(ev textualopenai.StreamEvent) string {
	if len(ev.Response) == 0 {
		return ""
	}
	var r responseIDEnvelope
	if err := json.Unmarshal(ev.Response, &r); err != nil {
		return ""
	}
	return strings.TrimSpace(r.ID)
}

// registerTools wires custom functions into the Responses API request via function tools.
func registerTools(req *textualopenai.ResponsesRequest, opts sessionOptions) error {
	if !opts.Model.SupportsTools() {
		return nil
	}

	// JSON Schema for the get_time tool arguments.
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": `IANA time zone database name (e.g. "Europe/Paris", "America/New_York")`,
			},
		},
		"required":             []string{"location"},
		"additionalProperties": false,
	}

	// Strict mode is only enabled when supported by the selected provider.
	if opts.Model.ProviderInfo().SupportsStrictFunctionTools {
		return req.RegisterFunctionToolStrict(
			"get_time",
			"Get the current date and time in a given IANA time zone database name.",
			schema,
			true,
			getTimeHandler,
		)
	}

	return req.RegisterFunctionTool(
		"get_time",
		"Get the current date and time in a given IANA time zone database name.",
		schema,
		getTimeHandler,
	)
}

func getTimeHandler(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	// args example: {"location":"Europe/Paris"}
	var payload struct {
		Location string `json:"location"`
	}
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	if err := json.Unmarshal(args, &payload); err != nil {
		return nil, err
	}

	dt, err := get_time(payload.Location)
	if err != nil {
		return nil, err
	}

	out := map[string]any{
		"location": strings.TrimSpace(payload.Location),
		"datetime": dt,
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(b), nil
}

// get_time returns the current date/time for the provided IANA time zone database name
// (e.g. "Europe/Paris") using a comprehensive, human-friendly format.
//
// Example output:
//
//	"Saturday, 03 January 2026 14:05:06 CET (UTC+01:00) [Europe/Paris] — 2026-01-03T14:05:06+01:00"
func get_time(location string) (string, error) {
	tz := strings.TrimSpace(location)
	if tz == "" {
		return "", fmt.Errorf(`get_time: location is required (IANA time zone name like "Europe/Paris")`)
	}

	// Load the time zone (IANA time zone database name).
	loc, err := time.LoadLocation(tz)
	if err != nil {
		return "", fmt.Errorf("get_time: failed to load IANA time zone %q: %w", tz, err)
	}

	// Get the current time in that location.
	nowInLocation := time.Now().In(loc)

	// Comprehensive date format:
	// - full weekday + date + time + zone abbreviation (human-friendly)
	// - UTC offset (explicit)
	// - IANA zone name (explicit)
	// - RFC3339 timestamp (machine-friendly)
	human := nowInLocation.Format("Monday, 02 January 2006 15:04:05 MST")
	utcOffset := nowInLocation.Format("UTC-07:00")
	iso := nowInLocation.Format(time.RFC3339)

	return fmt.Sprintf("%s (%s) [%s] — %s", human, utcOffset, tz, iso), nil
}
