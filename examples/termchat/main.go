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
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualopenai"
)

func main() {
	var (
		modelFlag            = flag.String("model", "", "model e.g. \"openai:gpt-4.1\" \"ollama:qwen3:32b\" (overrides TEXTUALAI_MODEL)")
		baseURLFlag          = flag.String("base-url", "", "API base URL, e.g. https://api.openai.com/v1 or http://localhost:11434/v1 (overrides TEXTUALAI_API_URL)")
		maxOutputTokensFlag  = flag.Int("max-output-tokens", 0, "Maximum output tokens (0 = omit)")
		instructionsFlag     = flag.String("instructions", "", "Optional assistant instructions (system prompt)")
		nonInteractivePrompt = flag.String("prompt", "", "If set, runs a single request and exits (otherwise starts a tiny REPL)")
		thinking             = flag.Bool("thinking", false, "If set, thinking event that separates their reasoning trace from the final answer")
		displayHeaderInfos   = flag.Bool("display-header-infos", false, "Display header infos")
	)
	flag.Parse()

	// Resolve the model the same way textualopenai.NewConfig does, so the CLI works
	// when -model is omitted and OPENAI_MODEL is set (or absent).
	resolvedModel := strings.TrimSpace(*modelFlag)
	if resolvedModel == "" {
		resolvedModel = strings.TrimSpace(os.Getenv("OPENAI_MODEL"))
		if resolvedModel == "" {
			resolvedModel = string(textualopenai.ModelGpt41)
		}
	}

	cfg, err := textualopenai.NewConfig(*baseURLFlag, textualopenai.Model(*modelFlag))
	if err != nil {
		log.Fatal(err)
	}
	client := textualopenai.NewClient(cfg, context.Background())

	// Ctrl-C cancellation.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// One-shot mode.
	if strings.TrimSpace(*nonInteractivePrompt) != "" {
		if err := runOnce(ctx, client, resolvedModel, *maxOutputTokensFlag, *instructionsFlag, *thinking, *nonInteractivePrompt, *displayHeaderInfos); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	// If no -prompt was provided but args exist, treat them as a one-shot prompt.
	if argPrompt := strings.TrimSpace(strings.Join(flag.Args(), " ")); argPrompt != "" {
		if err := runOnce(ctx, client, resolvedModel, *maxOutputTokensFlag, *instructionsFlag, *thinking, argPrompt, *displayHeaderInfos); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	runRepl(ctx, client, resolvedModel, *maxOutputTokensFlag, *instructionsFlag, *thinking, *displayHeaderInfos)
}

// runRepl is a Minimal REP that keeps conversation history in memory.
func runRepl(ctx context.Context, client textualopenai.Client, model string, maxOutputTokens int, instructions string, thinking bool, displayHeaders bool) {
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
		assistantText, err := streamResponsesWithTools(ctx, client, textualopenai.Model(model), maxOutputTokens, instructions, thinking, history, displayHeaders)
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
func runOnce(ctx context.Context, client textualopenai.Client, model string, maxOutputTokens int, instructions string, thinking bool, prompt string, displayHeaders bool) error {
	input := []textualopenai.InputItem{{Role: "user", Content: prompt}}
	_, err := streamResponsesWithTools(ctx, client, textualopenai.Model(model), maxOutputTokens, instructions, thinking, input, displayHeaders)
	return err
}

// streamResponsesWithTools performs a Responses request, streaming text to stdout,
// and automatically handles custom function tool calls by:
//  1. capturing tool call arguments during streaming,
//  2. executing registered handlers locally,
//  3. calling the Responses API again with function_call_output items using previous_response_id,
//  4. repeating until the model produces a response without new tool calls.
func streamResponsesWithTools(ctx context.Context, client textualopenai.Client, model textualopenai.Model, maxOutputTokens int, instructions string, thinking bool, initialInput any, displayHeaders bool) (string, error) {
	var full strings.Builder

	input := initialInput
	previousResponseID := ""

	for {
		var responseID string
		req, err := buildRequest(ctx, model, maxOutputTokens, instructions, thinking, input, previousResponseID, &responseID)
		if err != nil {
			return full.String(), err
		}

		assistantText, headerInfos, stErr := client.StreamAndTranscodeResponses(ctx, req)
		if displayHeaders {
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
	model textualopenai.Model,
	maxOutputTokens int,
	instructions string,
	thinking bool,
	input any,
	previousResponseID string,
	responseIDOut *string,
) (*textualopenai.ResponsesRequest, error) {

	req := textualopenai.NewResponsesRequest(ctx, model)
	req.Input = input
	req.Thinking = thinking
	req.Instructions = instructions
	req.MaxOutputTokens = maxOutputTokens
	req.PreviousResponseID = strings.TrimSpace(previousResponseID)

	// Register custom function tools (function calling / tool calling).
	if err := registerTools(req); err != nil {
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
func registerTools(req *textualopenai.ResponsesRequest) error {

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

	// Strict mode helps ensure args conform to the schema when supported by the model.
	return req.RegisterFunctionToolStrict(
		"get_time",
		"Get the current date and time in a given IANA time zone database name.",
		schema,
		true,
		func(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
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
		},
	)
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
