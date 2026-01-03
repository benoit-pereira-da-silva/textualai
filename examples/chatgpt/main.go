package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/textualopenai"
)

// InputItem is the minimal "message-like" shape used in the Responses `input` array.
//
// In the API, `content` can be:
//   - a plain string, OR
//   - a structured list of content parts.
//
// We keep it as `any` so callers can provide either representation.
type InputItem struct {
	Role    string `json:"role,omitempty"`
	Content any    `json:"content,omitempty"`
}

func main() {
	var (
		modelFlag            = flag.String("model", "", "OpenAI model (overrides OPENAI_MODEL)")
		baseURLFlag          = flag.String("base-url", "", "OpenAI base URL, e.g. https://api.openai.com/v1 (overrides OPENAI_API_URL)")
		maxOutputTokensFlag  = flag.Int("max-output-tokens", 0, "Maximum output tokens (0 = omit)")
		instructionsFlag     = flag.String("instructions", "", "Optional assistant instructions (system prompt)")
		nonInteractivePrompt = flag.String("prompt", "", "If set, runs a single request and exits (otherwise starts a tiny REPL)")
		thinking             = flag.Bool("thinking", false, "If set, thinking event that separates their reasoning trace from the final answer")
		displayHeaderInfos   = flag.Bool("display-header-infos", false, "Display header infos")
	)
	flag.Parse()

	cfg, _ := textualopenai.NewConfig(*baseURLFlag, textualopenai.Model(*modelFlag))
	client := textualopenai.NewClient(cfg, context.Background())

	// Ctrl-C cancellation.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// One-shot mode.
	if strings.TrimSpace(*nonInteractivePrompt) != "" {
		if err := runOnce(ctx, client, *modelFlag, *maxOutputTokensFlag, *instructionsFlag, *thinking, *nonInteractivePrompt, *displayHeaderInfos); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	// If no -prompt was provided but args exist, treat them as a one-shot prompt.
	if argPrompt := strings.TrimSpace(strings.Join(flag.Args(), " ")); argPrompt != "" {
		if err := runOnce(ctx, client, *modelFlag, *maxOutputTokensFlag, *instructionsFlag, *thinking, argPrompt, *displayHeaderInfos); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	runRepl(ctx, client, *modelFlag, *maxOutputTokensFlag, *instructionsFlag, *thinking, *displayHeaderInfos)
}

// runRepl is a Minimal REP that keeps conversation history in memory.
func runRepl(ctx context.Context, client textualopenai.Client, model string, maxOutputTokens int, instructions string, thinking bool, displayHeaders bool) {
	_, _ = fmt.Fprintln(os.Stderr, "Enter a prompt and press Enter (Ctrl-D to quit, Ctrl-C to interrupt).")
	scanner := bufio.NewScanner(os.Stdin)
	history := make([]InputItem, 0, 16)
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
		history = append(history, InputItem{Role: "user", Content: content})
		req, err := buildRequest(ctx, textualopenai.Model(model), maxOutputTokens, instructions, thinking, history)
		if err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			return
		}
		// Stream assistant response and append it to history.
		assistantText, headerInfos, err := client.StreamAndTranscodeResponses(ctx, req)
		if err != nil {
			_, _ = fmt.Fprintln(os.Stderr, "\nerror:", err)
			continue
		}
		if displayHeaders {
			_, _ = fmt.Fprintln(os.Stdout, "\n", headerInfos.ToString())
		}
		_, _ = fmt.Fprint(os.Stdout, "\n")
		history = append(history, InputItem{Role: "assistant", Content: assistantText})
	}
}

// runOnce just runs the request once.
func runOnce(ctx context.Context, client textualopenai.Client, model string, maxOutputTokens int, instructions string, thinking bool, prompt string, displayHeaders bool) error {
	history := []InputItem{{Role: "user", Content: prompt}}
	// Build the request and add the Listeners.
	req, err := buildRequest(ctx, textualopenai.Model(model), maxOutputTokens, instructions, thinking, history)
	if err != nil {
		return err
	}
	_, headerInfos, stErr := client.StreamAndTranscodeResponses(ctx, req)
	if displayHeaders {
		_, _ = fmt.Fprintln(os.Stdout, "\n", headerInfos.ToString())
	}
	return stErr
}

// buildRequest creates and configures a textualopenai.ResponsesRequest with input data,
// optional instructions, maximum output tokens, and thinking mode. Returns the configured
// request or an error if listener addition fails.
func buildRequest(ctx context.Context, model textualopenai.Model, maxOutputTokens int, instructions string, thinking bool, history []InputItem) (*textualopenai.ResponsesRequest, error) {

	req := textualopenai.NewResponsesRequest(ctx, model)
	req.Input = history
	req.Thinking = thinking
	req.Instructions = instructions
	req.MaxOutputTokens = maxOutputTokens

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
