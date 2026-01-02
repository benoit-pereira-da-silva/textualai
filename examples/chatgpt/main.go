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
		maxOutputTokensFlag  = flag.Int("max-output-tokens", 256, "Maximum output tokens (0 = omit)")
		instructionsFlag     = flag.String("instructions", "", "Optional assistant instructions (system prompt)")
		nonInteractivePrompt = flag.String("prompt", "", "If set, runs a single request and exits (otherwise starts a tiny REPL)")
		thinking             = flag.Bool("thinking", false, "If set, thinking event that separates their reasoning trace from the final answer")
	)
	flag.Parse()

	cfg := textualopenai.NewConfig("", *baseURLFlag, textualopenai.Model(*modelFlag))
	client := textualopenai.NewClient(cfg, context.Background())

	// Ctrl-C cancellation.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// One-shot mode.
	if strings.TrimSpace(*nonInteractivePrompt) != "" {
		if err := runOnce(ctx, client, *maxOutputTokensFlag, *instructionsFlag, *thinking, *nonInteractivePrompt); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	// If no -prompt was provided but args exist, treat them as a one-shot prompt.
	if argPrompt := strings.TrimSpace(strings.Join(flag.Args(), " ")); argPrompt != "" {
		if err := runOnce(ctx, client, *maxOutputTokensFlag, *instructionsFlag, *thinking, argPrompt); err != nil {
			_, _ = fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	runRepl(ctx, client, *maxOutputTokensFlag, *instructionsFlag, *thinking, *nonInteractivePrompt)
}

func runRepl(ctx context.Context, client textualopenai.Client, maxOutputTokens int, instructions string, thinking bool, prompt string) {
	// Minimal REPL: keeps conversation history in memory.
	_, _ = fmt.Fprintln(os.Stderr, "textualopenai: enter a prompt and press Enter (Ctrl-D to quit, Ctrl-C to interrupt).")
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

		// Stream assistant response and append it to history.
		assistantText, err := streamAssistant(ctx, client, maxOutputTokens, instructions, thinking, history)
		if err != nil {
			_, _ = fmt.Fprintln(os.Stderr, "\nerror:", err)
			continue
		}

		_, _ = fmt.Fprint(os.Stdout, "\n")
		history = append(history, InputItem{Role: "assistant", Content: assistantText})
	}
}

func runOnce(ctx context.Context, client textualopenai.Client, maxOutputTokens int, instructions string, thinking bool, prompt string) error {
	history := []InputItem{{Role: "user", Content: prompt}}
	_, err := streamAssistant(ctx, client, maxOutputTokens, instructions, thinking, history)
	if err != nil {
		return err
	}
	_, _ = fmt.Fprint(os.Stdout, "\n")
	return nil
}

func streamAssistant(ctx context.Context, client textualopenai.Client, maxOutputTokens int, instructions string, thinking bool, history []InputItem) (string, error) {

	// Build the request and add the Listeners.
	req, err := buildRequest(ctx, maxOutputTokens, instructions, thinking, history)
	if err != nil {
		return "", err
	}

	resp, err := client.Stream(req)
	if err != nil {
		return "", err
	}
	defer func() {
		req.RemoveListeners()
		_ = resp.Body.Close()
	}()

	// Apply the transcoder func to the body split by SSE event.
	ioT := textual.NewIOReaderTranscoder[textual.JsonGenericCarrier[textualopenai.StreamEvent], textual.StringCarrier](req.Transcoder(), resp.Body)
	ioT.SetContext(ctx)
	outCh := ioT.Start()

	// To accumulate the values, we Consume the response channel
	var b strings.Builder
	for item := range outCh {
		if gErr := item.GetError(); gErr != nil {
			// Keep the stream alive, but surface errors.
			_, _ = fmt.Fprintln(os.Stderr, "\nstream error:", gErr)
			continue
		}
		b.WriteString(item.Value)
		_, _ = fmt.Fprint(os.Stdout, item.Value)
	}
	return b.String(), nil
}

func buildRequest(ctx context.Context, maxOutputTokens int, instructions string, thinking bool, history []InputItem) (*textualopenai.ResponsesRequest, error) {
	req := textualopenai.NewResponsesRequest(ctx, textual.ScanJSON)
	req.Input = history
	req.Thinking = thinking
	if instructions != "" {
		req.Instructions = instructions
	}
	if maxOutputTokens > 0 {
		mot := maxOutputTokens
		req.MaxOutputTokens = &mot
	}
	err := req.AddListener(textualopenai.OutputTextDelta, func(e textualopenai.StreamEvent) textual.StringCarrier {
		return textual.StringCarrierFrom(e.Text)
	})
	if err != nil {
		return nil, err
	}

	return req, nil
}
