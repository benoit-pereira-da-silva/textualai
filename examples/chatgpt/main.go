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
		assistantText, err := streamAssistant(ctx, client, *maxOutputTokensFlag, *instructionsFlag, *thinking, history)
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

	resp, err := client.Stream(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()

	// Apply the transcoder func to the body split by SSE event.
	ioT := textual.NewIOReaderTranscoder(transcoder, resp.Body)
	ioT.SetContext(ctx)
	outCh := ioT.Start()

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

var transcoder = textual.TranscoderFunc[textual.JsonGenericCarrier[textualopenai.StreamEvent], textual.StringCarrier](
	func(ctx context.Context, in <-chan textual.JsonGenericCarrier[textualopenai.StreamEvent]) <-chan textual.StringCarrier {
		return textual.AsyncEmitter(ctx, in, func(ctx context.Context, c textual.JsonGenericCarrier[textualopenai.StreamEvent], emit func(s textual.StringCarrier)) {
			ev := c.Value
			switch ev.Type {

			// ─────────────────────────────────────────────────────
			// Lifecycle events
			// ─────────────────────────────────────────────────────

			case textualopenai.ResponseCreated:
				// Response object created; no textual payload to emit yet.

			case textualopenai.ResponseInProgress:
				// Model is generating output; informational only.

			case textualopenai.ResponseCompleted:
				// The entire response lifecycle is complete; the channel will close soon.

			case textualopenai.ResponseFailed:
				// Response failed; error details may appear elsewhere.

			// ─────────────────────────────────────────────────────
			// Text output events
			// ─────────────────────────────────────────────────────

			case textualopenai.OutputTextDelta:
				// Incremental text chunk; Text or Delta may be populated
				// depending on upstream normalization.
				emit(textual.StringFrom(ev.Text))

			case textualopenai.TextDone:
				// Text channel is complete; other response events may still follow.

			case textualopenai.OutputTextAnnotationAdded:
			// Text metadata / annotation; not part of user-visible text.

			// ReasoningSummaryTextDelta contains an incremental chunk of the model's
			// reasoning summary text (if reasoning summaries are enabled).
			// The `Delta` field will be populated.
			case textualopenai.ReasoningSummaryTextDelta:

				// ReasoningSummaryTextDone indicates that all reasoning summary text
				// has been streamed.
			case textualopenai.ReasoningSummaryTextDone:

			// ReasoningSummaryPartAdded signals that a new reasoning summary part
			// has been added (for multi-part summaries).
			case textualopenai.ReasoningSummaryPartAdded:

				// ReasoningSummaryPartDone indicates that the current reasoning summary part
				// has completed.
			case textualopenai.ReasoningSummaryPartDone:

			// ─────────────────────────────────────────────────────
			// Structured output events
			// ─────────────────────────────────────────────────────

			case textualopenai.OutputItemAdded:
				// Structured output item (tool call, block, etc.) added.

			case textualopenai.OutputItemDone:
				// Structured output item fully emitted.

			// ─────────────────────────────────────────────────────
			// Function / tool call events
			// ─────────────────────────────────────────────────────

			case textualopenai.FunctionCallArgumentsDelta:
				// Incremental function-call arguments (JsonCarrier); ignored here.

			case textualopenai.FunctionCallArgumentsDone:
				// Function-call arguments completed.

			// ─────────────────────────────────────────────────────
			// Code interpreter events
			// ─────────────────────────────────────────────────────

			case textualopenai.CodeInterpreterInProgress:
				// Code interpreter has started execution.

			case textualopenai.CodeInterpreterCallCodeDelta:
				// Incremental code being executed by the interpreter.

			case textualopenai.CodeInterpreterCallCodeDone:
				// Code emission complete.

			case textualopenai.CodeInterpreterCallInterpreting:
				// Interpreter evaluating results.

			case textualopenai.CodeInterpreterCallCompleted:
				// Interpreter execution fully completed.

			// ─────────────────────────────────────────────────────
			// File search events
			// ─────────────────────────────────────────────────────

			case textualopenai.FileSearchCallInProgress:
				// File search tool invocation started.

			case textualopenai.FileSearchCallSearching:
				// File search actively querying sources.

			case textualopenai.FileSearchCallCompleted:
				// File search completed.

			// ─────────────────────────────────────────────────────
			// Refusal & error events
			// ─────────────────────────────────────────────────────

			case textualopenai.RefusalDelta:
				// Partial refusal message; intentionally ignored.

			case textualopenai.RefusalDone:
				// Refusal message complete.

			case textualopenai.Error:
				// Stream-level error event; handled upstream or via context.

			default:
				// Unknown or future event type; safely ignored for forward compatibility.
			}
		},
		)
	},
)
