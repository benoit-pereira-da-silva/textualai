package textualopenai

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"sync"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
)

// ResponsesRequest
// https://platform.openai.com/docs/api-reference/responses
type ResponsesRequest struct {

	// Main fields for basic support.
	//
	// We intentionally keep this textual so we exclude multimodal support.
	// - tool
	// - no function calling.
	//
	// You can still support the conversation state by passing a full history in Input.
	Model        Model  `json:"model,omitempty"`
	Input        any    `json:"input,omitempty"`
	Stream       bool   `json:"stream,omitempty"`
	Instructions string `json:"instructions,omitempty"`

	// Limit the output
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	// If set to true when using a reasoning model, we get intermediary thinking summaries.
	Thinking bool `json:"thinking,omitempty"`

	// Non serializable
	ctx       context.Context
	splitFunc bufio.SplitFunc

	// Listeners
	mu        sync.Mutex
	listeners map[StreamEventType]func(e textual.JsonGenericCarrier[StreamEvent]) textual.StringCarrier
	observers map[StreamEventType]func(e textual.JsonGenericCarrier[StreamEvent])
}

func NewResponsesRequest(ctx context.Context, f bufio.SplitFunc, model Model) *ResponsesRequest {
	return &ResponsesRequest{
		ctx:             ctx,
		splitFunc:       f,
		Model:           model,
		Input:           nil,
		Stream:          true,
		MaxOutputTokens: nil,
		Thinking:        false,
		listeners:       make(map[StreamEventType]func(e textual.JsonGenericCarrier[StreamEvent]) textual.StringCarrier),
		observers:       make(map[StreamEventType]func(e textual.JsonGenericCarrier[StreamEvent])),
	}
}
func (r *ResponsesRequest) Context() context.Context {
	return r.ctx
}

func (r *ResponsesRequest) URL(baseURL string) (string, error) {
	if strings.TrimSpace(baseURL) == "" {
		return "", errors.New("textualopenai: missing OpenAI base URL")
	}
	u, err := url.Parse(baseURL)
	if err != nil {
		return "", fmt.Errorf("textualopenai: invalid base URL: %w", err)
	}
	basePath := strings.TrimSuffix(u.Path, "/")
	u.Path = basePath + "/responses"
	return u.String(), nil
}

func (r *ResponsesRequest) Validate() error {
	if r.Model == "" {
		return errors.New("textualopenai: model is required")
	}
	if r.Input == nil {
		return errors.New("textualopenai: input is required")
	}
	if r.Stream == false {
		return errors.New("textualopenai: streaming must be enabled")
	}
	return nil
}

func (r *ResponsesRequest) SplitFunc() bufio.SplitFunc {
	return r.splitFunc
}

func (r *ResponsesRequest) AddListeners(f func(e textual.JsonGenericCarrier[StreamEvent]) textual.StringCarrier, et ...StreamEventType) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, t := range et {
		if _, ok := r.listeners[t]; ok {
			return errors.New("duplicate listener call back")
		}
		r.listeners[t] = f
	}
	return nil
}

func (r *ResponsesRequest) RemoveListener(et StreamEventType) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.listeners[et]; ok {
		delete(r.listeners, et)
		return nil
	}
	return errors.New(fmt.Sprintf("listener %s not found", et))
}

func (r *ResponsesRequest) RemoveListeners() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for eventName, _ := range r.listeners {
		delete(r.listeners, eventName)
	}
}

func (r *ResponsesRequest) AddObservers(f func(e textual.JsonGenericCarrier[StreamEvent]), et ...StreamEventType) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, eventName := range et {
		if _, ok := r.observers[eventName]; ok {
			return errors.New("duplicate observer call back")
		}
		r.observers[eventName] = f
	}
	return nil
}

func (r *ResponsesRequest) RemoveObserver(et StreamEventType) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.observers[et]; ok {
		delete(r.observers, et)
		return nil
	}
	return errors.New(fmt.Sprintf("observer %s not found", et))
}

func (r *ResponsesRequest) RemoveObservers() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for eventName, _ := range r.observers {
		delete(r.observers, eventName)
	}
}

// Transcoder returns a Transcoder that execute observation logic, and emits StreamEvent that have listeners.
func (r *ResponsesRequest) Transcoder() textual.TranscoderFunc[textual.JsonGenericCarrier[StreamEvent], textual.StringCarrier] {
	return func(ctx context.Context, in <-chan textual.JsonGenericCarrier[StreamEvent]) <-chan textual.StringCarrier {
		return textual.AsyncEmitter(ctx, in, func(ctx context.Context, c textual.JsonGenericCarrier[StreamEvent], emit func(s textual.StringCarrier)) {
			ev := c.Value
			r.mu.Lock()
			defer r.mu.Unlock()
			// the StreamEvent is unaltered.
			observerFunc, ok := r.observers[ev.Type]
			if ok {
				observerFunc(c)
			} else {
				observerFunc, ok = r.observers[AllEvent]
				if ok {
					observerFunc(c)
				}
			}
			// The StreamEvent will be processed
			// we emit the result of the listener function.
			listenerFunc, ok := r.listeners[ev.Type]
			if ok {
				res := listenerFunc(c)
				emit(res)
			} else {
				listenerFunc, ok = r.listeners[AllEvent]
				if ok {
					res := listenerFunc(c)
					emit(res)
				}
			}
		})
	}
}
