package textualopenai

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"sync"
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
	Model        string `json:"model,omitempty"`
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
	mu       sync.Mutex
	callBack map[string]func(e StreamEvent)
}

func NewResponsesRequest(ctx context.Context, f bufio.SplitFunc) *ResponsesRequest {
	return &ResponsesRequest{
		ctx:             ctx,
		splitFunc:       f,
		Input:           nil,
		Stream:          true,
		MaxOutputTokens: nil,
		Thinking:        false,
		callBack:        make(map[string]func(e StreamEvent)),
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

func (r *ResponsesRequest) AddListener(eventName string, f func(e StreamEvent)) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.callBack == nil {
		r.callBack = make(map[string]func(e StreamEvent))
	}
	if _, ok := r.callBack[eventName]; ok {
		return errors.New("duplicate listener call back")
	}
	r.callBack[eventName] = f
	return nil
}

func (r *ResponsesRequest) RemoveListener(eventName string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.callBack[eventName]; ok {
		delete(r.callBack, eventName)
		return nil
	}
	return errors.New(fmt.Sprintf("listener %s not found", eventName))
}
