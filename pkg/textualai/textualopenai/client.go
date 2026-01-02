package textualopenai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// Client defines a textual client for OpenAI's platform
type Client struct {
	config     Config
	httpClient *http.Client
	ctx        context.Context
}

func NewClient(config Config, ctx context.Context) Client {
	// NOTE: http.Client.Timeout covers the whole request lifetime (including reading resp.Body).
	// For streaming requests, we rely on context cancellation instead, so Timeout is left to 0.
	return Client{
		config: config,
		httpClient: &http.Client{
			Timeout: 0,
		},
		ctx: ctx,
	}
}

// Stream opens a streaming connection to the Responses endpoint and returns the raw HTTP response.
// Callers must close resp.Body.
func (c Client) Stream(r Requestable) (*http.Response, error) {
	if err := c.ensureConfig(); err != nil {
		return nil, err
	}
	if r == nil {
		return nil, errors.New("textualopenai: nil ResponsesRequest")
	}
	if r.Validate() != nil {
		return nil, r.Validate()
	}
	endpoint, err := r.URL(c.config.baseURL)
	if err != nil {
		return nil, err
	}
	bodyBytes, err := json.Marshal(r)
	if err != nil {
		return nil, fmt.Errorf("textualopenai: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("textualopenai: create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.config.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("textualopenai: http request: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		msg := strings.TrimSpace(string(b))
		if msg == "" {
			msg = resp.Status
		}
		return nil, fmt.Errorf("textualopenai: responses stream failed: http %d: %s", resp.StatusCode, msg)
	}

	return resp, nil
}

func (c Client) ensureConfig() error {
	if strings.TrimSpace(c.config.apiKey) == "" {
		return errors.New("textualopenai: missing OPENAI_API_KEY")
	}
	if strings.TrimSpace(c.config.baseURL) == "" {
		return errors.New("textualopenai: missing OpenAI base URL (OPENAI_API_URL)")
	}
	if strings.TrimSpace(string(c.config.model)) == "" {
		return errors.New("textualopenai: missing model (OPENAI_MODEL)")
	}
	return nil
}
