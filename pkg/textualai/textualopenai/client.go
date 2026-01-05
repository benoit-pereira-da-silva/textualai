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

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
)

// Client defines a textual client for OpenAI-compatible /v1 endpoints.
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

func (c Client) Config() Config {
	return c.config
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

	// Authorization is optional for OpenAI-compatible providers (e.g. Ollama).
	// When apiKeyRequired=true, ensureConfig() guarantees apiKey is present.
	if strings.TrimSpace(c.config.apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+c.config.apiKey)
	}

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

func (c Client) StreamAndTranscodeResponses(ctx context.Context, req *ResponsesRequest) (string, HeaderInfos, error) {
	resp, err := c.Stream(req)
	headerInfos := HeaderInfosFromHTTPResponse(resp)
	if err != nil {
		return "", headerInfos, err
	}
	defer func() {
		req.RemoveListeners()
		req.RemoveObservers()
		_ = resp.Body.Close()
	}()

	// Apply the transcoder func to the body split by SSE event.
	ioT := textual.NewIOReaderTranscoder[textual.JsonGenericCarrier[StreamEvent], textual.StringCarrier](req.Transcoder(), resp.Body)
	ioT.SetSplitFunc(req.SplitFunc())
	ioT.SetContext(ctx)
	outCh := ioT.Start()

	// To accumulate the values, we Consume the response channel
	var b strings.Builder
	for {
		select {
		case <-ctx.Done():
			if errors.Is(ctx.Err(), context.Canceled) {
				return "", headerInfos, ctx.Err()
			}
			return b.String(), headerInfos, ctx.Err()

		case item, ok := <-outCh:
			b.WriteString(item.Value)
			if !ok {
				// Return the accumulated string
				return b.String(), headerInfos, nil // stream finished normally
			}
		}
	}
}

func (c Client) ensureConfig() error {
	if strings.TrimSpace(c.config.baseURL) == "" {
		return errors.New("textualopenai: missing base URL (TEXTUALAI_API_URL / TERMCHAT_API_URL)")
	}
	if strings.TrimSpace(string(c.config.model)) == "" {
		return errors.New("textualopenai: missing model (OPENAI_MODEL / TERMCHAT_MODEL)")
	}
	if c.config.apiKeyRequired && strings.TrimSpace(c.config.apiKey) == "" {
		return errors.New("textualopenai: missing API key (OPENAI_API_KEY)")
	}
	return nil
}
