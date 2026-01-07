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
	"net/url"
	"os"
	"strings"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
	"github.com/benoit-pereira-da-silva/textualai/pkg/textualai/models"
)

// Client defines a textual client for OpenAI-compatible /v1 endpoints.
type Client struct {
	httpClient *http.Client
	ctx        context.Context

	apiKey         string
	baseURL        string
	model          models.Model
	apiKeyRequired bool
}

func ClientFrom(baseURL string, model models.Model, ctx context.Context, apiKeyEnvVarNames ...string) (Client, error) {
	// NOTE: http.Client.Timeout covers the whole request lifetime (including reading resp.Body).
	// For streaming requests, we rely on context cancellation instead, so Timeout is left to 0.
	client := Client{
		httpClient: &http.Client{
			Timeout: 0,
		},
		ctx: ctx,
	}
	client.model = model
	client.apiKey = strings.TrimSpace(firstNonEmpty(apiKeyEnvVarNames...))
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		baseURL = model.ProviderInfo().DefaultBaseURL
	}
	validUrl, err := url.Parse(baseURL)
	if err != nil {
		return client, err
	}
	client.baseURL = validUrl.String()
	client.apiKeyRequired = model.ProviderInfo().APIKeyRequired
	return client, nil
}

func (c Client) WithApiKey(apiKey string) Client {
	c.apiKey = strings.TrimSpace(apiKey)
	return c
}

func (c Client) Model() models.Model {
	return c.model
}

func firstNonEmpty(values ...string) string {
	for _, key := range values {
		v := os.Getenv(key)
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// Stream opens a streaming connection to the Responses endpoint and returns the raw HTTP response.
// Callers must close resp.Body.
func (c Client) Stream(r Requestable) (*http.Response, error) {
	if r == nil {
		return nil, errors.New("textualopenai: nil ResponsesRequest")
	}
	if err := r.Validate(); err != nil {
		return nil, err
	}
	endpoint, err := r.URL(c.baseURL)
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
	if strings.TrimSpace(c.apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
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
