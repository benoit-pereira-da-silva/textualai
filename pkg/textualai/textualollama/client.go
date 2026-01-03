package textualollama

import (
	"context"
	"net/http"
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
