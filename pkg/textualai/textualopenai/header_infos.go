package textualopenai

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// HeaderInfos represents metadata and rate-limit information
// returned by the OpenAI HTTP API via response headers.
//
// JSON support is provided via struct tags. time.Time fields are encoded
// using RFC3339 by default (Go's standard library behavior). ProcessingTime
// is serialized as milliseconds to match the header semantics.
type HeaderInfos struct {
	// API Meta-Information
	Organization   string        `json:"organization,omitempty"`       // openai-organization
	ProcessingTime time.Duration `json:"processing_time_ms,omitempty"` // openai-processing-ms (milliseconds in header; encoded as ms in JSON)
	APIVersion     string        `json:"api_version,omitempty"`        // openai-version
	RequestID      string        `json:"request_id,omitempty"`         // x-request-id

	// Rate Limiting Information (Requests)
	RateLimitRequestsLimit     int       `json:"ratelimit_requests_limit,omitempty"`     // x-ratelimit-limit-requests
	RateLimitRequestsRemaining int       `json:"ratelimit_requests_remaining,omitempty"` // x-ratelimit-remaining-requests
	RateLimitRequestsReset     time.Time `json:"ratelimit_requests_reset,omitempty"`     // x-ratelimit-reset-requests

	// Rate Limiting Information (Tokens)
	RateLimitTokensLimit     int       `json:"ratelimit_tokens_limit,omitempty"`     // x-ratelimit-limit-tokens
	RateLimitTokensRemaining int       `json:"ratelimit_tokens_remaining,omitempty"` // x-ratelimit-remaining-tokens
	RateLimitTokensReset     time.Time `json:"ratelimit_tokens_reset,omitempty"`     // x-ratelimit-reset-tokens
}

// HeaderInfosFromHTTPResponse extracts OpenAI-specific header information
// from an http.Response. Missing or malformed headers are ignored.
func HeaderInfosFromHTTPResponse(resp *http.Response) HeaderInfos {
	if resp == nil {
		return HeaderInfos{}
	}

	h := resp.Header

	return HeaderInfos{
		Organization:   h.Get("openai-organization"),
		ProcessingTime: parseMillisecondsDuration(h.Get("openai-processing-ms")),
		APIVersion:     h.Get("openai-version"),
		RequestID:      h.Get("x-request-id"),

		RateLimitRequestsLimit:     parseInt(h.Get("x-ratelimit-limit-requests")),
		RateLimitRequestsRemaining: parseInt(h.Get("x-ratelimit-remaining-requests")),
		RateLimitRequestsReset:     parseResetTime(h.Get("x-ratelimit-reset-requests")),

		RateLimitTokensLimit:     parseInt(h.Get("x-ratelimit-limit-tokens")),
		RateLimitTokensRemaining: parseInt(h.Get("x-ratelimit-remaining-tokens")),
		RateLimitTokensReset:     parseResetTime(h.Get("x-ratelimit-reset-tokens")),
	}
}

// ToString returns a human-readable representation of HeaderInfos,
// with exactly one information per line. Zero-value fields are skipped.
func (h HeaderInfos) ToString() string {
	var b strings.Builder

	write := func(label, value string) {
		if value == "" {
			return
		}
		b.WriteString(label)
		b.WriteString(": ")
		b.WriteString(value)
		b.WriteByte('\n')
	}

	writeInt := func(label string, value int) {
		if value == 0 {
			return
		}
		b.WriteString(label)
		b.WriteString(": ")
		b.WriteString(strconv.Itoa(value))
		b.WriteByte('\n')
	}

	writeTime := func(label string, value time.Time) {
		if value.IsZero() {
			return
		}
		b.WriteString(label)
		b.WriteString(": ")
		b.WriteString(value.Format(time.RFC3339))
		b.WriteByte('\n')
	}

	write("Organization", h.Organization)
	write("API Version", h.APIVersion)
	write("Request ID", h.RequestID)

	if h.ProcessingTime > 0 {
		b.WriteString("Processing Time: ")
		b.WriteString(strconv.FormatInt(int64(h.ProcessingTime/time.Millisecond), 10))
		b.WriteString(" ms\n")
	}

	writeInt("RateLimit Requests Limit", h.RateLimitRequestsLimit)
	writeInt("RateLimit Requests Remaining", h.RateLimitRequestsRemaining)
	writeTime("RateLimit Requests Reset", h.RateLimitRequestsReset)

	writeInt("RateLimit Tokens Limit", h.RateLimitTokensLimit)
	writeInt("RateLimit Tokens Remaining", h.RateLimitTokensRemaining)
	writeTime("RateLimit Tokens Reset", h.RateLimitTokensReset)

	return strings.TrimRight(b.String(), "\n")
}

// ToJSON returns a JSON encoding of HeaderInfos.
// ProcessingTime is emitted in milliseconds.
func (h HeaderInfos) ToJSON() ([]byte, error) {
	return json.Marshal(h)
}

// MarshalJSON encodes HeaderInfos with milliseconds for ProcessingTime
// instead of the default nanoseconds used by time.Duration.
func (h HeaderInfos) MarshalJSON() ([]byte, error) {
	type Alias HeaderInfos
	aux := struct {
		Alias
		ProcessingTimeMS int64 `json:"processing_time_ms,omitempty"`
	}{
		Alias:            Alias(h),
		ProcessingTimeMS: durationToMilliseconds(h.ProcessingTime),
	}

	// Avoid emitting the default time.Duration (nanoseconds) form.
	aux.Alias.ProcessingTime = 0

	return json.Marshal(aux)
}

// parseInt safely parses an integer header value.
func parseInt(value string) int {
	i, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil {
		return 0
	}
	return i
}

// parseMillisecondsDuration converts a millisecond string into a time.Duration.
func parseMillisecondsDuration(value string) time.Duration {
	ms, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64)
	if err != nil {
		return 0
	}
	return time.Duration(ms) * time.Millisecond
}

// parseResetTime parses rate limit reset headers.
// Some APIs provide UNIX timestamps; if absent or invalid, returns zero time.
func parseResetTime(value string) time.Time {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}
	}

	// Try UNIX timestamp (seconds).
	if ts, err := strconv.ParseInt(value, 10, 64); err == nil {
		return time.Unix(ts, 0)
	}

	return time.Time{}
}

func durationToMilliseconds(d time.Duration) int64 {
	if d <= 0 {
		return 0
	}
	return int64(d / time.Millisecond)
}
