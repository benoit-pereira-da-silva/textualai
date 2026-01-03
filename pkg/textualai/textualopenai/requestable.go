package textualopenai

import (
	"context"
)

type Requestable interface {
	Context() context.Context

	Validate() error

	URL(baseURL string) (string, error)
}
