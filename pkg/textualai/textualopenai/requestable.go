package textualopenai

import (
	"bufio"
	"context"
)

type Requestable interface {
	Validate() error
	URL(baseURL string) (string, error)
	SplitFunc() bufio.SplitFunc
	Context() context.Context
	AddListener(eventName string, f func(e StreamEvent)) error
	RemoveListener(eventName string) error
}
