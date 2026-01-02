package textualopenai

import (
	"bufio"
	"context"
)

type Requestable interface {
	Context() context.Context
	URL(baseURL string) (string, error)
	Validate() error
	SplitFunc() bufio.SplitFunc
	AddListener(eventName string, f func(e StreamEvent)) error
	RemoveListener(eventName string) error
}
