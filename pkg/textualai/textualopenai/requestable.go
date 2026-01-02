package textualopenai

import (
	"bufio"
	"context"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
)

type Requestable interface {
	Context() context.Context
	URL(baseURL string) (string, error)
	Validate() error
	SplitFunc() bufio.SplitFunc
	AddListener(eventName StreamEventType, f func(e StreamEvent) textual.StringCarrier) error
	RemoveListener(eventName StreamEventType) error
}
