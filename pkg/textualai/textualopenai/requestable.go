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

	// AddListeners registers a callback function to handle events of specified types and returns an error if registration fails.
	AddListeners(f func(e StreamEvent) textual.StringCarrier, et ...StreamEventType) error

	// RemoveListener removes a previously registered listener for a specified StreamEventType and returns an error if the removal fails.
	RemoveListener(et StreamEventType) error

	// AddObservers registers a callback function to handle specified StreamEventType events and returns an error if registration fails.
	AddObservers(f func(e StreamEvent), et ...StreamEventType) error

	// RemoveObserver removes a previously registered observer for a specified StreamEventType and returns an error if the removal fails.
	RemoveObserver(et StreamEventType) error
}
