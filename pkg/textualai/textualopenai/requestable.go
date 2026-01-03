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
	AddListeners(f func(e textual.JsonGenericCarrier[StreamEvent]) textual.StringCarrier, et ...EventType) error

	// RemoveListener removes a previously registered listener for a specified EventType and returns an error if the removal fails.
	RemoveListener(et EventType) error

	// AddObservers registers a callback function to handle specified EventType events and returns an error if registration fails.
	AddObservers(f func(e textual.JsonGenericCarrier[StreamEvent]), et ...EventType) error

	// RemoveObserver removes a previously registered observer for a specified EventType and returns an error if the removal fails.
	RemoveObserver(et EventType) error
}
