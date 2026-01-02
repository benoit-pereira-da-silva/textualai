package textualopenai

import (
	"fmt"

	"github.com/benoit-pereira-da-silva/textual/pkg/textual"
)

// StringCarrierFrom converts a JsonGenericCarrier containing a StreamEvent into a StringCarrier.
// It maps the Index and Error fields directly and sets the Value field.
func StringCarrierFrom(c textual.JsonGenericCarrier[StreamEvent]) textual.StringCarrier {
	s := textual.StringCarrier{
		Index: c.Index,
		Error: c.Error,
	}
	ev := c.Value

	// Field semantics:
	//  - Type: The event type identifier (always present)
	//  - Delta: Incremental text or JSON fragment
	//  - Text: Full text payload (non-streamed events) not normal in our context.
	//  - Code: Code being executed (code interpreter events)
	// - Message: Error or informational message

	switch ev.Type {
	case OutputTextDelta:
		s.Value = ev.Delta
	case ResponseFailed, RefusalDone, RefusalDelta:
		s.Value = ev.Message
		s = s.WithError(fmt.Errorf("\neventType: %s error: %s", ev.Type, ev.Message))
	default:
		if ev.Text != "" {
			s.Value = ev.Text
		} else if ev.Code != "" {
			s.Value = ev.Code
		} else if ev.Message != "" {
			s.Value = ev.Message
		}
	}
	return s
}
