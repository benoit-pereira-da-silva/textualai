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
	//  - Delta: Incremental text or JSON fragment (depending on the event type)
	//  - Text: Finalized text payload for *.done events
	//  - Refusal: Finalized refusal payload for refusal *.done events
	//  - Code: Code being executed (code interpreter events) OR the error code (error events)
	//  - Message: Error or informational message

	switch ev.Type {
	case OutputTextDelta:
		s.Value = ev.Delta

	case TextDone:
		// Do not emit the full text again: streaming clients already received the deltas.
		s.Value = ""

	case RefusalDelta:
		s.Value = ev.Delta
		s = s.WithError(fmt.Errorf("\neventType: %s refusal: %s", ev.Type, ev.Delta))

	case RefusalDone:
		s.Value = ev.Refusal
		s = s.WithError(fmt.Errorf("\neventType: %s refusal: %s", ev.Type, ev.Refusal))

	case ResponseFailed, Error:
		// Response failures and "error" events should be surfaced as errors.
		msg := ev.Message
		if msg == "" && ev.Refusal != "" {
			msg = ev.Refusal
		}
		if msg == "" && ev.Text != "" {
			msg = ev.Text
		}
		s.Value = msg
		s = s.WithError(fmt.Errorf("\neventType: %s error: %s", ev.Type, msg))

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
