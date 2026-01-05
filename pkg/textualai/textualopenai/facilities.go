// Copyright 2026 Benoit Pereira da Silva
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
	case OutputTextDelta, ReasoningSummaryTextDelta:
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
