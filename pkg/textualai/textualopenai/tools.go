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
	"context"
	"encoding/json"
)

// JSONFunction is a minimal, dependency-free function signature for OpenAI function calling.
// The arguments are provided as raw JSON. The returned value MUST be valid JSON.
type JSONFunction func(ctx context.Context, args json.RawMessage) (json.RawMessage, error)

// FunctionCall describes a function call emitted by the model.
type FunctionCall struct {
	ItemID      string          `json:"item_id,omitempty"`
	CallID      string          `json:"call_id,omitempty"`
	Name        string          `json:"name,omitempty"`
	Arguments   json.RawMessage `json:"arguments,omitempty"`
	OutputIndex int             `json:"output_index,omitempty"`
}

// FunctionCallOutputItem is an input item you can send back to the Responses API to provide
// the output of a function call.
type FunctionCallOutputItem struct {
	Type   string `json:"type"`    // always "function_call_output"
	CallID string `json:"call_id"` // required
	Output string `json:"output"`  // JSON-encoded string
}

// FunctionCallObserver is called whenever a registered function call is finalized and executed
// by the embedded delegate.
type FunctionCallObserver func(ctx context.Context, call FunctionCall, output *FunctionCallOutputItem, err error)

// FunctionTool defines a "function" tool for the Responses API.
type FunctionTool struct {
	Type        string `json:"type"` // always "function"
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
	Strict      *bool  `json:"strict,omitempty"`
}

// BoolPtr returns a pointer to the provided bool.
func BoolPtr(v bool) *bool {
	return &v
}

// Float64Ptr returns a pointer to the provided float64.
func Float64Ptr(v float64) *float64 {
	return &v
}

// StringPtr returns a pointer to the provided string.
func StringPtr(v string) *string {
	return &v
}
