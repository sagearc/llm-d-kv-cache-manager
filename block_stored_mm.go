// Place this file in the llm-d-kv-cache repo, e.g. cmd/print_event/main.go
// Run: go run ./cmd/print_event/ (with block_stored_example.msgpack in the working dir)
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

func main() {
	payload, err := os.ReadFile("block_stored_example.msgpack")
	if err != nil {
		panic(err)
	}

	adapter := engineadapter.NewVLLMAdapter()
	_, _, batch, err := adapter.ParseMessage(&kvevents.RawMessage{
		Topic:   "kv@pod-1@test-model",
		Payload: payload,
	})
	if err != nil {
		panic(err)
	}

	for i, event := range batch.Events {
		out, err := json.MarshalIndent(event, "", "    ")
		if err != nil {
			panic(err)
		}
		fmt.Printf("event[%d] (%T):\n%s\n", i, event, string(out))
	}
}
