/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tokenization

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	tokenizerpb "github.com/llm-d/llm-d-kv-cache/api/tokenizerpb"
	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
)

// mockTokenizationServer implements the TokenizationServiceServer interface for testing
type mockTokenizationServer struct {
	tokenizerpb.UnimplementedTokenizationServiceServer
	initializeError bool
	tokenizeError   bool
	chatError       bool
	initialized     map[string]bool
}

func newMockTokenizationServer() *mockTokenizationServer {
	return &mockTokenizationServer{
		initialized: make(map[string]bool),
	}
}

func (m *mockTokenizationServer) InitializeTokenizer(
	ctx context.Context,
	req *tokenizerpb.InitializeTokenizerRequest,
) (*tokenizerpb.InitializeTokenizerResponse, error) {
	if m.initializeError {
		return &tokenizerpb.InitializeTokenizerResponse{
			Success:      false,
			ErrorMessage: "mock initialization error",
		}, nil
	}

	m.initialized[req.ModelName] = true
	return &tokenizerpb.InitializeTokenizerResponse{
		Success: true,
	}, nil
}

func (m *mockTokenizationServer) Tokenize(
	ctx context.Context,
	req *tokenizerpb.TokenizeRequest,
) (*tokenizerpb.TokenizeResponse, error) {
	if m.tokenizeError {
		return &tokenizerpb.TokenizeResponse{
			Success:      false,
			ErrorMessage: "mock tokenization error",
		}, nil
	}

	// Check if model was initialized (matches real service behavior)
	if !m.initialized[req.ModelName] {
		return &tokenizerpb.TokenizeResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("model %s not initialized", req.ModelName),
		}, nil
	}

	// Simple deterministic mock tokenization: convert each rune to a token ID
	// This makes tests more realistic - different inputs produce different tokens
	input := req.Input
	tokens := make([]uint32, 0, len(input))
	offsets := make([]uint32, 0, len(input)*2)

	for i, r := range input {
		tokens = append(tokens, uint32(r))
		offsets = append(offsets, uint32(i), uint32(i+1))
	}

	return &tokenizerpb.TokenizeResponse{
		InputIds:     tokens,
		Success:      true,
		OffsetPairs:  offsets,
		ErrorMessage: "",
	}, nil
}

func (m *mockTokenizationServer) RenderChatTemplate(
	ctx context.Context,
	req *tokenizerpb.ChatTemplateRequest,
) (*tokenizerpb.ChatTemplateResponse, error) {
	if m.chatError {
		return &tokenizerpb.ChatTemplateResponse{
			Success:      false,
			ErrorMessage: "mock chat template error",
		}, nil
	}

	// Check if model was initialized (matches real service behavior)
	if !m.initialized[req.ModelName] {
		return &tokenizerpb.ChatTemplateResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("model %s not initialized", req.ModelName),
		}, nil
	}

	// Mock chat template rendering by concatenating messages
	rendered := ""
	for _, turn := range req.ConversationTurns {
		for _, msg := range turn.Messages {
			rendered += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
		}
	}

	return &tokenizerpb.ChatTemplateResponse{
		RenderedPrompt: rendered,
		Success:        true,
	}, nil
}

// setupTestServer creates a mock gRPC server on a temporary Unix socket
func setupTestServer(t *testing.T, mockServer *mockTokenizationServer) (string, func()) {
	t.Helper()

	// Create a temporary directory for the socket
	tmpDir := t.TempDir()
	socketPath := filepath.Join(tmpDir, "test-tokenizer.socket")

	// Create a Unix listener
	listener, err := net.Listen("unix", socketPath)
	require.NoError(t, err)

	// Create and start the gRPC server
	grpcServer := grpc.NewServer()
	tokenizerpb.RegisterTokenizationServiceServer(grpcServer, mockServer)

	go func() {
		if err := grpcServer.Serve(listener); err != nil {
			t.Logf("Server error: %v", err)
		}
	}()

	// Wait for the server to be ready
	time.Sleep(100 * time.Millisecond)

	cleanup := func() {
		grpcServer.Stop()
		listener.Close()
		os.RemoveAll(socketPath)
	}

	return socketPath, cleanup
}

func TestNewUdsTokenizer_Success(t *testing.T) {
	mockServer := newMockTokenizationServer()
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	// Verify the model was initialized
	assert.True(t, mockServer.initialized["test-model"])

	// Clean up
	udsTokenizer, ok := tokenizer.(*UdsTokenizer)
	require.True(t, ok)
	err = udsTokenizer.Close()
	assert.NoError(t, err)
}

func TestNewUdsTokenizer_InitializationFailure(t *testing.T) {
	mockServer := newMockTokenizationServer()
	mockServer.initializeError = true
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	assert.Error(t, err)
	assert.Nil(t, tokenizer)
	assert.Contains(t, err.Error(), "tokenizer initialization failed")
}

func TestNewUdsTokenizer_ConnectionFailure(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: "/non/existent/socket.sock",
	}

	// This should fail quickly because the socket doesn't exist
	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	assert.Error(t, err)
	assert.Nil(t, tokenizer)
}

func TestUdsTokenizer_Render(t *testing.T) {
	mockServer := newMockTokenizationServer()
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	defer func() {
		udsTokenizer, ok := tokenizer.(*UdsTokenizer)
		require.True(t, ok)
		_ = udsTokenizer.Close()
	}()

	// Test Render - character-based tokenization
	input := "hello world"
	tokens, offsets, err := tokenizer.Render(input)
	require.NoError(t, err)
	
	// Each character becomes a token
	assert.Equal(t, len([]rune(input)), len(tokens))
	assert.Equal(t, len([]rune(input)), len(offsets))
	
	// Verify specific characters
	assert.Equal(t, uint32('h'), tokens[0])      // 'h' = 104
	assert.Equal(t, uint32(' '), tokens[5])      // space at position 5 = 32
	assert.Equal(t, uint32('d'), tokens[10])     // 'd' at end = 100
	
	// Verify offsets
	assert.Equal(t, preprocessing.Offset{0, 1}, offsets[0])  // 'h'
	assert.Equal(t, preprocessing.Offset{5, 6}, offsets[5])  // space
}
}

func TestUdsTokenizer_Encode(t *testing.T) {
	mockServer := newMockTokenizationServer()
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	udsTokenizer, ok := tokenizer.(*UdsTokenizer)
	require.True(t, ok)

	defer func() {
		_ = udsTokenizer.Close()
	}()

	// Test Encode method - verifies character-based tokenization
	prompt := "test prompt"
	tokens, offsets, err := udsTokenizer.Encode(prompt, true)
	require.NoError(t, err)
	
	// Each character becomes a token with its rune value
	assert.Equal(t, len([]rune(prompt)), len(tokens))
	assert.Equal(t, len([]rune(prompt)), len(offsets))
	
	// Verify first character 't' = 116
	assert.Equal(t, uint32('t'), tokens[0])
	// Verify space character at position 4 = 32
	assert.Equal(t, uint32(' '), tokens[4])
	// Verify last character 't' = 116
	assert.Equal(t, uint32('t'), tokens[len(tokens)-1])
}

func TestUdsTokenizer_RenderChat(t *testing.T) {
	mockServer := newMockTokenizationServer()
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	defer func() {
		udsTokenizer, ok := tokenizer.(*UdsTokenizer)
		require.True(t, ok)
		_ = udsTokenizer.Close()
	}()

	// Test RenderChat
	renderReq := &preprocessing.RenderChatRequest{
		Conversation: []preprocessing.ChatMessage{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there"},
		},
		AddGenerationPrompt: true,
		ChatTemplateKWArgs: map[string]interface{}{
			"key1": "value1",
			"key2": float64(42),
			"key3": true,
		},
	}

	tokens, offsets, err := tokenizer.RenderChat(renderReq)
	require.NoError(t, err)
	// Verify tokens are generated from rendered chat (character-based)
	assert.Greater(t, len(tokens), 0, "should return tokens from rendered chat")
	assert.Equal(t, len(tokens), len(offsets), "offsets should match token count")
}
}

func TestUdsTokenizer_RenderChat_ComplexKWArgs(t *testing.T) {
	mockServer := newMockTokenizationServer()
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	defer func() {
		udsTokenizer, ok := tokenizer.(*UdsTokenizer)
		require.True(t, ok)
		_ = udsTokenizer.Close()
	}()

	// Test RenderChat with complex kwargs (list and struct)
	renderReq := &preprocessing.RenderChatRequest{
		Conversation: []preprocessing.ChatMessage{
			{Role: "system", Content: "You are a helpful assistant"},
		},
		ChatTemplateKWArgs: map[string]interface{}{
			"list_arg": []interface{}{"item1", "item2", float64(3)},
			"struct_arg": map[string]interface{}{
				"nested_key": "nested_value",
				"nested_num": float64(100),
			},
			"nil_arg": nil,
		},
	}

	tokens, offsets, err := tokenizer.RenderChat(renderReq)
	require.NoError(t, err)
	assert.NotNil(t, tokens)
	assert.NotNil(t, offsets)
}

func TestUdsTokenizer_Type(t *testing.T) {
	mockServer := newMockTokenizationServer()
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	defer func() {
		udsTokenizer, ok := tokenizer.(*UdsTokenizer)
		require.True(t, ok)
		_ = udsTokenizer.Close()
	}()

	assert.Equal(t, "external-uds", tokenizer.Type())
}

func TestUdsTokenizer_TokenizeError(t *testing.T) {
	mockServer := newMockTokenizationServer()
	mockServer.tokenizeError = true
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	defer func() {
		udsTokenizer, ok := tokenizer.(*UdsTokenizer)
		require.True(t, ok)
		_ = udsTokenizer.Close()
	}()

	_, _, err = tokenizer.Render("test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "tokenization failed")
}

func TestUdsTokenizer_ChatTemplateError(t *testing.T) {
	mockServer := newMockTokenizationServer()
	mockServer.chatError = true
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	defer func() {
		udsTokenizer, ok := tokenizer.(*UdsTokenizer)
		require.True(t, ok)
		_ = udsTokenizer.Close()
	}()

	renderReq := &preprocessing.RenderChatRequest{
		Conversation: []preprocessing.ChatMessage{
			{Role: "user", Content: "Hello"},
		},
	}

	_, _, err = tokenizer.RenderChat(renderReq)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "chat template rendering failed")
}

func TestUdsTokenizer_InvalidOffsets(t *testing.T) {
	mockServer := newMockTokenizationServer()
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	// Override the Tokenize method to return invalid offsets
	originalTokenize := mockServer.Tokenize
	mockServer.Tokenize = func(ctx context.Context, req *tokenizerpb.TokenizeRequest) (*tokenizerpb.TokenizeResponse, error) {
		return &tokenizerpb.TokenizeResponse{
			InputIds:    []uint32{1, 2, 3},
			Success:     true,
			OffsetPairs: []uint32{0, 5, 10}, // Invalid: odd number of elements
		}, nil
	}
	defer func() {
		mockServer.Tokenize = originalTokenize
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	defer func() {
		udsTokenizer, ok := tokenizer.(*UdsTokenizer)
		require.True(t, ok)
		_ = udsTokenizer.Close()
	}()

	_, _, err = tokenizer.Render("test")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid offset_pairs")
}

func TestUdsTokenizer_ContextCancellation(t *testing.T) {
	mockServer := newMockTokenizationServer()
	socketPath, cleanup := setupTestServer(t, mockServer)
	defer cleanup()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	config := &UdsTokenizerConfig{
		SocketFile: socketPath,
	}

	tokenizer, err := NewUdsTokenizer(ctx, config, "test-model")
	require.NoError(t, err)
	require.NotNil(t, tokenizer)

	udsTokenizer, ok := tokenizer.(*UdsTokenizer)
	require.True(t, ok)

	// Cancel the context - this should trigger the cleanup goroutine
	cancel()

	// Give the goroutine time to run
	time.Sleep(200 * time.Millisecond)

	// The connection should be closed, but Close() should still be safe to call
	err = udsTokenizer.Close()
	assert.NoError(t, err)
}

func TestUdsTokenizerConfig_IsEnabled(t *testing.T) {
	tests := []struct {
		name     string
		config   *UdsTokenizerConfig
		expected bool
	}{
		{
			name:     "nil config",
			config:   nil,
			expected: false,
		},
		{
			name: "empty socket file",
			config: &UdsTokenizerConfig{
				SocketFile: "",
			},
			expected: false,
		},
		{
			name: "valid socket file",
			config: &UdsTokenizerConfig{
				SocketFile: "/tmp/test.socket",
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.config.IsEnabled()
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestConvertToProtoValue(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		validate func(*testing.T, *tokenizerpb.Value)
	}{
		{
			name:  "nil value",
			input: nil,
			validate: func(t *testing.T, v *tokenizerpb.Value) {
				assert.Equal(t, "", v.GetStringValue())
			},
		},
		{
			name:  "string value",
			input: "test string",
			validate: func(t *testing.T, v *tokenizerpb.Value) {
				assert.Equal(t, "test string", v.GetStringValue())
			},
		},
		{
			name:  "number value",
			input: float64(42.5),
			validate: func(t *testing.T, v *tokenizerpb.Value) {
				assert.Equal(t, float64(42.5), v.GetNumberValue())
			},
		},
		{
			name:  "bool value",
			input: true,
			validate: func(t *testing.T, v *tokenizerpb.Value) {
				assert.Equal(t, true, v.GetBoolValue())
			},
		},
		{
			name:  "list value",
			input: []interface{}{"a", float64(1), true},
			validate: func(t *testing.T, v *tokenizerpb.Value) {
				listVal := v.GetListValue()
				require.NotNil(t, listVal)
				assert.Equal(t, 3, len(listVal.Values))
				assert.Equal(t, "a", listVal.Values[0].GetStringValue())
				assert.Equal(t, float64(1), listVal.Values[1].GetNumberValue())
				assert.Equal(t, true, listVal.Values[2].GetBoolValue())
			},
		},
		{
			name: "struct value",
			input: map[string]interface{}{
				"key1": "value1",
				"key2": float64(100),
			},
			validate: func(t *testing.T, v *tokenizerpb.Value) {
				structVal := v.GetStructValue()
				require.NotNil(t, structVal)
				assert.Equal(t, "value1", structVal.Fields["key1"].GetStringValue())
				assert.Equal(t, float64(100), structVal.Fields["key2"].GetNumberValue())
			},
		},
		{
			name:  "unknown type",
			input: struct{ Name string }{Name: "test"},
			validate: func(t *testing.T, v *tokenizerpb.Value) {
				// Should convert to string representation
				assert.Contains(t, v.GetStringValue(), "test")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertToProtoValue(tt.input)
			require.NotNil(t, result)
			tt.validate(t, result)
		})
	}
}
