package farm

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestParseOffload(t *testing.T) {
	n, total, ok := ParseOffload("llama_model_loader: offloaded 32/32 layers to GPU")
	if !ok || n != 32 || total != 32 {
		t.Fatalf("full gpu: %v %d/%d", ok, n, total)
	}
	n, total, ok = ParseOffload("llm_load_tensors: offloaded 0/32 layers to GPU")
	if !ok || n != 0 || total != 32 {
		t.Fatalf("cpu: %v %d/%d", ok, n, total)
	}
	if _, _, ok := ParseOffload("ready"); ok {
		t.Fatal("expected no offload line")
	}
}

func TestSiblingBackendCudaVsCpuPath(t *testing.T) {
	dir := t.TempDir()
	cudaDir := filepath.Join(dir, "llama_cpp_cuda")
	cpuDir := filepath.Join(dir, "llama_cpp", "bin")
	if err := os.MkdirAll(cudaDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(cpuDir, 0o755); err != nil {
		t.Fatal(err)
	}
	dll := "ggml-cuda.dll"
	if runtime.GOOS != "windows" {
		dll = "ggml-cuda.so"
	}
	if err := os.WriteFile(filepath.Join(cudaDir, dll), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := siblingBackend(filepath.Join(cudaDir, "llama-server")); got != "cuda" {
		t.Fatalf("cuda siblings: %q", got)
	}
	if got := siblingBackend(filepath.Join(cpuDir, "llama-server")); got != "cpu" {
		t.Fatalf("export cpu path: %q", got)
	}
	if got := pathBackendHint(filepath.Join(dir, "llama_cpp_cuda", "llama-server.exe")); got != "cuda" {
		t.Fatalf("cuda folder name: %q", got)
	}
}

func TestClassifyStatus(t *testing.T) {
	status, reason := classifyStatus(99, "cpu", nil, nil)
	if status != "cpu" {
		t.Fatalf("cpu binary: %s %s", status, reason)
	}
	yes := true
	status, _ = classifyStatus(99, "unknown", nil, &yes)
	if status != "gpu" {
		t.Fatalf("confirmed gpu: %s", status)
	}
	no := false
	status, _ = classifyStatus(99, "cuda", []string{"RTX"}, &no)
	if status != "cpu" {
		t.Fatalf("offload 0: %s", status)
	}
	status, _ = classifyStatus(0, "cuda", []string{"RTX"}, nil)
	if status != "cpu" {
		t.Fatalf("ngl 0: %s", status)
	}
}

func TestHealthIncludesGPU(t *testing.T) {
	f := New(t.TempDir(), filepath.Join("llama_cpp", "bin", "llama-server"), 1, 99)
	f.gpuBackend = "cpu"
	h := f.Health()
	gpu, _ := h["gpu"].(GPUReport)
	if gpu.Status != "cpu" || !gpu.Requested {
		t.Fatalf("health gpu: %+v", gpu)
	}
	if h["ngl"] != 99 {
		t.Fatalf("ngl: %v", h["ngl"])
	}
}
