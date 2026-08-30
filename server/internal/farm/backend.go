package farm

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"
)

var (
	offloadRe = regexp.MustCompile(`(?i)offloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU`)
	deviceRe  = regexp.MustCompile(`(?im)^\s*Device\s+\d+:\s+(.+?)(?:,|$)`)
)

type GPUReport struct {
	Requested bool           `json:"requested"`
	Status    string         `json:"status"` // gpu | cpu | unknown
	Backend   string         `json:"backend"`
	Devices   []string       `json:"devices,omitempty"`
	Reason    string         `json:"reason"`
	NGL       int            `json:"ngl"`
	Llama     string         `json:"llama_server"`
	Confirmed bool           `json:"confirmed"`
	Loaded    []LoadedOffload `json:"loaded,omitempty"`
}

type LoadedOffload struct {
	ID      string `json:"id"`
	Offload string `json:"offload,omitempty"`
	OnGPU   bool   `json:"on_gpu"`
}

func siblingBackend(llamaPath string) string {
	dir := filepath.Dir(llamaPath)
	names := map[string]string{
		"ggml-cuda.dll":     "cuda",
		"ggml-cuda.so":      "cuda",
		"libggml-cuda.so":   "cuda",
		"ggml-hip.dll":      "hip",
		"ggml-hip.so":       "hip",
		"ggml-vulkan.dll":   "vulkan",
		"ggml-vulkan.so":    "vulkan",
		"ggml-metal.dylib":  "metal",
		"libggml-metal.dylib": "metal",
	}
	for name, backend := range names {
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			return backend
		}
	}
	return pathBackendHint(llamaPath)
}

func pathBackendHint(llamaPath string) string {
	parts := strings.Split(strings.ToLower(strings.ReplaceAll(llamaPath, "\\", "/")), "/")
	for _, part := range parts {
		switch {
		case part == "cuda" || part == "llama_cpp_cuda" || strings.Contains(part, "win-cuda"):
			return "cuda"
		case part == "vulkan" || strings.Contains(part, "win-vulkan"):
			return "vulkan"
		case part == "metal":
			return "metal"
		case part == "hip" || part == "rocm":
			return "hip"
		}
	}
	joined := strings.Join(parts, "/")
	if strings.Contains(joined, "llama_cpp/bin") {
		return "cpu"
	}
	return ""
}

func probeListDevices(llamaPath string) (backend string, devices []string, raw string) {
	if llamaPath == "" {
		return "", nil, ""
	}
	cmd := exec.Command(llamaPath, "--list-devices")
	cmd.Env = os.Environ()
	done := make(chan struct{})
	var out []byte
	var err error
	go func() {
		out, err = cmd.CombinedOutput()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(8 * time.Second):
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		return "", nil, ""
	}
	text := string(out)
	if err != nil && text == "" {
		return "", nil, ""
	}
	low := strings.ToLower(text)
	switch {
	case strings.Contains(low, "cuda") || strings.Contains(low, "nvidia"):
		backend = "cuda"
	case strings.Contains(low, "vulkan"):
		backend = "vulkan"
	case strings.Contains(low, "metal"):
		backend = "metal"
	case strings.Contains(low, "hip") || strings.Contains(low, "rocm"):
		backend = "hip"
	}
	for _, match := range deviceRe.FindAllStringSubmatch(text, -1) {
		name := strings.TrimSpace(match[1])
		if name != "" && !strings.EqualFold(name, "cpu") {
			devices = append(devices, name)
		}
	}
	if backend == "" && strings.Contains(low, "cpu") && !strings.Contains(low, "gpu") {
		backend = "cpu"
	}
	return backend, devices, text
}

func ParseOffload(logText string) (gpuLayers, total int, ok bool) {
	match := offloadRe.FindStringSubmatch(logText)
	if match == nil {
		return 0, 0, false
	}
	gpuLayers = atoiSafe(match[1])
	total = atoiSafe(match[2])
	return gpuLayers, total, true
}

func atoiSafe(s string) int {
	n := 0
	for _, ch := range s {
		if ch < '0' || ch > '9' {
			break
		}
		n = n*10 + int(ch-'0')
	}
	return n
}

func classifyStatus(ngl int, backend string, devices []string, confirmedOnGPU *bool) (status, reason string) {
	if ngl < 1 {
		return "cpu", "-ngl 0: GPU offload disabled"
	}
	if confirmedOnGPU != nil {
		if *confirmedOnGPU {
			return "gpu", "llama-server offloaded layers to GPU"
		}
		return "cpu", "llama-server offloaded 0 layers to GPU"
	}
	switch backend {
	case "cuda", "vulkan", "metal", "hip":
		if len(devices) > 0 {
			return "gpu", strings.ToUpper(backend) + " binary sees " + strings.Join(devices, ", ") + " (unconfirmed until a model loads)"
		}
		return "gpu", strings.ToUpper(backend) + " backend beside llama-server (unconfirmed until a model loads)"
	case "cpu":
		return "cpu", "llama-server has no GPU backend; -ngl is ignored"
	default:
		if runtime.GOOS == "darwin" {
			return "unknown", "could not tell GPU vs CPU; load a model and re-check /health"
		}
		return "unknown", "could not tell GPU vs CPU from this llama-server; load a model and re-check /health"
	}
}

func (f *Farm) probeBackend() {
	path := f.LlamaServer
	backend, devices, _ := probeListDevices(path)
	if backend == "" {
		backend = siblingBackend(path)
	}
	f.mu.Lock()
	f.gpuBackend = backend
	f.gpuDevices = devices
	f.mu.Unlock()
}

func (f *Farm) GPUSummary() string {
	h := f.Health()
	rep, _ := h["gpu"].(GPUReport)
	devs := ""
	if len(rep.Devices) > 0 {
		devs = " (" + strings.Join(rep.Devices, ", ") + ")"
	}
	return strings.ToUpper(rep.Status) + devs + " — " + rep.Reason
}
