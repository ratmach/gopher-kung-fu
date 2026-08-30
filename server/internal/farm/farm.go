package farm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type Card struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	GGUF        string `json:"gguf"`
	Quant       string `json:"quant"`
	BaseModel   string `json:"base_model"`
	Dir         string `json:"-"`
}

type instance struct {
	card    Card
	cmd     *exec.Cmd
	port    int
	proxy   *httputil.ReverseProxy
	last    time.Time
	cancel  func()
	offload string
	onGPU   bool
}

type logBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *logBuf) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.Write(p)
}

func (l *logBuf) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.b.String()
}

type Farm struct {
	Dir         string
	LlamaServer string
	Host        string
	MaxLoaded   int
	GPULayers   int
	CtxSize     int

	mu         sync.Mutex
	cards      map[string]Card
	loaded     map[string]*instance
	nextPort   int
	gpuBackend string
	gpuDevices []string
}

func New(dir, llama string, maxLoaded, gpuLayers int) *Farm {
	if maxLoaded < 1 {
		maxLoaded = 1
	}
	return &Farm{
		Dir:         dir,
		LlamaServer: llama,
		Host:        "127.0.0.1",
		MaxLoaded:   maxLoaded,
		GPULayers:   gpuLayers,
		CtxSize:     32768,
		cards:       map[string]Card{},
		loaded:      map[string]*instance{},
		nextPort:    18100,
	}
}

func (f *Farm) Scan() error {
	entries, err := os.ReadDir(f.Dir)
	if err != nil {
		if os.IsNotExist(err) {
			f.mu.Lock()
			f.cards = map[string]Card{}
			f.mu.Unlock()
			return nil
		}
		return err
	}
	next := map[string]Card{}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		folder := filepath.Join(f.Dir, entry.Name())
		raw, err := os.ReadFile(filepath.Join(folder, "card.json"))
		if err != nil {
			continue
		}
		raw = bytes.TrimPrefix(raw, []byte{0xEF, 0xBB, 0xBF})
		var card Card
		if err := json.Unmarshal(raw, &card); err != nil {
			log.Printf("skip %s: %v", folder, err)
			continue
		}
		if card.ID == "" {
			card.ID = entry.Name()
		}
		if card.Name == "" {
			card.Name = card.ID
		}
		card.Dir = folder
		next[card.ID] = card
	}
	f.mu.Lock()
	f.cards = next
	f.mu.Unlock()
	return nil
}

func (f *Farm) Probe() {
	f.probeBackend()
}

func (f *Farm) Health() map[string]any {
	f.mu.Lock()
	loaded := make([]LoadedOffload, 0, len(f.loaded))
	var confirmed *bool
	for id, inst := range f.loaded {
		item := LoadedOffload{ID: id, Offload: inst.offload, OnGPU: inst.onGPU}
		loaded = append(loaded, item)
		if inst.offload != "" {
			v := inst.onGPU
			confirmed = &v
		}
	}
	backend := f.gpuBackend
	devices := append([]string(nil), f.gpuDevices...)
	llama := f.LlamaServer
	ngl := f.GPULayers
	f.mu.Unlock()
	sort.Slice(loaded, func(i, j int) bool { return loaded[i].ID < loaded[j].ID })
	status, reason := classifyStatus(ngl, backend, devices, confirmed)
	gpu := GPUReport{
		Requested: ngl > 0,
		Status:    status,
		Backend:   backend,
		Devices:   devices,
		Reason:    reason,
		NGL:       ngl,
		Llama:     llama,
		Confirmed: confirmed != nil,
		Loaded:    loaded,
	}
	return map[string]any{
		"ok":           true,
		"models":       len(f.Models()),
		"llama_server": llama,
		"ngl":          ngl,
		"gpu":          gpu,
	}
}

func (f *Farm) Models() []Card {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]Card, 0, len(f.cards))
	for _, card := range f.cards {
		out = append(out, card)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func (f *Farm) Ensure(id string) (*instance, error) {
	f.mu.Lock()
	if inst, ok := f.loaded[id]; ok {
		inst.last = time.Now()
		f.mu.Unlock()
		return inst, nil
	}
	card, ok := f.cards[id]
	f.mu.Unlock()
	if !ok {
		return nil, ErrUnknown
	}
	gguf := filepath.Join(card.Dir, card.GGUF)
	if card.GGUF == "" {
		matches, _ := filepath.Glob(filepath.Join(card.Dir, "*.gguf"))
		if len(matches) == 1 {
			gguf = matches[0]
		}
	}
	if _, err := os.Stat(gguf); err != nil {
		return nil, fmt.Errorf("gguf missing for %s: %w", id, err)
	}
	if f.LlamaServer == "" {
		return nil, ErrNoLlama
	}
	f.evictIfNeeded(id)
	port, err := f.allocPort()
	if err != nil {
		return nil, err
	}
	ctx := f.CtxSize
	if ctx < 1 {
		ctx = 32768
	}
	args := []string{
		"-m", gguf,
		"--host", f.Host,
		"--port", fmt.Sprintf("%d", port),
		"--ctx-size", fmt.Sprintf("%d", ctx),
	}
	if f.GPULayers > 0 {
		args = append(args, "-ngl", fmt.Sprintf("%d", f.GPULayers))
	}
	logs := &logBuf{}
	cmd := exec.Command(f.LlamaServer, args...)
	cmd.Stdout = io.MultiWriter(os.Stdout, logs)
	cmd.Stderr = io.MultiWriter(os.Stderr, logs)
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start llama-server: %w", err)
	}
	target, _ := url.Parse(fmt.Sprintf("http://%s:%d", f.Host, port))
	proxy := httputil.NewSingleHostReverseProxy(target)
	inst := &instance{
		card:  card,
		cmd:   cmd,
		port:  port,
		proxy: proxy,
		last:  time.Now(),
		cancel: func() {
			_ = cmd.Process.Kill()
		},
	}
	if err := waitReady(target.String()+"/health", 60*time.Second); err != nil {
		inst.cancel()
		return nil, fmt.Errorf("llama-server for %s did not become ready: %w", id, err)
	}
	if n, total, ok := ParseOffload(logs.String()); ok {
		inst.offload = fmt.Sprintf("%d/%d", n, total)
		inst.onGPU = n > 0
		if inst.onGPU {
			log.Printf("loaded %s on :%d GPU offload %s (%s)", id, port, inst.offload, gguf)
		} else {
			log.Printf("loaded %s on :%d CPU (offload %s) (%s)", id, port, inst.offload, gguf)
		}
	} else {
		log.Printf("loaded %s on :%d (%s)", id, port, gguf)
	}
	f.mu.Lock()
	f.loaded[id] = inst
	f.mu.Unlock()
	return inst, nil
}

func (f *Farm) evictIfNeeded(incoming string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.loaded) < f.MaxLoaded {
		return
	}
	var oldestID string
	var oldest time.Time
	for id, inst := range f.loaded {
		if id == incoming {
			continue
		}
		if oldestID == "" || inst.last.Before(oldest) {
			oldestID = id
			oldest = inst.last
		}
	}
	if oldestID == "" {
		return
	}
	log.Printf("unloading %s (LRU)", oldestID)
	f.loaded[oldestID].cancel()
	delete(f.loaded, oldestID)
}

func (f *Farm) allocPort() (int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for i := 0; i < 40; i++ {
		port := f.nextPort
		f.nextPort++
		if f.nextPort > 19000 {
			f.nextPort = 18100
		}
		ln, err := net.Listen("tcp", fmt.Sprintf("%s:%d", f.Host, port))
		if err != nil {
			continue
		}
		_ = ln.Close()
		return port, nil
	}
	return 0, errors.New("no free port for llama-server")
}

func (f *Farm) Shutdown() {
	f.mu.Lock()
	defer f.mu.Unlock()
	for id, inst := range f.loaded {
		inst.cancel()
		delete(f.loaded, id)
	}
}

var (
	ErrUnknown = errors.New("unknown specialist")
	ErrNoLlama = errors.New("llama-server binary not found; set --llama-server or PATH")
)

func (f *Farm) ServeChat(w http.ResponseWriter, r *http.Request, model string) error {
	inst, err := f.Ensure(model)
	if err != nil {
		return err
	}
	inst.proxy.ServeHTTP(w, r)
	return nil
}

func waitReady(url string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode < 500 {
				return nil
			}
		}
		// llama-server may not expose /health on older builds; try models
		resp, err = client.Get(replaceHealth(url))
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode < 500 {
				return nil
			}
		}
		time.Sleep(400 * time.Millisecond)
	}
	return errors.New("timeout")
}

func replaceHealth(u string) string {
	if len(u) > 7 {
		return u[:len(u)-len("/health")] + "/v1/models"
	}
	return u
}
