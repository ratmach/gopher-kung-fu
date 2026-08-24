package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"customslm/farm/internal/farm"
)

func main() {
	var (
		addr       = flag.String("addr", "127.0.0.1:8080", "listen address")
		cartridges = flag.String("cartridges", "./cartridges", "directory of specialist cartridges")
		llama      = flag.String("llama-server", envOr("LLAMA_SERVER", "llama-server"), "path to llama-server")
		maxLoaded  = flag.Int("max-loaded", 2, "how many GGUFs stay resident (LRU)")
		gpuLayers  = flag.Int("ngl", 99, "GPU layers to offload (-ngl)")
	)
	flag.Parse()

	abs, err := filepath.Abs(*cartridges)
	if err != nil {
		log.Fatal(err)
	}
	f := farm.New(abs, lookPath(*llama), *maxLoaded, *gpuLayers)
	if err := f.Scan(); err != nil {
		log.Fatal(err)
	}
	log.Printf("farm root %s — %d specialists", abs, len(f.Models()))

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"ok": true, "models": len(f.Models())})
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		_ = f.Scan()
		type item struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
			Name    string `json:"name"`
			Desc    string `json:"description"`
		}
		data := []item{}
		for _, card := range f.Models() {
			data = append(data, item{
				ID:      card.ID,
				Object:  "model",
				OwnedBy: "custom-slm",
				Name:    card.Name,
				Desc:    card.Description,
			})
		}
		writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		_ = f.Scan()
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var envelope struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(body, &envelope); err != nil || envelope.Model == "" {
			http.Error(w, `{"error":{"message":"model is required"}}`, http.StatusBadRequest)
			return
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))
		if err := f.ServeChat(w, r, envelope.Model); err != nil {
			status := http.StatusNotFound
			if errors.Is(err, farm.ErrNoLlama) {
				status = http.StatusServiceUnavailable
			}
			writeJSON(w, status, map[string]any{
				"error": map[string]string{"message": err.Error(), "model": envelope.Model},
			})
		}
	})

	srv := &http.Server{Addr: *addr, Handler: withCORS(mux), ReadHeaderTimeout: 10 * time.Second}
	go func() {
		log.Printf("cartridge farm listening on http://%s", *addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(err)
		}
	}()

	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGINT, syscall.SIGTERM)
	<-ch
	f.Shutdown()
	_ = srv.Close()
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func lookPath(name string) string {
	if name == "" {
		return ""
	}
	if _, err := os.Stat(name); err == nil {
		return name
	}
	if path, err := exec.LookPath(name); err == nil {
		return path
	}
	return ""
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}
