package farm

import (
	"os"
	"path/filepath"
	"testing"
)

func TestScanCards(t *testing.T) {
	dir := t.TempDir()
	folder := filepath.Join(dir, "gopher-kungfu")
	if err := os.Mkdir(folder, 0o755); err != nil {
		t.Fatal(err)
	}
	card := `{
	  "id": "gopher-kungfu",
	  "name": "Gopher Kungfu",
	  "description": "Go concurrency",
	  "gguf": "gopher-kungfu.Q4_K_M.gguf"
	}`
	if err := os.WriteFile(filepath.Join(folder, "card.json"), []byte(card), 0o644); err != nil {
		t.Fatal(err)
	}
	f := New(dir, "", 1, 0)
	if err := f.Scan(); err != nil {
		t.Fatal(err)
	}
	models := f.Models()
	if len(models) != 1 || models[0].ID != "gopher-kungfu" {
		t.Fatalf("unexpected models: %#v", models)
	}
}
