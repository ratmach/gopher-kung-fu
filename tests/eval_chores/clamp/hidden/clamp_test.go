package clamp

import "testing"

func TestClamp(t *testing.T) {
	if got := Clamp(5, 0, 10); got != 5 {
		t.Fatalf("in range: got %d", got)
	}
	if got := Clamp(-2, 0, 10); got != 0 {
		t.Fatalf("below: got %d", got)
	}
	if got := Clamp(99, 0, 10); got != 10 {
		t.Fatalf("above: got %d", got)
	}
}
