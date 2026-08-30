package greet

import "testing"

func TestGreet(t *testing.T) {
	if got := Greet("gopher"); got != "hello, gopher" {
		t.Fatalf("got %q", got)
	}
	if got := Greet(""); got != "hello" {
		t.Fatalf("empty: got %q", got)
	}
}
