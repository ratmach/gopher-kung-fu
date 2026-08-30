---
name: gopher-kungfu
description: Offload Go implementation to the local specialist. Call run_job with workspace and one impl path; the host writes files and runs go test; do not copy source into your write tool.
---

Use `run_job` (spec + constraints + one impl path + workspace). The MCP host writes the Go and runs tests. You receive a summary only. Do not reimplement the specialist's package. If go test still FAILs after host retries, or imports are off the cartridge allowlist, implement that one file yourself — do not start another loop on the same file. `implement_go` is a lower-level escape. Use `ask_gopher_kungfu` only for review/debug of existing code.
