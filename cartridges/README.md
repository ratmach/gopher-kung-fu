# Cartridges

Each exported specialist lands here as:

```
cartridges/<slug>/
  <slug>.Q4_K_M.gguf
  <slug>.Q5_K_M.gguf
  card.json
```

`card.json` names the shipped quant (`gguf` + `quant`, default Q4_K_M) plus `allowed_imports` / `forbidden_imports`. Eval Q5 separately; point `gguf` at the winner if it scores better.

The farm server scans this directory and exposes every `card.json` as an OpenAI model id.
