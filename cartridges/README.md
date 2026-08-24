# Cartridges

Each exported specialist lands here as:

```
cartridges/<slug>/
  <slug>.Q4_K_M.gguf
  card.json
```

The farm server scans this directory and exposes every `card.json` as an OpenAI model id.
