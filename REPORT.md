# WORM FARM v1.0

## Engineering state
The colony engine now supports a real HTTP-backed LLM brain path with strict JSON protocol validation, per-worm brain memory, deterministic replay caching, safety gating, and persistent checkpoint state.

## Verified in this workspace
- 33/33 unit tests passed.
- Full synthetic farm run using an in-process HTTP OpenAI-compatible stub completed successfully.
- The live adapter was called 12 times and its proposal materially changed finding titles.
- Brain responses were cached and persisted in colony state.
- Default offline mode remains deterministic rule-based.

## Important limitation
The repository does not claim a remote provider was contacted. The real HTTP adapter was verified against a local standards-shaped test server. A user can point it at Ollama or another OpenAI-compatible endpoint to run with an actual model.
