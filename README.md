# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/ruuwao/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/ruuwao/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that picks the best free model for every request and keeps answering when a provider's free tier refuses. It serves OpenAI Chat Completions (`POST /v1/chat/completions`) and Anthropic Messages (`POST /v1/messages`), JSON and SSE streaming, against ten simulated free-tier providers on the OpenAI, Anthropic and Gemini wires, and lists the whole free catalog it found at `GET /v1/models`.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog document the server registers, byte for byte. Node 22, standard library only, nothing installed.

## How it works

- **Routing.** Each request goes to the best free model that meets its requirements, by the quality table's `rating_milli` — the top free model overall, the only tools+vision model for image+tool requests, a long-context model for `min_context` requests. A JSON-mode request on the OpenAI wire never rides an anthropic-wire provider, because that wire echoes `json_mode` false.
- **Load spreading.** A provider that has already taken its share of calls (4) yields to a lower-rated provider that has not: every provider's per-trial budget is small, and a provider that is out of budget takes every request routed to it down with it.
- **Failover.** A refusal (402, 429, 5xx) retries the request on the next candidate. The refusal's message names the kind: `budget is spent` and `is out` are permanent (that provider serves no more of the trial), `refused this call` is transient. One provider call answers each request: the first success is used.
- **Wires.** A client request is normalized once (system text, messages with text and images, tools, the output cap, stop sequences, JSON mode, streaming) and any provider's request is built from it. A provider's answer is read back into one shape — the whole text, the usage counts, the stop reason — and rendered in the client's wire. The provider's text is returned whole, never rewritten: it carries the echo the client compares. An OpenAI stream asks the provider for its usage chunk, because the client needs the usage in the stream and the provider only sends it when asked.
- **Determinism.** No wall clock, no unseeded randomness: the same brief and the same provider answers give the same bytes, so a replay reproduces the run.

## Files

- `main.ts` — the server: the brief, the catalog, the routes, register, the work poll.
- `lib/providers.ts` — the brief's three documents, the auth headers, the catalog probe.
- `lib/wire.ts` — the client wires and the three upstream wires, through one normalized shape.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?ref=tv86zvxh&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
