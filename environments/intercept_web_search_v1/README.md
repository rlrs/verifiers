# intercept-web-search-v1

Small V1 example showing `@vf.intercept` on native vf model-turn types.

## Develop

The bundled harness runs the local Codex CLI with native Responses `web_search` enabled. The taskset intercepts
the parsed `vf.Response`, asks a rewrite model for a German JSON rewrite, mutates
`response.message.content` and `response.message.provider_state`, and lets Verifiers serialize the
modified response back into Codex's native stream.

```text
Codex --search
  -> V1 interception server
  -> OpenAI /v1/responses
  <- vf.Response
  <- @vf.intercept rewrites text + citation/source state
  <- Codex receives rewritten Responses SSE
```

This example requires `codex` on `PATH` and an OpenAI Responses endpoint that supports native
`web_search`.

```bash
uv run --with-editable . --with-editable environments/intercept_web_search_v1 \
  eval intercept-web-search-v1 \
  -m gpt-4.1-mini \
  --client.base-url https://api.openai.com/v1 \
  --client.api-key-var OPENAI_API_KEY \
  -n 1 -r 1 --max-turns 2 --timeout.rollout 180
```

## Layout

- `intercept_web_search_v1/taskset.py` — one task plus the `@vf.intercept` rewrite hook.
- `intercept_web_search_v1/harness.py` — local Codex CLI harness with native search enabled.

## Notes

The interceptor receives typed vf objects, not SSE bytes or provider JSON. For Responses streams,
Verifiers buffers the turn only when an interceptor is present, then emits a rewritten stream after
the modified `vf.Response` is committed to the trace.
