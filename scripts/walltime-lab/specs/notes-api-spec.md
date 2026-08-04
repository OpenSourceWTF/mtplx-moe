# Markdown Notes API — Design Document

Build a zero-dependency Node.js HTTP JSON API for markdown notes with
file-backed storage. Follow this spec exactly. Create every file listed, then
run `node --test tests/` and fix failures until green.

## Files

1. `src/store.js` — File-backed note store.
   - Notes live as `<id>.json` files under a data directory:
     `{ id, title, body, tags: [], createdAt, updatedAt }`.
   - `createStore(dir)` returns `{ list, get, create, update, remove }`, all
     async. `create` assigns `id` = 12-hex random. `update` bumps
     `updatedAt`. `remove` returns false for unknown ids. `list` supports
     `{ tag }` filtering and returns summaries (no body).
2. `src/validate.js` — Input validation.
   - `validateNote(payload, { partial })` returns `{ ok, errors }`:
     title required non-empty string ≤200 chars (unless partial), body
     string ≤50_000, tags array of ≤16 lowercase slugs (`/^[a-z0-9-]{1,32}$/`).
3. `src/router.js` — Routing without frameworks.
   - `route(method, url)` → `{ handlerName, params }` for:
     `GET /notes`, `POST /notes`, `GET /notes/:id`, `PATCH /notes/:id`,
     `DELETE /notes/:id`, `GET /health`. Unknown → `{ handlerName: 'notFound' }`.
4. `src/server.js` — HTTP wiring.
   - `createServer(store)` returns a `node:http` server: JSON bodies
     (reject >1 MB with 413), correct status codes (200/201/204/400/404),
     `content-type: application/json` on every response, errors as
     `{ error: { code, message } }`.
5. `bin/serve.js` — `#!/usr/bin/env node`; env `PORT` (default 3000) and
   `DATA_DIR` (default `./data`); prints one startup line.
6. `tests/store-validate.test.mjs` — store CRUD round-trip in a temp dir,
   tag filtering, validation accept/reject table.
7. `tests/http.test.mjs` — boots the server on an ephemeral port, exercises
   every route incl. 400/404/413 paths via `fetch`.

## Conventions

- ES modules, `"type": "module"` in a minimal `package.json`, Node 20+.
- No third-party packages. Handlers small and pure where possible.

## Acceptance

- `node --test tests/` exits 0.
- `PORT=0 node bin/serve.js` starts and answers `GET /health` with
  `{ "ok": true }` (verify once with curl or fetch).
