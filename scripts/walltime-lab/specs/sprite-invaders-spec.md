# Sprite Invaders — Design Document

Build a small browser canvas shooter as ES modules. Follow this spec exactly.
Create every file listed, then run `node --test tests/` (the engine modules
are DOM-free on purpose) and fix failures until green.

## Files

1. `js/engine/vec.js` — `add`, `scale`, `clamp(v, min, max)`,
   `aabbOverlap(a, b)` for `{x, y, w, h}` boxes. Pure functions.
2. `js/engine/world.js` — Game state, DOM-free.
   - `createWorld({ width, height })`: player (bottom-center, 3 lives),
     `bullets: []`, `invaders` = 5×8 grid with per-invader `{alive}`,
     `direction: 1`, `score: 0`, `wave: 1`, `status: 'playing'`.
   - `step(world, input, dt)`: player moves ±240 px/s clamped; `input.fire`
     spawns a bullet (max 3 live player bullets, 360 px/s up); invader block
     marches horizontally at `24 + 6*wave` px/s, drops 18 px and reverses on
     wall contact; bottom-row invaders fire downward at most one bullet per
     900 ms globally; bullet↔invader and bullet↔player collisions via
     `aabbOverlap`; scoring 10×row-bonus; lives hit → respawn or
     `status:'gameover'`; all invaders dead → next wave (faster, grid reset).
3. `js/engine/spawn.js` — `invaderGrid(cols, rows, spacing)` layout helper
   and `nextWave(world)`.
4. `js/render.js` — Canvas drawing (rects + score/lives/wave text, simple
   invader sprite from a 2D bit array). Only file allowed to touch the DOM
   besides input.js and main.js.
5. `js/input.js` — Keyboard state tracker (arrows/A-D + space), returns a
   `read()` snapshot `{ left, right, fire }`.
6. `js/main.js` — Bootstrap: canvas 480×560, `requestAnimationFrame` loop
   with fixed 16 ms max dt, wires input → `step` → render, R restarts.
7. `index.html` — Canvas + module script tag, dark background, centered.
8. `tests/world.test.mjs` — `node:test`: grid layout counts, march-and-drop
   reversal, player bullet cap, collision scoring, wave advance, game over.
9. `tests/vec.test.mjs` — vector/AABB cases.

## Conventions

- ES modules, no bundler, no dependencies. Engine files must never import
  DOM APIs (tests run in plain Node).
- A minimal `package.json` with `"type": "module"`.

## Acceptance

- `node --test tests/` exits 0.
- Opening `index.html` shows the grid marching and a controllable player.
