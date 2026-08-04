# CONTROL CHESS — Build Specification

**Paste target:** opencode agent. Work milestone by milestone (M0→M6). Do not advance a milestone until its acceptance criteria pass. Run tests with `node --test`. The rules in §2 and decisions in §3 are final — do not re-ask questions already answered here.

---

## 1. Summary

A browser chess game, human (White) vs AI (Black), with one variant mechanic: **Reanimation**. When you capture an enemy piece it *defects to your side* and enters your pool. Three full rounds after the capture, you may spend your entire turn dropping it onto any **empty square your pieces attack**. Each piece can be reanimated **once, ever** — if a reanimated piece is captured, it goes to the graveyard permanently and joins nobody's pool. The AI has the identical mechanic and plays it strategically. Dark, modern, high-polish presentation with big bold pieces and ghostly reanimation VFX.

---

## 2. Definitive Rules

Standard FIDE chess applies except as amended below. "Round" = one fullmove (White moves, then Black moves), tracked by the fullmove number.

- **R1 — Capture → pool.** When a player captures a non-reanimated enemy piece, that piece switches to the capturer's color and enters the capturer's pool with a cooldown. Kings are never captured (normal check/checkmate rules), so a king can never appear in a pool.
- **R2 — Cooldown = 3 full rounds.** A piece captured during round *R* becomes droppable on its owner's turn in round *R+3*. UI shows a countdown badge on the pooled piece: 3 → 2 → 1 → ready. Implementation: store `capturedOnRound`; droppable when `currentFullmove >= capturedOnRound + 3`.
- **R3 — Drop legality.** On your turn you may, instead of moving, drop exactly one ready pool piece onto a square that is (a) empty — occupied by neither side — and (b) attacked by at least one of your on-board pieces. "Attacked" uses standard chess attack semantics: pawn diagonals only (never the forward push squares), sliders blocked by occupancy, king adjacency counts, and absolutely pinned pieces still project attacks. The drop must not leave your own king in check. Dropping consumes the entire turn.
- **R4 — One-shot reanimation.** Every dropped piece is flagged `reanimated`. Capturing a `reanimated` piece sends it to the graveyard — it enters no pool. Original (never-dropped) pieces are the only ones that enter pools when captured.
- **R5 — Drops count as legal moves for game-end detection.** Checkmate = in check with no legal board moves *and* no legal drops. Stalemate = not in check with no legal board moves *and* no legal drops. Consequences (intended): a ready drop that blocks a check or a stalemate is legal and prevents the game from ending; drops may also deliver check or checkmate.
- **R6 — Pawn drops and promotion.** Dropping a pawn on its promotion rank (rank 8 for White, rank 1 for Black) opens the promotion picker (Q/R/B/N); the resulting piece is still flagged `reanimated`, and the drop still consumes the turn. A pawn dropped on any other rank is just a pawn. A pawn dropped on its color's 2nd rank may later use the two-square advance.
- **R7 — En passant.** A drop never creates an en-passant opportunity, and a `reanimated` pawn can **never** be captured en passant — even after a later two-square advance. Normal en passant between original pawns is unchanged.
- **R8 — Castling.** Castling rights attach only to the original king and original rooks on their starting squares, per FIDE. A dropped rook never confers or restores castling rights, regardless of where it lands.
- **R9 — Symmetry.** The AI has the identical pool/cooldown/drop system and its reanimations render with the same VFX at the moment they happen (no advance telegraphing).
- **R10 — Draws.** Threefold repetition auto-draws; the repetition key must include board (with reanimated flags), side to move, castling rights, ep square, and both pools including cooldown states. Fifty-move rule applies; any drop resets the halfmove clock. Insufficient material: declare only for bare K vs K with both pools empty.

## 3. Resolved Design Decisions

These settle everything the requirements left open. Implement the defaults; where noted, gate behind a config constant in `js/config.js` so they're one-line flips.

- **D1** Captured pieces defect to the capturer's color (this is the whole mechanic — a captured black knight becomes a white knight in White's pool).
- **D2** A piece promoted by normal play (original pawn walks to the 8th) is captured *as its current type* — a promoted queen enters the pool as a queen. Config: `REVERT_PROMOTED_ON_CAPTURE = false` (Crazyhouse-style pawn reversion if flipped).
- **D3** Pawn drops on the owner's own 1st rank are allowed (nothing in the rules forbids it; no double-step from rank 1). Config: `ALLOW_BACK_RANK_PAWN_DROP = true`.
- **D4** Pinned pieces still create legal drop squares (standard attack semantics, see R3).
- **D5** Human plays White and moves first; AI plays Black. Side selection ships later alongside the difficulty selector.
- **D6** Exactly one action per turn: a board move *or* a single drop, never both, never two drops.

---

## 4. Tech Stack & Hard Constraints

- **Vanilla JS (ES2022 modules), HTML, CSS. No framework, no bundler, no npm dependencies.** Runs by serving the folder statically (`npx serve .` or any static server; document this in README). Rationale: instant iteration, zero build failures, and the visual bar is met with modern CSS + SVG + a canvas particle layer.
- **Engine is DOM-free.** Everything under `js/engine/` must run in Node with zero browser globals so `node --test` can exercise it directly. UI code imports the engine; never the reverse.
- **AI runs in a Web Worker** (`js/ai/worker.js`). The main thread must never block during search; animations stay at 60fps while the AI thinks.
- **Zero binary assets.** Pieces are hand-authored inline SVG (one sprite file, 12 `<symbol>`s: 6 types × 2 colors — note: 12, not 8). All sounds are synthesized at runtime with the Web Audio API. Nothing is downloaded.
- Use JSDoc type annotations (`// @ts-check` optional) on engine modules.

## 5. File Structure

```
control-chess/
├── index.html
├── css/
│   └── style.css              // design tokens + all styling
├── js/
│   ├── config.js              // D2/D3 flags, timing constants, AI limits
│   ├── main.js                // bootstrap, game loop wiring
│   ├── engine/
│   │   ├── state.js           // state model, clone, serialize/deserialize
│   │   ├── movegen.js         // board moves + drop generation, attack maps
│   │   ├── rules.js           // legality, check/mate/stalemate/draws, make/unmake
│   │   └── zobrist.js         // hashing incl. pools & flags
│   ├── ai/
│   │   ├── worker.js          // Web Worker entry, message protocol
│   │   ├── search.js          // iterative deepening negamax + alpha-beta + TT
│   │   └── eval.js            // evaluation terms (§7)
│   ├── ui/
│   │   ├── board.js           // render, drag-and-drop, highlights, animations
│   │   ├── pool.js            // pool panels, cooldown badges, drop dragging
│   │   ├── vfx.js             // canvas particle overlay, reanimation effect
│   │   ├── modal.js           // promotion picker, game-over overlay
│   │   └── audio.js           // Web Audio synthesis (§9)
│   └── pieces.svg.js          // exports the inline SVG sprite string
├── tests/
│   ├── perft.test.mjs
│   ├── drops.test.mjs
│   └── endstates.test.mjs
└── README.md
```

## 6. State Model & Move Representation

```js
// Piece (board or pool)
{ id: number, type: 'P'|'N'|'B'|'R'|'Q'|'K', color: 'w'|'b',
  reanimated: boolean, droppedPawn: boolean /* for R7 ep immunity */ }

// Pool entry
{ piece: Piece, capturedOnRound: number }   // ready when fullmove >= capturedOnRound + 3

// GameState
{ board: (Piece|null)[64],                  // 0x88 or 8x8 flat — your call, be consistent
  turn: 'w'|'b',
  castling: { wK, wQ, bK, bQ },
  epSquare: number|null,
  pools: { w: PoolEntry[], b: PoolEntry[] },
  graveyard: Piece[],
  halfmoveClock: number, fullmove: number,
  repetitionKeys: Map<bigint, number>,       // Zobrist → count
  history: Move[] }

// Move — two kinds
{ kind: 'move', from, to, promo?: 'Q'|'R'|'B'|'N', /* flags: capture, ep, castle, doubleStep */ }
{ kind: 'drop', poolIndex: number, to: number, promo?: 'Q'|'R'|'B'|'N' }
```

Serialization: extended FEN — standard six fields, with `~` suffixed to reanimated pieces on the board (lichess Crazyhouse convention), plus a seventh field for pools, e.g. `pools:w[N@2,P@0]b[Q@1]` where the number is rounds remaining. Implement `serialize(state)` / `deserialize(str)`; tests use these to set up positions.

## 7. AI Requirements

**Protocol:** main thread posts `{type:'search', stateFEN, limits:{ms, maxDepth}}`; worker replies `{type:'bestmove', move, info:{depth, score, nodes}}`. Worker owns its own engine import.

**Search:** iterative deepening negamax with alpha-beta; transposition table keyed by Zobrist hash (must incorporate pools: piece counts per type per side per cooldown-bucket, plus reanimated flags on board squares, turn, castling, ep); move ordering: TT move → captures by MVV-LVA → checking moves/drops → killer moves → history heuristic; quiescence search on captures only. **Drop branching control:** generate all legal drops at ply 0–1; at deeper plies restrict drop candidates to drops that give check, land adjacent to the enemy king, block a current check, or land on the central 16 squares — cap 12 drop candidates per node.

**Evaluation (centipawns):**
- Material on board: P=100, N=320, B=330, R=500, Q=950.
- Pool material ramps toward full value as it nears readiness: `value × (0.55 + 0.15 × (3 − roundsRemaining))` → 55% at capture, 100% when ready. This makes the AI value its bank, avoid feeding one-shot pieces cheaply, and time drops rather than dumping them the instant they unlock.
- Standard piece-square tables (any published simplified-eval set).
- King safety: attack-unit count on the king ring, **scaled up when the opponent has ready or nearly-ready pool pieces** — as in Crazyhouse, droppable material near your king is the dominant tactical threat in this variant.
- Small tempo bonus.

**Intermediate difficulty (ship default):** 1.2 s/move time budget, maxDepth 5. Difficulty selector later maps to `{easy: depth 2 + Gaussian eval noise σ≈60cp, intermediate: 1.2s, hard: 4s}` — structure `limits` so this is trivial.

## 8. UI / UX Specification

**Layout.** Board centered, sized `min(86vh, 720px)`, White at bottom, subtle file/rank coordinates on the board edge. Right rail (~280px): AI pool panel on top, player pool panel on bottom, slim status strip between them (turn indicator, check warning, engine "thinking" shimmer). Header: game title, New Game button. Responsive: below 900px the rail moves under the board as a horizontal strip; support Pointer Events so touch dragging works.

**Design tokens** (CSS custom properties; this is a spectral/necromancy theme — commit to it rather than a generic dark dashboard):

```css
:root {
  --bg-0:#0a0c12; --bg-1:#11141d;            /* page: near-black indigo, radial vignette */
  --sq-dark:#262c3a; --sq-light:#8f9ab0;      /* board */
  --ivory:#f2ead9; --obsidian:#15161c;        /* piece bodies */
  --accent:#5eead4;                            /* player highlights: cold teal */
  --spectral:#8b5cf6;                          /* reanimation: violet ghost-light */
  --danger:#f87171;                            /* checks & threats */
}
```

Typography: a characterful display face for the title/status (e.g. a sharp variable serif or engraved-feel face via system/`@font-face`-free fallback stack — do not default to Inter-everywhere), a clean grotesque for UI labels, tabular numerals for cooldown badges.

**Pieces.** The 12 SVG symbols are the centerpiece: bold silhouettes filling ~80% of the square, layered gradients for volume, 1.5px rim-light stroke, soft drop shadow. White = warm ivory with cool sheen; Black = obsidian with faint violet rim. Reanimated pieces on the board carry a permanent subtle spectral tint (thin `--spectral` outer glow) so both players can read at a glance which pieces are one-shot.

**Signature element — the pool + reanimation flow.** Pool pieces render as spectral cards: the piece ghosted at ~55% opacity behind frosted glass (`backdrop-filter: blur`), with a circular cooldown badge showing the number inside a radial progress ring that depletes each round. When a piece becomes ready it "ignites": opacity to 100%, badge dissolves, slow violet pulse. Dragging a ready piece over the board shows a translucent ghost under the cursor; every legal drop square pulses with a soft `--spectral` ring; illegal squares give no affordance. On drop: materialize effect ≈ 450ms — scale 0.6→1.0, blur 8px→0, additive glow flash, 20–30 rising violet wisp particles on the canvas overlay. The AI's drops play the identical effect.

**Moves & feedback.** Board moves animate as 160ms transform glides; captures play a 200ms shatter-fade on the victim while it flies to the capturer's pool card; last move highlighted; selected piece shows legal-move dots (rings on captures); check pulses a `--danger` glow on the king; promotion picker is a compact 4-piece modal; game over is a dimmed overlay with result + New Game. Respect `prefers-reduced-motion`: swap glides/particles for instant moves and simple fades.

## 9. Audio (Web Audio synthesis — no files)

Build a tiny synth in `audio.js`: `move` = short filtered click (noise burst through bandpass, 60ms); `capture` = low 90Hz thump + noise crunch (120ms); `reanimate` = ethereal chime — two detuned sine partials (~880/1320Hz) with slow attack, feedback-delay shimmer, 700ms; `promote` = quick ascending 4-note arpeggio; `check` = muted two-tone alert; `gameEnd` = resolving triad. Master gain node with a mute toggle in the header. Instantiate the `AudioContext` on first user gesture (autoplay policy).

## 10. Milestones & Acceptance Criteria

**M0 — Scaffold & static render.** Folder structure, tokens, board renders with all 32 pieces from the SVG sprite, rail panels empty. ✅ Loads from a static server with zero console errors; board is crisp at both 1440p and 390px width.

**M1 — Standard chess engine (DOM-free).** Full legal move generation, make/unmake, check/mate/stalemate, castling, en passant, promotion; `serialize`/`deserialize`. ✅ `node --test tests/perft.test.mjs` passes perft from the initial position: depth 1=20, 2=400, 3=8 902, 4=197 281. ✅ Kiwipete (`r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -`) depth 3 = 97 862.

**M2 — Variant layer.** Pools, cooldown timing, drop generation/legality, reanimated/droppedPawn flags, R4–R10 semantics, extended FEN. ✅ `tests/drops.test.mjs` + `tests/endstates.test.mjs` cover at minimum: capture enters pool at cooldown 3 and unlocks in round R+3 exactly; drops only onto empty attacked squares; pawn-attack squares are diagonals only; drop resolving a check is legal and mate detection accounts for it; position with no board moves but a ready drop is *not* stalemate; capturing a reanimated piece yields no pool entry; dropped rook confers no castling; dropped pawn is ep-immune after a double-step; pawn drop on rank 8 requires a promo choice and stays flagged; repetition key distinguishes positions differing only in pool cooldowns.

**M3 — Interactive UI.** Click-or-drag board moves, pool dragging with legal-square affordances, cooldown badges ticking, promotion modal, status strip, New Game. ✅ A full human-vs-human hotseat game is playable end to end with only legal actions possible; badges visibly count 3→2→1→ready.

**M4 — AI.** Worker protocol, search, eval, drop candidates. ✅ AI never proposes an illegal action across 50 automated AI-vs-AI games (assert legality on every applied move, run via a Node harness reusing the worker's search module); AI responds within its time budget; UI stays interactive at 60fps while it thinks; the AI demonstrably drops pieces (log drop frequency in the harness — must be > 0).

**M5 — VFX & audio polish.** Materialize effect, particles, shatter-fade captures, spectral pool cards, full synth set, reduced-motion path. ✅ Reanimation effect plays for both sides; muting works; `prefers-reduced-motion` verified.

**M6 — QA hardening.** Rapid-input fuzz on the UI (spam clicks/drags during animations and AI think — state must never desync), resize/mobile pass, README with run instructions and rules summary. ✅ No console errors across a full game; refresh mid-AI-think doesn't wedge state.

## 11. Non-Goals (this build)

Difficulty selector UI and side selection (structure for them per §7/D5, don't build), online play, PGN export, opening book, clocks, save/resume across sessions.
