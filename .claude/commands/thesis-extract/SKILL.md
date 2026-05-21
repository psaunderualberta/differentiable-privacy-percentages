---
name: thesis-extract
description: Extract thesis-relevant information (design decisions, numerical stability tricks, JAX/Equinox patterns, privacy accounting math) from a targeted area of the codebase and append findings to thesis/notes/<topic>.md. Use when the user wants to document codebase internals for their thesis, capture design rationale, record implementation details, or mentions "thesis notes" / "extract for thesis".
---

# Thesis Extract

Pulls thesis-relevant information out of a targeted part of the codebase and writes it to a per-topic notes file under `thesis/notes/`.

## Workflow

### 1. Identify the target

Ask the user **what topic, module, file, function, or class** to extract from. Examples: "the projection algorithm", "alternating schedule", "`train_with_noise`".

Infer the output filename from the topic: lowercase, hyphenated, no extension prompt needed. E.g. "projection algorithm" → `thesis/notes/projection-algorithm.md`.

### 2. Ask depth per category

For each of the four categories below, ask the user for depth: **brief / standard / exhaustive**. Present all four in a single question. If the user says "all standard" or similar, apply uniformly.

| Category | What to extract |
|---|---|
| **Design decisions & rationale** | Why this approach was chosen; alternatives implicitly rejected; non-obvious behaviour and gotchas |
| **Numerical stability tricks** | eps clamps, log-space math, bisection/Newton bounds, NaN/Inf guards, `eqx.error_if`, `lax.select` safeguards |
| **JAX/Equinox patterns & performance** | `shard_map`, `jax.checkpoint`, `eqx.filter_jit`, `partition`/`combine`, `pvary`, `vmap` structure, scan + checkpoint memory tradeoffs |
| **Privacy accounting math** | GDP conversions, `project_weights` constraint derivations, weights→σ→clip mappings, ε/δ↔μ relationships |

Depth guide:
- **brief** — 1–3 bullets per category, no snippets
- **standard** — full prose paragraphs, snippets only for non-obvious lines
- **exhaustive** — every relevant detail, all non-intuitive snippets, full derivation chains

### 3. Extract

For the chosen target, search the codebase (Grep/Read) and assemble findings. Every claim must carry a **file + function/class citation**, e.g. `src/privacy/gdp_privacy.py :: GDPPrivacyParameters.project_weights`. Use line numbers when a specific line is being referenced.

Include **short verbatim snippets (≤8 lines)** only when the behaviour is non-intuitive or the logic is unique — e.g. a custom bisection invariant, a `lax.stop_gradient` placement that flips alternating optimization, a non-obvious `pvary` call. Skip snippets for routine code.

### 4. Write to file

Target path: `thesis/notes/<inferred-topic>.md` (relative to repo root).

- **If the file does not exist**: create it with a `# <Topic>` H1 and write the four category sections.
- **If the file exists**: append. Insert a horizontal rule and a line stating the append, then the new sections:
  ```
  ---

  _Appended 2026-05-14: <one-line note on what was added or refined>._
  ```
  Use today's date from the environment context. Do not rewrite or remove existing content unless the user explicitly asks.

### 5. Confirm

After writing, report: the path, which categories were populated, and a one-sentence summary of what each category covered. Offer to drill deeper into any category.

## Output structure

```md
# <Topic>

**Scope:** <files / functions / classes covered>

## Design decisions & rationale
- <finding> — `path/to/file.py :: ClassName.method` (L123)
  > optional snippet if non-intuitive

## Numerical stability
...

## JAX/Equinox patterns
...

## Privacy accounting math
...
```

Omit a category section entirely if the target has nothing relevant for it (don't write empty headings).

## Rules

- Never invent rationale. If the "why" is not visible in code, comments, commit messages, or `CLAUDE.md`/`MEMORY.md`, mark it as **(rationale not in repo — confirm with author)**.
- Cite file + function/class for every claim. Add line numbers when pointing at a specific line.
- Snippets only when non-intuitive or unique. No snippets for boilerplate.
- Always append to existing topic files with the dated separator above; never silently overwrite.
- Stay inside the user's chosen target — do not expand scope without asking.
