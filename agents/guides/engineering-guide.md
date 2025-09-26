# Engineering Guidelines – Micro‑Design Principles

A living guide for everyday decisions that keep the codebase evolvable, testable, and pleasant to work in.

## Audience & Scope

- Audience: Tomeo contributors (core + plugins).
- Scope: Micro‑level decisions (functions, classes, modules, small features). Pairs with the Architecture doc (macro boundaries).
- Default stance: Design Mode (production‑ready).

## North‑Star Principles

- Seams over Singletons – prefer interfaces/abstractions and dependency seams to enable swapping and testing.
- Small, Focused Units – one purpose per function/module; fewer than ~40 lines per function is a good heuristic.
- Pure Core, Dirty Edges – keep pure logic isolated; push I/O and side‑effects to the boundaries.
- Make Change Easy – design for the likely next change; prefer composition, not inheritance.
- Readable First – code is for humans; optimize for clarity before cleverness.

> Rule of thumb: if you can’t write a 1‑sentence purpose for a function, it’s doing too much.

## Layering & Boundaries (Micro)

- CLI/TUI ↔ Core: UI calls commands (use‑cases) that orchestrate work; it never reaches into repos/services directly.
- Plugins ↔ Kernel: plugins implement hooks/contracts; the kernel coordinates. No plugin reaches into another plugin’s internals.
- DB Access: isolate with repositories or service objects; no SQL/ORM calls from UI or plugin entrypoints.
- AI Providers: go through adapters; no direct SDK calls in feature code.

**Do**

```
ui → command/usecase → services/repos → infra
```

**Avoid**

```
ui → random helper → ORM call → half business logic → AI SDK
```


## Function & Class Design

- Function size: aim for ≤ 40 LOC; break by intent (parse → validate → act → format).
- Arguments: prefer small, typed parameter lists. Use dataclasses/Pydantic models for structured inputs.
- Return types: return data, not print; UI decides presentation.
- Classes: use for stateful workflows or cohesive behavior; otherwise keep it functional.


**Template**

```
def do_thing(input: InputModel, deps: Deps) -> OutputModel:
    """Parse → validate → compute → persist → format."""
    parsed = parse(input)
    ensure(valid(parsed))
    result = core_compute(parsed, deps)
    persist(result, deps)
    return format_out(result)
```


## Error Handling & Results

- Business errors: raise domain exceptions (e.g., TomeNotFound, ReconstructionFailed). Catch at the command boundary to map to user‑friendly messages.
- Expected fallible ops: consider Result[Ok, Err] pattern (typed container) when it improves clarity.
- Logging on edges: log context at boundaries; avoid logging deep in pure core.


**Do**

```
try:
    run_job(job_id)
except JobAlreadyRunning as e:
    return warn(str(e))
```

Avoid swallowing exceptions or returning None for error states.


## Dependency Injection (Practical)

- Contracts first: define protocols/ABCs in core (e.g., AIProvider, Clock, Keychain).
- Impls in adapters: concrete classes live in ai/, security/, db/, or plugin packages.
- Bootstrap composes: wire dependencies in a bootstrap() or factories; pass them explicitly into commands/services.
- Runtime switches: feature flags select impls; avoid globals/singletons.
- Anti‑pattern: from my_impl import GlobalClient; GlobalClient.do() in business code.


## Side‑Effects & I/O

- Boundary functions should be small and thin (open file, call API, emit event).
- Core functions should be deterministic and easy to unit test (pure in → pure out).
- Idempotency: where feasible, make commands safe to retry.


## Naming & Structure

- Modules: verbs for commands (ingest.py, reconstruct.py), nouns for models (tome.py, digest.py).
- Functions: imperative verbs (generate_keywords, attach_anchor).
- Events: past‑tense facts (digest.created, jobs.failed).


## Testing Guide (Fast Feedback)

- Unit: pure core logic; no DB/network.
- Contract: adapters honor interfaces (fake + real); snapshot/golden tests for CLI.
- Integration: happy‑path workflows (ingest → keywords → quiz → digest).
- Fakes over Mocks: prefer simple fakes/stubs; mock only hard edges.

**Checklist**

- Unit tests for core logic
- Contract tests for providers/repos
- Golden CLI/TUI snapshots updated intentionally
- Error paths tested (not just happy paths)


## Async/Concurrency

- Prefer sync unless there’s real parallel I/O.
- Use the JobRunner + APScheduler for background/long‑running tasks.
- Keep async localized; don’t leak async into pure core.


## Logging & Telemetry

- Levels: DEBUG (developer), INFO (state changes), WARNING (recoverable issues), ERROR (failures), CRITICAL (system down).
- Structure logs (key=value) at boundaries; avoid noisy logs in tight loops.
- Respect privacy: redact secrets; opt‑in telemetry only.


## Data & Schema Micro‑Rules

- Migrations are append‑only; never edit old migrations.
- Use ULIDs for IDs; avoid meaning‑laden keys.
- Keep derived data denormalized only when measured wins exist.


## Performance Micro‑Heuristics

- First make it correct & clear. Optimize when it hurts and is measured.
- Prefer streaming/iterators for large files; avoid loading entire PDFs/audio into memory.


## Security Basics

Where applicable;

- Use SQLCipher via the provided engine; never bypass it.
- Key material goes through keychain; no custom crypto.
- Treat AI provider keys as secrets; never log or store in plaintext.


## PR & Code Review Checklist

- Function/module has a single clear purpose.
- Dependencies injected at edges; no hidden globals.
- Errors handled at command boundaries; domain exceptions used.
- Tests: unit + contract; golden snapshots intentional.
- Naming consistent with conventions.
- No gratuitous async; jobs used for long tasks.
- Logs are actionable; no secrets.


> Blocker labels: “Leaky boundary”, “God function”, “Hidden global”, “Adapter drift”.


## Examples (Before → After)

**Monolithic**

```
def run():
    # parses args, queries db, calls OpenAI, writes file, prints UI
    ...
```

**Refactored**

```
def command_run(args: Args, deps: Deps) -> ExitCode:
    data = load(args.source, deps.repo)
    items = generate_keywords(data, deps.ai)
    save(items, deps.repo)
    return present(items, deps.ui)
```


## Living Document

- Update this guide when a rule causes friction.
- Record exceptions via lightweight ADR notes in PRs ("we deviated because …").
- Shortlink mantra: Small seams, pure core, dirty edges.

