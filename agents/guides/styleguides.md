# Style Guides

## Python

- Generally follow standard PEP8 rules--even in tests!
 - Ruff is used for linting and formatting.
- Use Google style docstrings.

### Conventions and examples

```python
# Good: imports grouped, typed signatures, 4-space indents, clear names
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


def chunk(items: Iterable[int], size: int) -> list[list[int]]:
    """Split items into contiguous chunks of at most size.

    Args:
      items: Source integers to group.
      size: Maximum chunk length; must be positive.

    Returns:
      A list of chunks preserving original order.

    Raises:
      ValueError: If size is not positive.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    out: list[list[int]] = []
    buf: list[int] = []
    for x in items:
        buf.append(x)
        if len(buf) == size:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out


@dataclass
class Point:
    x: float
    y: float

    def dist2(self, other: "Point") -> float:
        dx, dy = self.x - other.x, self.y - other.y
        return dx * dx + dy * dy
```

```python
# Bad: ambiguous names, mixed indent, unclear exceptions, no typing
def c(a, b):
  if b<=0: raise Exception('bad')
  r=[]; t=[]
  for i in a: t.append(i); 
  if len(t)==b: r.append(t); t=[]
  if t: r.append(t)
  return r
```

### Docstrings (Google style)

```python
def normalize_email(email: str) -> str:
    """Normalize an email address to lowercase without surrounding spaces.

    Args:
      email: The input address which may contain spaces or mixed case.

    Returns:
      A lowercase, trimmed email address.
    """
    return email.strip().lower()
```

### Docstrings coverage

- Modules: Add a brief module-level docstring stating purpose and key concepts.
- Classes: Include a short summary and an Attributes section for key fields.
- Properties: Docstring the getter if computed or has side effects.
- Cross-link rationale: see Patterns and Architecture "Why/How/Example/Pitfalls" for organization guidance (guides/patterns_and_architecture.md).

### Ruff configuration (example)

```toml
# pyproject.toml
[tool.ruff]
line-length = 80
target-version = "py311"
select = [
  "E",  # pycodestyle
  "F",  # pyflakes
  "I",  # import order
  "UP", # pyupgrade
  "D",  # pydocstyle (Google style via convention below)
]
ignore = [
  "D203", # one-blank-line-before-class; prefer D211
]

[tool.ruff.pydocstyle]
convention = "google"
```

### Pytest conventions

- Structure: `tests/` with `unit/` and `integration/` subfolders when useful.
- Naming: files `test_*.py`; functions `test_<unit>_<behavior>`; fixtures in `conftest.py`.
- Fixtures: prefer factory-style fixtures; scope narrowly; avoid `autouse` except for env setup.
- Marks: use `@pytest.mark.slow`, `@pytest.mark.integration`, etc.; register in config.
- Assertions: use bare `assert`; prefer `pytest.raises` for exception checks.
- Minimal pyproject config example:

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
markers = [
  "slow: long-running tests",
  "integration: crosses process or network boundaries",
]
```


## Javascript and Typescript

The ff. utilities and the standards they check against are observed:

- https://github.com/standard/standard
- https://github.com/standard/ts-standard

### Module boundaries and imports

- Prefer absolute imports from the project root using path aliases; avoid deep relative paths.
- Order imports: standard library/built-ins, third-party, internal (domain-first), then relatives.
- Group with blank lines between groups; prefer named exports over default.
- Use barrels (`index.ts`) sparingly; avoid cross-barrel cycles and re-exporting deep internals.

### JavaScript (standard) example

```js
// 2 spaces, no semicolons, single quotes, spacing around keywords
export function sum (xs) {
  if (!Array.isArray(xs)) return 0
  return xs
    .filter(x => typeof x === 'number')
    .reduce((acc, x) => acc + x, 0)
}
```

### TypeScript (ts-standard) example

```ts
export interface User {
  id: string
  email: string
  active: boolean
}

export function activate (u: User): User {
  if (u.active) return u
  return { ...u, active: true }
}

export async function fetchUser (id: string): Promise<User | null> {
  const res = await fetch(`/api/users/${id}`)
  if (!res.ok) return null
  return await res.json() as User
}
```

### Async and error handling

- Wrap `fetch` with a small helper to set base URL, timeouts, and JSON handling.
- Catch errors at boundaries (UI handlers, API adapters); keep inner logic mostly error-agnostic.
- Prefer discriminated unions or `Result`-style types for recoverable errors.

```ts
// Minimal fetch wrapper with timeout and JSON
export async function http<T>(input: RequestInfo, init: RequestInit = {}): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 10_000)
  try {
    const res = await fetch(input, { ...init, signal: controller.signal, headers: {
      'content-type': 'application/json',
      ...(init.headers ?? {})
    }})
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`)
    }
    return await res.json() as T
  } finally {
    clearTimeout(timeout)
  }
}
```

### Minimal tsconfig example

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "noImplicitAny": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "lib": ["ES2020", "DOM"],
    "baseUrl": ".",
    "paths": {
      "@app/*": ["src/*"]
    }
  },
  "include": ["src"]
}
```

### Common pitfalls

- Avoid implicit any in TS; type all public surfaces.
- Prefer named exports; avoid default unless a module has a single export.

## Indentation

- Python will be indented using 4 spaces
- scripts (JS/TS/SH), data formats (JSON/YAML), markup languages (MD/HTML/CSS), configs, etc
  will use 2 spaces.

### Examples

```python
# Python: 4 spaces
def greet(name: str) -> str:
    if not name:
        return "Hello"
    return f"Hello, {name}"
```

```yaml
# YAML: 2 spaces
version: '3'
services:
  web:
    image: myapp:latest
    ports:
      - '8080:8080'
```

```json
// JSON: 2 spaces
{
  "name": "myapp",
  "version": "1.0.0"
}
```

```sh
# Shell: 2 spaces
if [ -f package.json ]; then
  echo "Found"
fi
```

```html
<!-- HTML/CSS: 2 spaces -->
<div class="card">
  <h3>Title</h3>
  <p>Content</p>
</div>

<style>
  .card {
    padding: 8px;
  }
</style>
```

## Documentation

- Use semantic linebreaks
- With markdown;
  - start with a title using a H1 heading
  - use H2 for main sections
  - never skip a heading
  - limit headings to H4
- Write to reduce cognitive load by layering information
  and allowing the reader to decide if they want to read further;
  - start with a short rationale or hook
  - layout what the reader will expect if they read further
  - unpack the information clearly and succinctly
  - know when to break these rules depending on context
  - see examples in Patterns and Architecture and Workflow guides (guides/patterns_and_architecture.md, guides/workflow.md)

### Semantic line breaks

Write sentences with deliberate line breaks at natural phrase boundaries,
not at hard wrap length.
This makes diffs smaller and reviews easier.

```md
Caching Strategy
===

We cache GET responses for 5 minutes
to reduce load on the upstream API.
This balances freshness with performance
for typical browsing sessions.
```

### Heading levels and examples

```md
Feature Toggle Rollout
===

## Overview

### Goals

#### Metrics

## Implementation
```

- Do not jump from H2 to H4; progress one level at a time.
- Keep headings concise and parallel in structure.

### Layering information

```md
Background Jobs Strategy
===

Offload non-blocking work to improve latency.

Key points:
- Queue selection and trade-offs
- Retry and idempotency guidelines
- Monitoring and alerting

Details and examples:
Jobs must be idempotent.
Prefer at-least-once delivery with deduplication keys.
Record metrics for enqueue time, run time, and failures.
...
```

## Cross-language conventions

- Maximum line length: 120 characters (code and prose), strict.
- Trailing commas: prefer in multi-line constructs to minimize diff noise (where language allows).
- Files should end with a newline; avoid trailing whitespace.

### EditorConfig (example)

```ini
# .editorconfig
root = true

[*]
charset = utf-8
end_of_line = lf
insert_final_newline = true
trim_trailing_whitespace = true
indent_style = space
indent_size = 2

[*.py]
indent_size = 4

[Makefile]
indent_style = tab
```
