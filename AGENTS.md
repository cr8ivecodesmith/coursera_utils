# AGENTS Guidelines

## Directory Overview

- `agents/guides/`: Core reference material on engineering practice, patterns, style, and workflow.
- `agents/tasks/feat/`: Feature work specs, each folder numbered with a slug and containing `spec.md` plus follow-on artifacts.
- `agents/tasks/patch/`: Patch or follow-up efforts (refactors, hardening), organized the same way.

## Guides

- `agents/guides/engineering-guide.md`: Micro-design principles covering seam-first architecture, dependency injection, testing focus, and other day-to-day engineering defaults.
- `agents/guides/patterns-and-architecture.md`: Deep dive on organization patterns, composition vs. inheritance guidance, module layout, logging, and anti-patterns.
- `agents/guides/styleguides.md`: Language and tooling conventions (Python, docs, front-end snippets), docstring guidance, Ruff configuration, pytest patterns, and semantic line break practices.
- `agents/guides/workflow.md`: Canonical collaboration loop with the human, including task lifecycle, templates, and expectations for history updates and reviews.
- `agents/guides/workflow-extras/`: Template library used by `workflow.md`.
  - `codereview-tpl.md`: Structure for optional review deliverables.
  - `implementation(-mini)-tpl.md`: Full and lightweight execution logs.
  - `spec(-mini)-tpl.md`: Full and lightweight task specs.

Use these references when scoping new work, reviewing deliverables, or aligning implementation details with existing expectations.
