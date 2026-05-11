---
name: code-style
description: Universal code style and formatting rules for Python and TypeScript/JavaScript. Use when writing, reviewing, or refactoring code to ensure consistency across languages.
---

# Code Style

## Python

- Indentation: 4 spaces
- Max line length: 100 characters
- Imports: sorted alphabetically, grouped by stdlib / third-party / local
- Docstrings: Google style
- Type hints: required for all function signatures

## TypeScript / JavaScript

- Indentation: 2 spaces
- Quotes: single
- Semicolons: required
- Trailing commas: always

## Naming

- Variables / functions: `camelCase` (JS/TS) or `snake_case` (Python)
- Constants: `UPPER_SNAKE_CASE`
- Classes: `PascalCase`
- Private members: `_prefixWithUnderscore`

## Comments

- `TODO:` describe what needs to be done
- `FIXME:` describe the problem
- `NOTE:` for important clarifications
