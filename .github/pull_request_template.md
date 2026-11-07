<!-- Title format: type(scope): summary [K-x.yy] -->

## Task

**Task ID:** K-

## Done when

<!-- Paste the "Done when" text for this task as a checklist. -->

- [ ]

## Checklist

- [ ] One subtask, one commit; the subject ends with the task ID
- [ ] CHANGELOG.md updated if the change is user-visible
- [ ] Tests ship with the code; `uv run pytest -m "not slow and not gpu"` passes
- [ ] `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy src` pass
- [ ] No thresholds or `λ` hard-coded in code (they live in versioned YAML)
- [ ] Backtests use only data vintages available at each cutoff (no lookahead)
- [ ] Any figure showing `C(t)` also shows `H(t)` and the recommended responses
