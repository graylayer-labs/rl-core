## What does this PR do?
<!-- One paragraph. Link to the issue it closes if applicable: "Closes #123" -->

## Is this a breaking change?

- [ ] **Yes — I have labelled this PR `breaking`**
  - Algorithms or protocols affected: <!-- list them -->
  - Required migration or reruns: <!-- be specific -->
- [ ] No — fully backwards compatible

## Checklist

- [ ] `poetry run ruff check .` passes
- [ ] `poetry run ruff format --check .` passes
- [ ] `poetry run python -m rl_core.reproductions validate` passes
- [ ] `poetry run pytest tests/ -v` passes
- [ ] Claims, deviations, and result status remain accurate
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml` if this is a release PR
