# Contributing to PaperLens

Thank you for your interest in contributing to PaperLens.

PaperLens is an AI-powered research intelligence platform focused on document ingestion, semantic retrieval, citation aware workflows, and local-first AI tooling.

---

## Development Workflow

### Branching Strategy

All work should be performed in dedicated branches created from main.

### Branch Naming Convensions

| **Prefix** | **Purpose** | **Example** |
|------------|-------------|-------------|
| `feature/` | New functionality | `feature/arxiv-ingestion` |
| `fix/` | Bug fixes | `fix/pdf-parser-timeout` |
| `docs/` | Documentation updates | `docs/setp-guide` |
| `chore/` | Tooling, CI, dependencies | `chor/update-precommit` |


---

## Wrokflow Process

1. Create a branch from `main`
2. Keep commits focused and descriptive
3. Push the branch to GitHub
4. Open a Pull Request targeting `main`
5. Ensure CI passes (introduced in Milestone 0.3)
6. Use squash merge unless otherwise specified


---

## Pull Request Expectations

Each Pull Request (PR) should:

* Clearly explain what changed
* Explain why the change was necessary
* Link related issues where applicable
* Avoid unrelated modification
* Exclude secrets, credentials, or environment specific files

For ML/AI-related changes, include:

* Model impact summary
* Evaluation notes if applicable
* Or state:

*N/A - infrastructure tooling only*


---

## Code Style

### Python Version

* Python 3.11+

### Formatting & Linting

Formatting and linting will be enforced through:

* pre-commit
* Ruff
* Black

These tools will be configured during Milestone 0.3


---

## Environment & Configuration

Environment-specific values must never be committed.

Examples:

* `.env`
* API keys
* model credentials
* tokens

Use `.env.example` for shared configuration structre.


---

## Project Philosophy

PaperLens prioritizes:

* local-first AI workflows
* maintainable architecture
* reproducible developement environment
* modular engineering practices
* transparent documentation


---

## Documentation

When introducing major functionality:

* update relevent documentations
* add setup notes if required
* document architectural decisions where appropriate

Future developer setup instructions will be maintained in:

*docs/dev-environment.md*