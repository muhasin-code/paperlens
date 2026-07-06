# PaperLens — Completed Milestones Index

This file is a **summary index** for completed milestones. Full details, plan content, and verification data live in the linked Markdown files under `plan/completed-milestones/`.

---

## Archived Phases

### Phase 0 — Project Foundation

Directory: `plan/completed-milestones/phase-0-project-foundation/`

| # | Milestone | Completed | Git ref | Summary | Key Deliverables | Detail |
|---|---|---|---|---|---|---|
| 0.1 | [GitHub Repository Setup](completed-milestones/phase-0-project-foundation/0.1-github-repository-setup.md) | 2026-06-09 | `48b9aaa` | Repo, license, README, issues, contributing guide, project-setup docs. | [x] Create repo [x] MIT license [x] CI skeleton | [Link](completed-milestones/phase-0-project-foundation/0.1-github-repository-setup.md) |
| 0.2 | [Local Development Environment](completed-milestones/phase-0-project-foundation/0.2-local-development-environment.md) | 2026-06-17 | — | Python 3.11 venv, Ollama + phi4-mini, Docker, .env.example, dev-environment docs. | [x] Python venv [x] Ollama [x] Docker [x] .env.example | [Link](completed-milestones/phase-0-project-foundation/0.2-local-development-environment.md) |
| 0.3 | [Project Scaffolding](completed-milestones/phase-0-project-foundation/0.3-project-scaffolding.md) | 2026-06-19 | — | Skeleton package, FastAPI app, requirements, Makefile, ruff, pre-commit, CI, docs. | [x] src/ package [x] Makefile [x] CI [x] ruff/pre-commit | [Link](completed-milestones/phase-0-project-foundation/0.3-project-scaffolding.md) |

### Phase 1 — Core RAG Pipeline

Directory: `plan/completed-milestones/phase-1-core-rag-pipeline/`

| # | Milestone | Completed | Git ref | Summary | Key Deliverables | Detail |
|---|---|---|---|---|---|---|
| 1.1 | [arXiv Data Ingestion](completed-milestones/phase-1-core-rag-pipeline/1.1-arxiv-data-ingestion.md) | 2026-06-25 | — | Idempotent ingestion of 500 cs.LG papers into metadata.jsonl + PDF corpus. | [x] ArxivFetcher [x] JSONL persistence [x] 500 papers | [Link](completed-milestones/phase-1-core-rag-pipeline/1.1-arxiv-data-ingestion.md) |

---

## How to Archive a New Milestone

1. Complete a milestone per `plan/milestone_plan.md`.
2. Create a new file in the matching `plan/completed-milestones/phase-{n}-{slug}/` directory using the individual milestone template (see any existing file for the pattern).
3. Add a summary row to the appropriate phase table in this file.
4. Update the corresponding milestone checkbox in `plan/blueprint.md`.
5. Replace `plan/milestone_plan.md` with the next milestone from `plan/blueprint.md`.
6. Move the GitHub Projects card to Done.
