# PaperLens - Project Setup Documentation

## Repository

| **Field** | **Values** |
|-----------|------------|
| Repository Name | `paperlens` |
| Web URL | `https://gitub.com/muhasin-code/paperlens` |
| Visibility | Public |
| Default Branch | `main` |


---

## Git Remote Configuration

## SSH Remote

git@github.com:muhasin-code/paperlense.git

## SSH Host Alias

PaperLens uses the SSH host alias:

github-personal

Configured in:

~/.ssh/config

Mapped SSH key:

~/.ssh/id_ed25519


---

## Push Workflow

Initial upstream setup:

```bash
git push -u origin main
```

Daily workflow afterwards:

```bash
git push
```

This repository uses SSH authentication instead of HTTPS.


---

## License

PaperLens is licensed under the MIT License.

## Why MIT

The MIT License was selected because it is:

* permissive
* portfolio-friendly
* industry-standard
* compatible with the planned technology stack

See:

[LICENSE](LICENSE.md)


---

## Branch Strategy

### Main Branch

The `main` branch represents the stable integration branch.

Direct commits to `main` should be avoided whenever possible.

All the work should be performed through dedicated branches.


---

### Branch Naming Conventions

| **Prefix** | **Purpose** |
|------------|-------------|
| `feature/` | New functionality |
| `fix/` | Bug fixes |
| `docs/` | Documentation |
| `chore/` | Tooling, CI, maintenance |


Examples:

`feature/pdf-ingestion`
`fix/embedding-timeout`
`docs/setup-guide`
`chore/precommit-hooks`


---

### Merge Strategy

PaperLens uses:

*Squash Merge*

Rationale:

* cleaner commit history
* easier milestone tracking 
* simpler rollback management

Additionl workflow details documented in: [CONTRIBUTING](CONTRIBUTING.md)


---

## Project Tracking

### GitHub Project Board

Project board:

`PaperLens Roadmap`

Tracks:
* milestones
* active work
* reviews
* completed tasks


---

### Workflow Columns

| **Column** |
|------------|
| Backlog |
| In Progress |
| Review |
| Done |


---

### Milestone Mapping

Milestones map directly to the implementation phases defined in: [blueprint](plan/blueprint.md)

Example:

| **Milestone** | **Phase** |
|---------------|-----------|
| Milestone 0.1 | Project Foundation |
| Milestone 0.2 | Development environment |
| Milestone 0.3 | Project Scaffolding |


---

### Label Conventions

| **Label** | **Purpose** |
|-----------|-------------|
| phase-0 | Foundation tasks |
| milestone | Major milestone work |
| documentation | Docs-only changes |
| blocked | Blocked work items |


---

## Planned Repository Layout

The repository structure will evolve during Milestone 0.3

Planned directories include:

src/
docs/
tests/
config/
scripts/
data/


---

## Planning Directory

The `plan/` directory contains:
* project blueprint


---

## Initial Commit

### Setup Phase

Initial repository setup includes:

* Git repository initialization
* SSH remote configuration
* MIT License
* `.gitignore`
* README.md
* GitHub labels
* GitHub project board
* CONTRIBUTING guide
* project setup documentation


---

### Repository Status

PaperLens is currently in:

Phase 0 - Project Foundation

Application developments has not yet started at this stage.