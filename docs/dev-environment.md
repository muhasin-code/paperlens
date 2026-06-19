# PaperLens — Development Environment

## Overview

This document describes the local development environment used for PaperLens during Milestone 0.2 (Project Foundation).

The goal of this milestone is to establish and verify all runtime dependencies before application development begins.

This includes:

* Python 3.11+
* Project virtual environment
* Ollama local interface
* Docker and Docker Compose
* Hardware baseline
* Environment configuration

Project dependencies and application scaffolding will be introduced in Milestone 0.3.

---

# Hardware Baseline

## Machine

Dell Latitude 7490

| **Component**  | **Value **             |
|----------------|------------------------|
| CPU            | Intel Core i7-8650U    |
| Cores/Threads  | 4 Cores / 8 Threads    |
| Max Clock      | 4.2 GHz                |
| RAM            | 7.6 GiB                |
| Available Disk | 102 GiB                |
| GPU            | Intel UHD Graphics 620 |
| OS             | Ubuntu 22.04.5 LTS     |

## Notes

* No NVIDIA GPU is available.
* No CUDA acceleration is available.
* All inference is performed on CPU.
* SSD storage is sufficient for local models, vector databases, and document caches.

---

# Prerequisites Checklist

Before working on PaperLens, ensure the following are installed

* Python 3.11+
* Git
* SSH access to GitHub
* Ollama
* Docker Engine
* Docker Compose
* Approximately 20 GB free disk space

Repository access uses the SSH host alias:

```text
github-personal
```

See:

```text
docs/project-setup.md
```

---

# Python Environment

## Create Virtual Environment

From the repository root:

```bash
python3.11 -m venv .venv
```

## Activate

```bash
source .venv/bin/activate
```

## Verify

```bash
which python
python --version
```

Expected:

```text
Python 3.11.x
```

Application dependencies are intentionally deferred in Milestone 0.3.

---

# Ollama Setup

## Installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Enable service:

```bash
sudo systemctl enable ollama
sudo systemctl start ollama
```

---

## Development Model

Due to the system's 8 GB RAM limitation, a lightweight model is used during development:

```text
phi4-mini
```

Alternative:

```text
llama3.2:1b
```

The blueprint recommendation of `llama3.2:3b` may cause excessive memory pressure on this hardware.

---

## Pull Model

```bash
ollama pull phi4-mini
```

---

## Verify

```bash
ollama list
```

---

## Smoke Test

```bash
ollama run phi4-mini "Reply with exactly: PaperLens OK"
```

Expected output:

```text
PaperLens OK
```

---

## API Verification

```bash
curl http://localhost:11434/api/tags

curl http://localhost:11434/api/generate
```

Both endpoints should respond successfully.

---

## Memory Observation

Memory usage while running `phi4-mini`:

| **Metric**    | **Value** |
|---------------|-----------|
| Total RAM     | 7.6 GiB   |
| Used RAM      | 4.8 GiB   |
| Free RAM      | 172 MiB   |
| Available RAM | 2.1 GiB   |
| Swap Used     | 1.2 GiB   |

Observations:

* The system remains usable.
* Memory usage is high.
* Larger models should be avoided during development.

---

# Docker Setup

## Verify Installation

```bash
docker --version
docker compose version
```

---

## Verify Engine

```bash
docker run --rm hello-world
```

Expected:

```text
Hello from Docker!
```

---

## Docker Compose

PaperLens includes a placeholder:

```text
docker-compose.yml
```

This file will be expanded in later milestones.

Future services include:

* Langfuse
* Backend API
* Frontend UI
* Supporting infrastructure

---

# Environment Variables

PaperLens uses:

```text
.env.example
```

as the configuration template.

Create a local configuration:

```bash
cp .env.example .env
```

Never commit:

```text
.env
```

---

## Configuration Groups

| **Group**   | **Purpose**             |
|-------------|-------------------------|
| Application | Runtime settings        |
| Paths       | Data locations          |
| Ollama      | Local LLM configuration |
| Embeddings  | Retrieval models        |
| arXiv       | Data ingestion          |
| Fine-tuning | Model training          |
| Langfuse    | Observability           |
| Evaluation  | RAG metrics             |
| Prompts     | Prompt management       |

See:

```text
.env.example
```

for full configuration details.

---

# Verification Checklist

* [X] Python 3.11+ installed
* [X] Virtual environment created
* [X] Ollama installed
* [X] Local model downloaded
* [X] Ollama API reachable
* [X] Docker installed
* [X] Docker Compose installed
* [X] Hardware baseline recorded
* [X] `.env.example` created
* [X] Documentation completed

---

# Troubleshooting

## Ollama Not Running

Check:

```bash
sudo systemctl status ollama
```

Restart:

```bash
sudo systemctl restart ollama
```

---

## Port 11434 Already in Use

Check

```bash
sudo ss -tulpn | grep 11434
```

Stop conflicting process or restart Ollama.

---

## Docker Permission Denied

Add user to group:

```bash
sudo usermod -aG docker $USER
```

Log out and log in again.

---

## Out of Memory During Model Pull

Use:

```text
phi4-mini
```

or:

```text
llama3.2:1b
```

instead of larger models.

---

## Wrong Python Version

Verify

```bash
python3.11 --version
```

Recreate the virtual environment if necessary.

---

# Related Documentation

* `docs/project-setup.md`
* `.env.example`
* `plan/blueprint.md`
