# Hybrid RAG — Work in Progress

A learning project exploring the components behind a retrieval-augmented generation (RAG) pipeline. The current implementation focuses on loading a PDF and splitting its pages into overlapping chunks.

## Current status

### Implemented
- Load page text from a PDF using `pypdf`.
- Convert pages into LangChain `Document` objects.
- Split documents with `RecursiveCharacterTextSplitter` using a 1,000-character chunk size and 200-character overlap.
- A separate model experiment using NVIDIA-hosted inference and Langfuse callbacks.

### Not implemented yet
- Dense embedding generation and sparse retrieval (the `src/embedding` modules are placeholders).
- Vector database indexing and retrieval.
- Hybrid score fusion, reranking, answer generation, and retrieval evaluation.
- End-to-end automated tests.

## Setup

Requirements: Python 3.13+, [uv](https://docs.astral.sh/uv/), and an NVIDIA API key for the model experiment.

```bash
git clone https://github.com/Dhairy4/Hybrid_RAG.git
cd Hybrid_RAG
uv sync
cp .env.example .env
```

Set `NVIDIA_API_KEY` in your local `.env`. Langfuse tracing can be configured with `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`.

## Run the current PDF chunking example

```bash
uv run python src/main.py
```

The script reads the example PDF at `src/Artifical Intelligence.pdf` and prints the number of chunks. Run commands from the repository root.

## Run the model/observability experiment

```bash
uv run python src/config/model.py
```

This makes a live hosted-model request and may incur provider usage. Configure credentials before running it.

## Next milestones

1. Add tests for PDF extraction and chunk boundaries.
2. Implement dense and sparse retrieval in separate modules.
3. Store document IDs, page numbers, and chunk metadata for traceable citations.
4. Add retrieval evaluation examples and compare retrieval configurations.
5. Build an end-to-end query endpoint only after the retrieval path is tested.

This repository is an active experiment, not a production-ready hybrid RAG service yet.
