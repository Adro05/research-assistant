# Research Assistant

An AI-powered research assistant for discovering, organizing, and analyzing academic literature.

## Vision

The Research Assistant is designed to help researchers move from paper discovery to evidence-grounded literature analysis in a single platform.

The system will combine:

- Academic paper discovery
- Project-based research organization
- PDF and URL ingestion
- Retrieval-Augmented Generation (RAG)
- Hybrid search and semantic retrieval
- Multimodal document understanding
- Knowledge-graph-enhanced retrieval
- Evidence verification
- Citation-aware responses

## Core Design Principle

The academic paper discovery layer is optional.

The core document ingestion and RAG pipeline must continue to function even when external search providers or APIs are unavailable.

Users can therefore build a research corpus through:

1. Academic search
2. Accessible paper URLs
3. Direct PDF uploads

If a paper is behind authentication, a subscription, or an institutional paywall, the system will not attempt to bypass the restriction. The user can access the paper through the official source and upload the PDF manually.

## Project-Based Research

Users can create independent research projects.

Each project maintains its own research context and retrieval scope.

The same paper or document may be associated with multiple projects without requiring unnecessary duplication of the underlying document.

## Planned Architecture

The system will progressively evolve from a core RAG application into a multimodal, graph-enhanced, and agentic research intelligence platform.

### Initial Technology Stack

- Python 3.12
- FastAPI
- Streamlit
- PostgreSQL
- Hugging Face
- Sentence Transformers
- FAISS

### Planned Extensions

- Semantic Scholar
- arXiv
- Hybrid retrieval
- Cross-encoder reranking
- Multimodal RAG
- Neo4j knowledge graph
- Agentic evidence verification
- Redis caching
- Asynchronous document processing
- Background workers
- Observability
- Horizontal scaling

## Core RAG Pipeline

The initial RAG pipeline will follow:

```text
PDF
  ↓
Text Extraction
  ↓
Chunking
  ↓
Hugging Face Embeddings
  ↓
FAISS
  ↓
Retrieval
  ↓
Relevant Evidence
  ↓
Hugging Face Generation
  ↓
Evidence-Grounded Answer
  ↓
Citation Mapping

## Project-Based Research

Users can create independent research projects.

Each project maintains its own research context and retrieval scope.

The same paper or document may be associated with multiple projects without requiring unnecessary duplication of the underlying document.
## Long-Term Retrieval Architecture

The target retrieval system will combine:

- Dense semantic retrieval
- Sparse keyword retrieval
- Knowledge graph retrieval
- Hybrid result fusion
- Cross-encoder reranking

This architecture will be introduced incrementally and evaluated at each stage.
## Evidence Verification

The long-term system will use specialized agents to improve research reliability.

The planned pipeline includes:

```text
User Query
    ↓
Retrieval Agent
    ↓
Evidence Pool
    ↓
Synthesis Agent
    ↓
Draft Answer
    ↓
Critic Agent
    ↓
Verification Agent
    ↓
Evidence Judge
    ↓
Final Answer

### 4. Then scalability

```markdown
## Scalability

The long-term architecture is designed with large-scale usage in mind.

Planned scalability mechanisms include:

- Redis caching
- Query caching
- Embedding caching
- Asynchronous processing
- Background workers
- Task queues
- Stateless API services
- Horizontal scaling

The 1M+ user target is a future scalability objective and will require benchmarking before any capacity claims are made.
## Development Philosophy

The project is being developed incrementally.

Each major component will be:

1. Implemented
2. Tested
3. Evaluated
4. Documented

before additional complexity is introduced.

The goal is not to maximize architectural complexity.

The goal is to build a reliable research system where every major component provides a measurable improvement in capability, retrieval quality, reliability, scalability, or user experience.
## Status

🚧 Active development