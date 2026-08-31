# Project Specification

## 1. Project Goal

Build an AI-powered research assistant that allows users to discover academic literature, organize papers into research projects, build project-specific RAG corpora, and perform evidence-grounded research analysis.

---

## 2. Research Projects

Users can create multiple independent research projects.

Each project has:

- Unique project name
- Creation timestamp
- Last updated timestamp

Each project maintains its own research corpus and conversations.

---

## 3. Paper Discovery

The system should provide a unified academic paper discovery experience.

### Initial Providers

- Semantic Scholar
- arXiv

### Future Providers

- IEEE Xplore
- Additional academic sources

Google Scholar scraping is not part of the core architecture.

### Search Behavior

Search results should primarily be ranked by relevance.

The initial relevance system should combine:

- Keyword relevance
- Semantic embedding similarity
- Citation signal

Ranking weights should remain configurable and be evaluated experimentally.

### Filters

Publication year filtering is required.

The system should support selecting a publication-year range.

---

## 4. Search Fault Tolerance

External search providers must not be single points of failure.

If one provider fails:

- Other available providers should continue returning results.
- The failed provider should be reported gracefully.

If all external providers fail:

- The search interface should report that search is temporarily unavailable.
- Existing projects and RAG functionality must continue working.

---

## 5. Paper Metadata

A paper may exist in the system without an accessible full-text document.

Paper metadata may include:

- Title
- Abstract
- Authors
- Publication date
- DOI
- Source
- Original source URL
- PDF URL
- Citation count

---

## 6. Research Library

Users can save papers to their research projects.

The same paper may belong to multiple research projects.

Saving a paper does not automatically add its contents to the RAG corpus.

---

## 7. Document Ingestion

Documents can enter the system through:

### PDF

Users can directly upload a PDF.

### URL

Users can provide a paper URL.

### Search

A paper discovered through an integrated academic provider can be added to a project.

---

## 8. Restricted Papers

The system must not bypass:

- Authentication
- Paywalls
- Subscription requirements
- Institutional access restrictions

If a URL cannot provide accessible full text, the system should:

1. Preserve available paper metadata.
2. Explain that full-text access may require login, subscription or institutional access.
3. Provide the official source link where available.
4. Allow the user to obtain the paper through legitimate access.
5. Allow the user to upload the PDF afterward.
6. Allow cancellation.

---

## 9. Project-Specific RAG

Every research project must have an isolated retrieval space.

Documents can be reused across projects.

The same processed document should not need to be duplicated merely because it belongs to multiple projects.

A project's RAG system must only retrieve evidence belonging to that project's corpus.

---

## 10. RAG Pipeline

The initial core pipeline is:

PDF
→ Text Extraction
→ Chunking
→ Embeddings
→ FAISS
→ Retrieval
→ Hugging Face Generation
→ Evidence-Grounded Answer
→ Citations

---

## 11. Evidence

Retrieved information should preserve its relationship to the original document.

Evidence should retain information such as:

- Paper
- Document
- Chunk
- Page
- Section
- Content
- Source URL
- Modality

---

## 12. Paper Analysis

A paper overview should provide, where available:

- Authors
- Publication date
- Abstract
- Introduction
- Methodology summary
- Methodology flow representation where reliable
- Results

Methodology summaries must be based on actual paper content and should not invent information.

---

## 13. Multimodal RAG

The target architecture should support:

- Text
- Tables
- Figures
- Images
- Mathematical content where practical

Multimodal evidence should preserve relationships between figures/tables and their surrounding paper context.

---

## 14. Knowledge Graph

The target architecture includes a knowledge graph representing relationships such as:

- Paper → Author
- Paper → Method
- Paper → Dataset
- Paper → Citation
- Method → Dataset
- Method → Result

Neo4j is the planned graph technology.

---

## 15. Agentic Evidence Verification

The target architecture includes specialized agents for:

- Evidence retrieval
- Answer synthesis
- Claim criticism
- Fact verification
- Evidence judging

The system should identify claims as:

- Supported
- Partially supported
- Unsupported
- Conflicting

The system should not present arbitrary numerical confidence values as factual certainty.

---

## 16. Scalability

The target architecture should be designed for horizontal scaling.

Future scalability mechanisms include:

- Redis caching
- Query caching
- Embedding caching
- Computation caching
- Asynchronous document processing
- Background workers
- Task queues
- Stateless API services

The 1M+ user requirement is a future scalability target, not an MVP dependency.

---

## 17. Core Architectural Principle

No external academic search provider should be capable of bringing down the core document/RAG functionality.

The system must degrade gracefully rather than fail completely.