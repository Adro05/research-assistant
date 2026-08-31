# Research Assistant — System Architecture

## 1. Architectural Vision

The Research Assistant is designed as a modular, production-oriented research intelligence platform.

The system will initially be implemented as a modular monolith to keep development manageable. Individual modules will maintain clear boundaries so that they can later be extracted, scaled, or replaced independently.

The long-term architecture is designed around:

- Academic paper discovery
- Project-based research organization
- Document ingestion
- Project-isolated RAG
- Hybrid retrieval
- Multimodal RAG
- Knowledge-graph-enhanced retrieval
- Agentic evidence verification
- Citation-aware generation
- Caching
- Asynchronous processing
- Observability
- Horizontal scalability

The system must be developed incrementally. Advanced components must not be introduced before the core system is stable.
# 2. Architectural Principles

## 2.1 Modular Design

The initial implementation will use a modular monolith rather than immediately splitting the system into microservices.

Modules should have clearly defined responsibilities and interfaces.

Potential future services include:

- Search Service
- Document Processing Service
- Retrieval Service
- Generation Service
- Knowledge Graph Service
- Evaluation Service

These should only become independent services when scale or maintainability justifies the separation.

---

## 2.2 Graceful Degradation

External academic search providers must never become a single point of failure.

If an external provider becomes unavailable:

1. Other providers should continue operating.
2. The unavailable provider should be marked as failed.
3. The user should receive a clear status message.
4. Core project, document, and RAG functionality must remain available.

The system must never require academic search APIs for the local PDF-based RAG pipeline to function.

---

## 2.3 Project Isolation

Every research project has an independent research context.

A document may belong to multiple projects.

However, retrieval must always respect project membership.

Therefore:

```text
Project A
    |
    +---- Document X
    |
    +---- Document Y


Project B
    |
    +---- Document X
    |
    +---- Document Z
    
# 3. High-Level System Architecture

```text
                              USER
                                |
                                v
                    +-----------------------+
                    |   Streamlit Frontend  |
                    +-----------+-----------+
                                |
                           HTTPS / API
                                |
                                v
                    +-----------------------+
                    |     FastAPI Backend   |
                    +-----------+-----------+
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
       Search Module      Project Module     Document Module
             |                  |                  |
             v                  v                  v
      External APIs         PostgreSQL        Object Storage
                                                  |
                                                  v
                                         Document Processing
                                                  |
                                  +---------------+---------------+
                                  |               |               |
                                  v               v               v
                               Text            Tables          Figures
                                  |               |               |
                                  +---------------+---------------+
                                                  |
                                                  v
                                         Evidence / Chunks
                                                  |
                                  +---------------+---------------+
                                  |               |               |
                                  v               v               v
                               FAISS           Neo4j          PostgreSQL
                                  |               |               |
                                  +---------------+---------------+
                                                  |
                                                  v
                                         Retrieval Engine
                                                  |
                                  +---------------+---------------+
                                  |               |               |
                                  v               v               v
                              Dense Search    Sparse Search   Graph Search
                                  |               |               |
                                  +---------------+---------------+
                                                  |
                                                  v
                                            Hybrid Retrieval
                                                  |
                                                  v
                                             Reranking
                                                  |
                                                  v
                                          Evidence Pool
                                                  |
                                                  v
                                         Agent Pipeline
                                                  |
                                  +---------------+---------------+
                                  |               |               |
                                  v               v               v
                             Synthesis          Critic       Verification
                                Agent            Agent           Agent
                                  |               |               |
                                  +---------------+---------------+
                                                  |
                                                  v
                                            Evidence Judge
                                                  |
                                                  v
                                         HF Generation Model
                                                  |
                                                  v
                                         Citation Mapping
                                                  |
                                                  v
                                           Final Response
```
# 4. Frontend Layer

## Technology

Streamlit

## Responsibilities

The frontend is responsible for presentation and user interaction.

It will eventually provide:

- Project creation
- Project selection
- Paper discovery
- Search filters
- Paper details
- Paper saving
- PDF upload
- URL submission
- Document processing status
- Research chat
- Evidence display
- Citations
- Methodology summaries
- Figures and tables
- Verification results

Business logic must not be embedded directly into Streamlit pages.

The frontend should communicate with the backend through defined interfaces.

---

# 5. API / Backend Layer

## Technology

FastAPI

## Responsibilities

The backend acts as the main application layer.

Responsibilities include:

- API routing
- Request validation
- Response serialization
- Authentication in a future phase
- Project management
- Document management
- Search orchestration
- RAG orchestration
- Error handling
- Service coordination

The backend should remain stateless wherever practical.

Persistent state belongs in dedicated storage systems.
# 6. Database Layer

## Primary Database

PostgreSQL

## Responsibilities

PostgreSQL stores structured application metadata.

It will eventually contain entities such as:

- Users
- Projects
- Papers
- ProjectPapers
- Documents
- ProjectDocuments
- DocumentChunks
- Conversations
- Messages

---

# 7. Database Relationships

The conceptual relationship model is:

```text
User
 |
 +----< Project
          |
          +----< ProjectPaper >---- Paper
          |
          +----< ProjectDocument >---- Document
                                         |
                                         +----< DocumentChunk
          |
          +----< Conversation
                       |
                       +----< Message
```

A many-to-many relationship exists between:

```text
Projects ↔ Papers
```

and:

```text
Projects ↔ Documents
```

This allows the same paper or document to participate in multiple research projects.

A document should be stored and processed only once where possible, even if it is associated with multiple projects.

Project membership determines whether that document is eligible for retrieval within a particular project's RAG context.
# 8. Paper Discovery Architecture

The discovery system is intentionally decoupled from document ingestion and RAG.

Initial providers:

- Semantic Scholar
- arXiv

Future providers may include:

- IEEE Xplore
- Additional academic databases

Google Scholar scraping is not part of the core architecture.

The discovery layer must be treated as an optional subsystem. Failure of the discovery layer must not prevent users from uploading PDFs, ingesting accessible URLs, or using the RAG pipeline.

---

## Search Flow

```text
User Query
    |
    v
Search Service
    |
    +-------------------+
    |                   |
    v                   v
Semantic Scholar       arXiv
    |                   |
    +---------+---------+
              |
              v
      Result Normalization
              |
              v
          Deduplication
              |
              v
       Relevance Ranking
              |
              v
         Search Results
```

---

# 9. Search Ranking

Initial ranking should combine three signals:

- Keyword relevance
- Semantic embedding similarity
- Citation signal

Conceptually:

```text
Final Score =
    w1 × Keyword Score
  + w2 × Semantic Similarity
  + w3 × Citation Score
```

The ranking weights should remain configurable and should be evaluated experimentally.

Relevance should be prioritized over simply sorting results by citation count.

Citation count should act as a supporting signal rather than the primary ranking criterion.

---

## Publication Year Filtering

Search results must support publication-year filtering.

Users should be able to specify:

- Minimum publication year
- Maximum publication year

This allows users to restrict results to a relevant research period, such as recent papers.

---

## Future Search Improvements

Potential future improvements include:

- Query expansion
- Query decomposition
- Hybrid retrieval
- Cross-encoder reranking
- Field-aware ranking
- Author relevance
- Venue relevance
- Recency weighting

These should only be introduced after evaluating the initial ranking system.
# 10. Document Ingestion Architecture

Documents can enter the system through three primary paths:

1. Direct PDF upload
2. Accessible paper URL
3. Paper discovered through an integrated academic search provider

All three paths should eventually converge into the same document processing pipeline.

```text
                    Document Source
                         |
          +--------------+--------------+
          |              |              |
          v              v              v
     PDF Upload      Accessible URL   Search Result
          |              |              |
          +--------------+--------------+
                         |
                         v
                 Document Ingestion
                         |
                         v
                Document Validation
                         |
                         v
                Document Processing
```

The ingestion layer is independent from the academic discovery layer.

Therefore, the RAG system must continue to function even when external search providers are unavailable.

---

# 11. Restricted Paper Handling

The system must never attempt to bypass:

- Authentication
- Paywalls
- Subscription requirements
- Institutional access controls
- Other access restrictions

When a user provides a URL whose full text cannot be accessed, the system should fail gracefully.

The system should:

1. Preserve any available paper metadata.
2. Detect that the full text could not be retrieved.
3. Explain that the paper may require authentication, subscription, or institutional access.
4. Provide the official source link where available.
5. Give the user an option to access the paper through the official website.
6. Give the user the option to cancel.
7. Allow the user to upload the PDF manually instead.

Conceptually:

```text
User submits URL
       |
       v
Attempt legitimate retrieval
       |
       +----------------------+
       |                      |
    Accessible            Restricted
       |                      |
       v                      v
Continue ingestion       Preserve metadata
                              |
                              v
                     Explain access issue
                              |
                 +------------+------------+
                 |            |            |
                 v            v            v
             Official      Upload PDF    Cancel
              Website
```

The system must not scrape or circumvent protected content.

---

# 12. Document Validation

Before processing, uploaded or retrieved documents should be validated.

Validation should eventually include:

- File type
- File size
- PDF validity
- Content availability
- Duplicate detection
- Basic metadata extraction

Invalid documents should produce a clear user-facing error rather than causing the entire application to crash.

---

# 13. Document Processing Pipeline

The target document processing pipeline is:

```text
PDF / Accessible URL
        |
        v
Document Validation
        |
        v
Text Extraction
        |
        +------------------+
        |                  |
        v                  v
      Tables            Figures
        |                  |
        +--------+---------+
                 |
                 v
        Document Normalization
                 |
                 v
             Chunking
                 |
                 v
         Evidence Objects
```

The processing pipeline should preserve document structure wherever possible.

Important metadata includes:

- Page number
- Section
- Paragraph position
- Figure number
- Table number
- Caption
- Source document
# 14. Evidence Objects

The system should use a common evidence representation throughout the RAG pipeline.

An evidence object may contain:

```text
Evidence
├── paper_id
├── document_id
├── chunk_id
├── project_id
├── page
├── section
├── modality
├── content
├── source_url
└── metadata
```

Supported modalities include:

- TEXT
- TABLE
- FIGURE
- IMAGE

Every piece of retrieved evidence should remain traceable to its original document.

This allows the same evidence representation to be used by:

- Retrieval
- Reranking
- Generation
- Citation mapping
- Agentic verification
- Multimodal processing

---

# 15. Core RAG Architecture

The initial RAG system is intentionally simple.

```text
PDF
 |
 v
Text Extraction
 |
 v
Chunking
 |
 v
Hugging Face Embeddings
 |
 v
FAISS
 |
 v
Retrieval
 |
 v
Relevant Evidence
 |
 v
Hugging Face Generation
 |
 v
Evidence-Grounded Answer
 |
 v
Citation Mapping
```

The initial implementation should prioritize correctness, traceability, and testability rather than advanced retrieval complexity.

---

# 16. Embedding Layer

Sentence Transformers / Hugging Face models will initially be used for embedding generation.

The embedding pipeline is:

```text
Document Chunk
      |
      v
Embedding Model
      |
      v
Embedding Vector
      |
      v
FAISS Index
```

The embedding model and its version should be recorded so that embeddings can be reproduced when required.

Embedding generation should eventually be cached because document embeddings are expensive and are deterministic for a fixed model and input.

---

# 17. Vector Storage

FAISS will initially provide vector similarity search.

PostgreSQL stores the metadata required to interpret FAISS results.

The relationship is:

```text
FAISS Vector
    |
    v
Chunk ID
    |
    v
PostgreSQL
    |
    v
Document / Page / Section / Project
```

FAISS must not become the only source of document metadata.

Vector search results must be mapped back to the corresponding document chunks.

---

# 18. Project-Specific Retrieval

Each project has an independent retrieval namespace.

When a user asks a question inside a project:

```text
User Query
    |
    v
Project Context
    |
    v
Eligible Project Documents
    |
    v
Retrieval
```

Only documents associated with that project are eligible for retrieval.

A document shared across multiple projects may be indexed once where possible, while project membership determines whether the document can be retrieved within a specific project.

This separation is important for both correctness and future scalability.
# 19. Multimodal RAG

Academic papers contain information beyond plain text.

The target system should support multiple modalities while preserving their relationships to the original document.

Supported information types should include:

- Text
- Tables
- Figures
- Images
- Mathematical content where practical

The multimodal pipeline should not treat extracted figures or tables as isolated objects.

Each modality should retain its relationship to:

- The source paper
- The source document
- Page number
- Section
- Figure or table number
- Caption
- Surrounding text

---

# 20. Multimodal Document Representation

The target architecture is:

```text
                       Document
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
         Text            Tables          Figures
          |                |                |
          v                v                v
     Text Embedding   Table Representation  Image/Multimodal
                                             Representation
          |                |                |
          +----------------+----------------+
                           |
                           v
                    Evidence Objects
```

All modalities should ultimately map to a common evidence representation.

For example:

```text
Figure
├── document_id
├── page
├── figure_number
├── caption
├── image_reference
├── surrounding_text
└── embedding
```

Similarly, a table should preserve its structural information:

```text
Table
├── document_id
├── page
├── table_number
├── caption
├── rows
├── columns
├── surrounding_text
└── representation
```

---

# 21. Multimodal Retrieval

The target retrieval architecture should allow queries to retrieve evidence from multiple modalities.

```text
                         User Query
                              |
                              v
                       Query Processing
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
          Text Search    Table Search     Figure Search
             |                |                |
             +----------------+----------------+
                              |
                              v
                       Evidence Fusion
                              |
                              v
                         Reranking
                              |
                              v
                      Multimodal Evidence
                              |
                              v
                         Generation
```

A textual question may therefore retrieve:

- A paragraph explaining a method
- A table containing experimental results
- A figure showing the architecture

The final response should preserve the source relationships between these pieces of evidence.

---

# 22. Multimodal RAG Development Strategy

Multimodal functionality should be introduced only after the text-based RAG pipeline is stable.

The intended progression is:

```text
Phase 1
Text RAG
   |
   v
Phase 2
Tables
   |
   v
Phase 3
Figures / Images
   |
   v
Phase 4
Unified Multimodal Retrieval
```

The initial MVP should not depend on multimodal processing to answer questions from ordinary text-based PDFs.
# 23. Knowledge Graph Architecture

The target architecture includes a knowledge graph for representing relationships between entities found in academic literature.

Neo4j is the planned graph database.

The knowledge graph will complement vector-based retrieval rather than replace it.

---

# 24. Knowledge Graph Entities

The graph may contain entities such as:

- Paper
- Author
- Method
- Dataset
- Result
- Institution
- Research Topic
- Citation

Example relationships include:

```text
Paper
 |
 +---- AUTHORED_BY ------> Author
 |
 +---- PROPOSES ---------> Method
 |
 +---- EVALUATES_ON -----> Dataset
 |
 +---- REPORTS ----------> Result
 |
 +---- CITES -------------> Paper
```

Additional relationships may include:

```text
Method ---- IMPROVES_UPON ----> Method

Method ---- EVALUATED_ON -----> Dataset

Paper ---- CONTAINS -----------> Method

Method ---- PRODUCES ----------> Result

Paper ---- RELATED_TO ---------> Research Topic
```

The exact ontology should evolve based on evaluation rather than being unnecessarily complex from the beginning.

---

# 25. Knowledge Graph Construction

The target graph construction pipeline is:

```text
Processed Document
        |
        v
Entity Extraction
        |
        v
Relationship Extraction
        |
        v
Entity Resolution
        |
        v
Knowledge Graph
        |
        v
Neo4j
```

Extracted entities and relationships must remain traceable to their source documents.

For example:

```text
Method
   |
   +---- source_paper_id
   +---- source_document_id
   +---- source_page
   +---- source_chunk_id
```

This allows graph-derived information to be connected back to the original evidence.

---

# 26. Graph-Enhanced Retrieval

Knowledge graph retrieval will complement vector and keyword retrieval.

Vector retrieval answers questions such as:

> What content is semantically similar to this query?

Graph retrieval answers questions such as:

> What entities and relationships are connected to this concept?

The combined architecture is:

```text
                         User Query
                              |
                              v
                       Query Processing
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
      Dense Search        Sparse Search       Graph Search
       Embeddings             BM25                Neo4j
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                        Result Fusion
                              |
                              v
                     Hybrid Retrieval
                              |
                              v
                         Reranking
                              |
                              v
                       Evidence Pool
```

---

# 27. Graph and Vector Retrieval Relationship

The graph and vector stores have different responsibilities.

```text
FAISS
 |
 +---- Semantic similarity
 +---- Chunk retrieval
 +---- Dense retrieval


Neo4j
 |
 +---- Entity relationships
 +---- Citation relationships
 +---- Method relationships
 +---- Dataset relationships
 +---- Multi-hop traversal
```

PostgreSQL remains responsible for structured application metadata.

The three systems therefore serve complementary purposes:

```text
PostgreSQL
    → Application state and metadata

FAISS
    → Dense vector retrieval

Neo4j
    → Relationships and graph traversal
```

---

# 28. Graph Retrieval Development Strategy

Neo4j should be introduced only after the core text-based RAG system and retrieval evaluation are stable.

The intended progression is:

```text
Core Text RAG
      |
      v
Hybrid Retrieval
      |
      v
Retrieval Evaluation
      |
      v
Knowledge Graph Construction
      |
      v
Graph Retrieval
      |
      v
Graph + Vector Fusion
```

The initial RAG system must remain functional if Neo4j is unavailable.

If graph retrieval fails:

```text
Neo4j unavailable
       |
       v
Graph retrieval disabled
       |
       v
Vector / keyword retrieval continues
```
# 29. Agentic Evidence Verification

The target architecture includes a multi-agent evidence verification pipeline.

The purpose of the agent layer is not to make the system unnecessarily complex. Its purpose is to improve reliability by separating:

- Evidence retrieval
- Answer synthesis
- Criticism
- Claim verification
- Evidence judgment

Each agent should have a clearly defined responsibility.

---

# 30. Retrieval Agent

The Retrieval Agent is responsible for identifying and collecting relevant evidence.

Responsibilities include:

- Interpreting the user's research question
- Identifying the information required to answer the question
- Querying available retrieval systems
- Selecting relevant evidence
- Returning evidence with source metadata

The Retrieval Agent should not generate the final research answer.

---

# 31. Synthesis Agent

The Synthesis Agent constructs a candidate answer from the retrieved evidence.

Responsibilities include:

- Combining relevant evidence
- Constructing a coherent response
- Preserving source attribution
- Avoiding unsupported claims
- Identifying areas where evidence is insufficient

The output of this stage is a draft answer rather than the final answer.

---

# 32. Critic Agent

The Critic Agent challenges the draft answer.

Responsibilities include:

- Identifying unsupported claims
- Identifying missing evidence
- Detecting reasoning gaps
- Detecting overgeneralization
- Identifying weak comparisons
- Identifying claims that require additional verification

The Critic Agent should not automatically assume that the draft answer is correct.

---

# 33. Verification Agent

The Verification Agent checks individual claims against the available evidence.

Responsibilities include:

- Checking claims against source passages
- Verifying numerical values
- Checking source attribution
- Detecting contradictions
- Identifying claims that cannot be supported by the retrieved evidence

Verification should be performed against actual retrieved evidence rather than relying solely on another model's opinion.

---

# 34. Evidence Judge

The Evidence Judge determines the support status of claims.

Claims should be classified as:

```text
SUPPORTED
PARTIALLY_SUPPORTED
UNSUPPORTED
CONFLICTING
```

Definitions:

### SUPPORTED

The available evidence directly supports the claim.

### PARTIALLY_SUPPORTED

The evidence supports part of the claim but does not fully establish it.

### UNSUPPORTED

No sufficient evidence was found to support the claim.

### CONFLICTING

Available sources provide contradictory evidence.

The system should surface conflicting evidence rather than silently choosing one source.

---

# 35. Agentic Verification Pipeline

The target pipeline is:

```text
                         User Query
                              |
                              v
                       Retrieval Agent
                              |
                              v
                         Evidence Pool
                              |
                              v
                       Synthesis Agent
                              |
                              v
                         Draft Answer
                              |
                              v
                         Critic Agent
                              |
                              v
                      Verification Agent
                              |
                              v
                       Evidence Judge
                              |
                 +------------+------------+
                 |                         |
                 v                         v
          Evidence Sufficient       Evidence Problems
                 |                         |
                 |                         v
                 |                  Additional Retrieval
                 |                         |
                 +------------<------------+
                 |
                 v
           Final Generation
                 |
                 v
           Citation Mapping
                 |
                 v
            Final Answer
```

---

# 36. Claim-Level Verification

The system should eventually decompose generated responses into individual claims.

Conceptually:

```text
Draft Answer
     |
     v
Claim Extraction
     |
     +--------+--------+--------+
     |        |        |        |
     v        v        v        v
 Claim 1   Claim 2   Claim 3   Claim 4
     |        |        |        |
     v        v        v        v
 Evidence  Evidence  Evidence  Evidence
     |        |        |        |
     +--------+--------+--------+
              |
              v
        Evidence Judge
              |
              v
       Claim Classifications
```

This enables the final answer to distinguish between well-supported conclusions and areas where evidence is incomplete.

---

# 37. Contradictory Evidence

Academic literature may contain conflicting findings.

The system should not automatically treat disagreement as an error.

Instead:

```text
Paper A
   |
   | supports
   v
Claim X
   ^
   |
   | contradicts
   |
Paper B
```

The system should identify the disagreement and provide the relevant evidence from both sources.

Where possible, the final response should explain potential reasons for disagreement, such as:

- Different datasets
- Different experimental settings
- Different methodologies
- Different evaluation metrics
- Different populations or samples

Such explanations must themselves be evidence-grounded.

---

# 38. Agent Failure Handling

Agent failures must not automatically crash the entire research system.

For example:

```text
Critic Agent unavailable
        |
        v
Log failure
        |
        v
Continue with available verification
        |
        v
Return answer with reduced verification coverage
```

However, the system should clearly indicate when an answer has undergone reduced verification.

---

# 39. Agentic Architecture Development Strategy

Agentic verification should be introduced only after the core RAG pipeline is stable.

The intended progression is:

```text
Core RAG
   |
   v
Reliable Retrieval
   |
   v
Citation Mapping
   |
   v
Claim Extraction
   |
   v
Critic Agent
   |
   v
Verification Agent
   |
   v
Evidence Judge
   |
   v
Iterative Agentic Retrieval
```

The initial RAG implementation should not depend on the agentic pipeline.

The system must remain capable of producing evidence-grounded answers without agents.
# 40. Citation Architecture

Every generated research claim should remain traceable to retrieved evidence wherever possible.

The system should preserve the relationship:

```text
Answer Claim
     |
     v
Evidence
     |
     v
Document
     |
     v
Paper
     |
     v
Page / Section
```

This allows the system to provide citations that identify the original research source.

For example:

```text
[Paper A, p. 5]
[Paper B, p. 8]
```

The citation system must never invent:

- Papers
- Authors
- Page numbers
- Sections
- URLs
- Evidence

If sufficient evidence cannot be found, the system should explicitly indicate that the claim could not be verified.

---

# 41. Caching Architecture

The target architecture will use multiple levels of caching.

Caching is intended to reduce:

- Repeated computation
- Embedding generation
- Retrieval latency
- External API requests
- LLM generation costs

The initial implementation does not require a distributed caching system.

---

## 41.1 Query Cache

Stores reusable results for previously processed queries.

A query cache may eventually store:

- Normalized query
- Project context
- Retrieval configuration
- Retrieved evidence identifiers
- Timestamp
- Corpus version

---

## 41.2 Embedding Cache

Stores embeddings generated for previously processed document chunks.

Conceptually:

```text
Document Chunk
      |
      v
Hash / Cache Key
      |
      +---- Cached ----> Reuse Embedding
      |
      +---- Not Cached -> Generate Embedding
                              |
                              v
                         Store in Cache
```

The cache key should account for the embedding model and relevant preprocessing configuration.

---

## 41.3 Computation Cache

Expensive deterministic processing may eventually be cached.

Examples include:

- Text extraction
- OCR
- Figure processing
- Table processing
- Document normalization
- Chunk generation

---

## 41.4 Generation Cache

Generation results may eventually be cached when the same request can safely reuse an existing answer.

Generation caching must account for changes in:

- Project corpus
- Retrieved evidence
- Model
- Prompt
- Model configuration

---

# 42. Cache Invalidation

Caching must account for version changes.

A conceptual cache key may include:

```text
project_id
+
normalized_query
+
corpus_version
+
embedding_model_version
+
reranker_version
+
generation_model_version
```

If the underlying corpus or relevant model configuration changes, stale cached results must not be returned.

The cache design should prioritize correctness over maximum cache hit rate.
# 43. Asynchronous Document Processing

Expensive document-processing operations should eventually run asynchronously.

Potential asynchronous operations include:

- PDF text extraction
- OCR
- Embedding generation
- Figure processing
- Table processing
- Knowledge graph construction
- Large document indexing

The user should not need to keep an HTTP request open while expensive processing is running.

The target architecture is:

```text
User Upload
     |
     v
FastAPI
     |
     v
Create Processing Job
     |
     v
Task Queue
     |
     +------------+------------+
     |            |            |
     v            v            v
 Worker 1      Worker 2      Worker N
     |            |            |
     +------------+------------+
                  |
                  v
        Document Processing
                  |
                  v
          Processing Result
                  |
                  v
        Update Document Status
```

Documents should have processing states such as:

```text
UPLOADED
PROCESSING
COMPLETED
FAILED
```

A failed document should not cause unrelated documents or projects to fail.

---

# 44. Background Worker Architecture

Background workers should eventually handle computationally expensive operations.

Potential worker responsibilities include:

- Document parsing
- Chunking
- Embedding generation
- Multimodal extraction
- Vector indexing
- Knowledge graph construction

Workers should be independently scalable.

For example:

```text
                    Task Queue
                        |
        +---------------+---------------+
        |               |               |
        v               v               v
   Ingestion        Embedding        Graph
    Worker           Worker         Worker
        |               |               |
        v               v               v
     Storage          FAISS          Neo4j
```

The initial project should not introduce a distributed worker system until asynchronous processing is actually required.

---

# 45. Scalability Architecture

The target architecture should support horizontal scaling.

The API layer should be designed to remain stateless wherever practical.

Target architecture:

```text
                         Load Balancer
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
            API 1           API 2           API N
              |               |               |
              +---------------+---------------+
                              |
                            Redis
                              |
                         Task Queue
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
           Worker 1        Worker 2        Worker N
              |               |               |
              +---------------+---------------+
                              |
                    Processing Systems
```

Potential future infrastructure includes:

- Load balancing
- Stateless API instances
- Redis
- Background workers
- Task queues
- Object storage
- Distributed vector infrastructure
- Database replication
- Connection pooling

---

# 46. 1M+ User Scalability Target

The architecture should be designed with a future scale target of 1M+ users in mind.

This does not mean that the MVP must support 1M concurrent users.

Instead, the system should avoid architectural decisions that make horizontal scaling unnecessarily difficult.

Important scalability principles include:

- Stateless API services
- Externalized persistent state
- Cacheable computation
- Asynchronous processing
- Independent worker scaling
- Connection pooling
- Rate limiting
- Efficient database queries
- Observability
- Failure isolation

The system should be benchmarked before making claims about its actual capacity.

No performance claim should be made without measurement.

---

# 47. Scalability Evolution

The architecture should evolve approximately as follows:

```text
Stage 1
Modular Monolith
      |
      v
Stage 2
Async Processing + Caching
      |
      v
Stage 3
Horizontally Scaled API
      |
      v
Stage 4
Independent Workers
      |
      v
Stage 5
Service-Level Scaling
```

Microservices should only be introduced when there is a demonstrated engineering reason for them.

The project should not use microservices purely for complexity or resume value.
# 48. Observability Architecture

The production system should provide visibility into application, retrieval, generation, and infrastructure behavior.

Observability should eventually cover:

## Application Metrics

- Request count
- Error rate
- Response latency
- Throughput
- Endpoint performance

## Retrieval Metrics

- Retrieval latency
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

## Generation Metrics

- Faithfulness
- Answer relevance
- Citation precision
- Citation recall
- Unsupported claim rate

## Infrastructure Metrics

- CPU utilization
- Memory utilization
- Database latency
- Queue depth
- Cache hit rate
- Worker utilization

Potential future tooling includes:

- OpenTelemetry
- Prometheus
- Grafana

These tools should only be introduced when production observability becomes relevant.

---

# 49. Evaluation Architecture

Evaluation is a first-class component of the system.

The project should maintain evaluation datasets containing research questions and expected evidence.

A conceptual evaluation record may contain:

```text
Question
Expected Evidence
Relevant Paper
Relevant Page
Expected Answer Characteristics
```

Retrieval and generation should be evaluated separately.

```text
                Evaluation Dataset
                       |
          +------------+------------+
          |                         |
          v                         v
   Retrieval Evaluation      Generation Evaluation
          |                         |
          v                         v
   Recall / MRR / NDCG       Faithfulness / Relevance
```

This separation allows improvements to retrieval quality to be measured independently from LLM generation quality.

---

# 50. Retrieval Evaluation

The retrieval system should eventually be evaluated using metrics such as:

- Recall@K
- Precision@K
- MRR
- NDCG

Evaluation should compare different retrieval configurations.

For example:

```text
Dense Retrieval
       vs.
Sparse Retrieval
       vs.
Hybrid Retrieval
       vs.
Hybrid + Reranking
```

The project should use measurable experiments rather than assuming that a more complex retrieval architecture is automatically better.
# 51. Security Architecture

Security is an important consideration for the production version of the system.

The following principles should be followed:

- Never expose API keys or credentials.
- Never store secrets in Git.
- Use environment variables for secrets.
- Never create or commit a real `.env` file.
- Validate uploaded files.
- Restrict accepted file types.
- Enforce upload-size limits.
- Sanitize user-controlled inputs.
- Prevent unauthorized access to project data.
- Isolate project-specific retrieval contexts.
- Handle external URLs safely.
- Never bypass authentication, paywalls, or subscription restrictions.

Authentication and authorization are intentionally deferred to a later development phase.

The architecture should allow authentication to be introduced without requiring major changes to the core RAG pipeline.

---

# 52. Failure Isolation

A failure in one subsystem should not bring down unrelated functionality.

For example:

```text
Semantic Scholar API
        |
      FAILURE
        |
        v
Search layer reports provider failure
        |
        v
arXiv continues
        |
        v
RAG remains operational
```

Similarly:

```text
Neo4j unavailable
        |
        v
Graph retrieval disabled
        |
        v
Vector + sparse retrieval continue
```

Similarly, a failure during document processing should affect the relevant document rather than the entire project.

The architecture should support graceful degradation wherever practical.

---

# 53. Development Strategy

The system will be developed incrementally.

Each major component must be independently testable before additional architectural complexity is introduced.

The development sequence is:

## Phase 2 — Core RAG

1. Project foundation
2. PDF ingestion
3. Text extraction
4. Chunking
5. Embeddings
6. FAISS indexing
7. Retrieval
8. Hugging Face generation
9. Citation mapping

---

## Phase 3 — Retrieval Quality

- Hybrid retrieval
- Sparse search
- Cross-encoder reranking
- Retrieval evaluation
- Benchmarking

---

## Phase 4 — Multimodal RAG

- Figure extraction
- Table extraction
- Figure understanding
- Table understanding
- Multimodal evidence retrieval

---

## Phase 5 — Knowledge Graph

- Neo4j integration
- Entity extraction
- Relationship extraction
- Citation graph
- Graph-enhanced retrieval

---

## Phase 6 — Agentic Verification

- Retrieval agent
- Synthesis agent
- Critic agent
- Verification agent
- Evidence judge
- Claim-level verification
- Contradiction detection
- Iterative evidence retrieval

---

## Phase 7 — Academic Discovery

- Semantic Scholar
- arXiv
- Search aggregation
- Publication-year filtering
- Result deduplication
- Provider fault isolation

---

## Phase 8 — Production Engineering

- Redis caching
- Asynchronous processing
- Background workers
- Observability
- Performance benchmarking
- Horizontal scaling
- Production hardening

---

# 54. Technology Strategy

## Current / Initial

The initial implementation is expected to use:

- Python 3.12
- FastAPI
- Streamlit
- PostgreSQL
- Hugging Face
- Sentence Transformers
- FAISS

## Planned

The following technologies and capabilities may be introduced during later phases:

- Hybrid retrieval
- Cross-encoder reranking
- Multimodal processing
- Neo4j
- Redis
- Background workers
- Task queues
- OpenTelemetry
- Prometheus
- Grafana
- Object storage
- Distributed vector infrastructure

Technologies must be introduced only when their corresponding functionality is implemented.

The project should avoid adding infrastructure purely for complexity or resume value.

---

# 55. Core Architectural Rule

The most important architectural rule is:

> **The academic discovery layer is optional; the research analysis and RAG layer must remain independently functional.**

Therefore:

```text
                    Academic Search
                          |
                    OPTIONAL LAYER
                          |
                          v
                    Paper Discovery
                          |
                          v
                    Research Library
                          |
                          v
                  Document Ingestion
                          |
                          v
                    RAG Pipeline
                          |
                          v
                 Evidence + Answers
```

A failure in academic search must never cause the core RAG pipeline to fail.

A user must always be able to:

```text
Create Project
     |
     v
Upload PDF
     |
     v
Process Document
     |
     v
Ask Questions
     |
     v
Receive Evidence-Grounded Answers
```

without requiring any external academic search API.

---

# 56. Architectural Evolution

The system should evolve from a simple, reliable foundation toward the full research intelligence platform.

```text
                    CORE RAG
                       |
                       v
                Better Retrieval
                       |
                       v
                 Multimodal RAG
                       |
                       v
               Knowledge Graph
                       |
                       v
             Agentic Verification
                       |
                       v
              Academic Discovery
                       |
                       v
          Caching + Async Processing
                       |
                       v
              Production Scaling
```

Each stage should be evaluated before moving to the next.

The goal is not maximum architectural complexity.

The goal is a system where every layer provides a measurable improvement in capability, reliability, retrieval quality, scalability, or user experience.