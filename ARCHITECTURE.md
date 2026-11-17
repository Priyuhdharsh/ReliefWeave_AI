
# ReliefWeave AI – System Architecture

## 1. Project Overview

ReliefWeave AI is an Agentic AI-powered Emergency Response and Resource Coordination Platform.

The system collects information from multiple sources, analyses available evidence, identifies emergency needs and coordinates available resources to support informed disaster-response decisions.

The initial implementation focuses on Chennai flood-response scenarios using sample datasets.

## 2. System Architecture

The system consists of five logical layers.

### 2.1 Presentation Layer
- Streamlit
- Hospital Portal
- Stakeholder Portal
- Emergency Coordinator Dashboard

### 2.2 Backend Layer
- FastAPI
- Authentication
- Role-Based Access Control (RBAC)
- Request validation using Pydantic
- API and service integration

### 2.3 AI Orchestration Layer
- LangGraph Supervisor
- Incident Agent
- Resource Agent
- RAG Agent
- Impact Agent

### 2.4 Evidence Intelligence Layer
- Multi-source evidence fusion
- Evidence provenance and freshness
- Contradiction detection
- Dynamic source reliability
- Uncertainty propagation
- Scenario comparison

### 2.5 Data Layer
- SQLite – Structured operational data
- FAISS – Semantic document retrieval
- BM25 – Keyword-based retrieval
- Document Store – Emergency guidelines and reports

## 3. Data Flows

### 3.1 Operational Data Flow

Hospital / PWD / Disaster Management Portals
→ FastAPI
→ Authentication and Authorization
→ Pydantic Validation
→ SQLite
→ Controlled retrieval tools
→ AI Agents

### 3.2 Knowledge Document Flow

PDF / DOCX / TXT
→ Document Loader
→ Text Splitter
→ Embeddings
→ FAISS Index

Documents are also indexed using BM25 for keyword retrieval.

Hybrid retrieval combines semantic and keyword search, followed by reranking.

### 3.3 Agentic Workflow

Coordinator Query
→ FastAPI
→ LangGraph Supervisor
→ Relevant Agents and Tools
→ Evidence Intelligence
→ Recommendation
→ Coordinator Review and Approval

## 4. Agent Responsibilities

### Supervisor
- Coordinates agent execution.
- Routes tasks to relevant agents.
- Maintains shared workflow state.
- Combines agent results.

### Incident Agent
- Retrieves incident reports.
- Identifies affected locations.
- Analyses road conditions and incident severity.
- Identifies conflicting incident information.

### Resource Agent
- Retrieves hospital capacity.
- Checks shelter availability.
- Identifies available supplies and rescue resources.
- Evaluates resource constraints.

### RAG Agent
- Retrieves emergency guidelines and reports.
- Performs hybrid retrieval.
- Provides relevant document citations.

### Impact Agent
- Analyses dependencies between incidents and resources.
- Identifies cascading impacts.
- Compares response scenarios.

## 5. Evidence Intelligence Engine

The Evidence Intelligence Engine evaluates information collected from multiple sources.

Its responsibilities include:

- Tracking evidence sources and timestamps.
- Evaluating source reliability and freshness.
- Detecting contradictory reports.
- Preserving uncertainty in downstream recommendations.
- Comparing alternative response scenarios.

## 6. Human Approval

The platform provides decision support rather than autonomous emergency dispatch.

Emergency coordinators review AI-generated recommendations and approve or reject them.

Critical decisions remain under human control.

## 7. Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Backend | FastAPI |
| Validation | Pydantic |
| Agent orchestration | LangGraph |
| LLM integration | LangChain |
| Vector retrieval | FAISS |
| Sparse retrieval | BM25 |
| Reranking | Cross-Encoder |
| Database | SQLite / SQLAlchemy |
| UI | Streamlit |
| Testing | Pytest |
| Deployment | Docker |
