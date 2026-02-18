# 03_CMS_Archi-MIT: Agentic Physics Intelligence

This module implements an **Agentic AI Workflow** designed to bridge the gap between Large Language Models (LLMs) and real-world High Energy Physics data from the CMS experiment.

## 🔬 Project Overview
The "Archi-MIT" system allows an AI agent to not only reason about physics but also interact with datasets and detector status monitors via the **Model Context Protocol (MCP)**. 

> **Note on Script Variety:** This directory contains multiple experimental implementations (`langchain_agent.py`, `physics_agent.py`, `main_orchestrator.py`). These represent different prototyping stages used to evaluate various frameworks, tool-calling patterns, and integration strategies (LangChain vs. custom ReAct loops).

## 📂 Directory Structure

* **`main_orchestrator.py`**: The central entry point that coordinates the agent, tools, and physics guardrails.
* **`agents/`**: Contains the "brain" of the system.
    * `cms-researcher.agent.md`: The system prompt defining the agent's persona and physics protocols.
    * `physics_agent.py` & `langchain_agent.py`: Different experimental implementations of the agentic reasoning loop.
* **`mcp_server/`**: The "hands" of the agent. Implements tools following the Model Context Protocol.
    * `physics_mcp_server.py`: Tool for analyzing `MuRun2010B.csv` and calculating Invariant Mass.
    * `server.py`: Mock server for monitoring CMS detector hardware status.
    * `phys_tool.py`: Mathematical utility for pulse area integration.
    * `mock_client.py`: A testing utility to simulate LLM tool-calling.
* **`middleware/`**: Safety and validation layer.
    * `physics_guardrails.py`: Intercepts agent outputs to validate that reported physics values (like Z-boson mass) are within scientific ranges.
* **`embeddings/`**: Scripts for vectorizing physics documentation (`study_embeddings.py`) to provide the agent with RAG (Retrieval-Augmented Generation) capabilities.
* **`tests/`**: Unit tests for the internal logic and physics calculations.



## 🛠️ Key Capabilities

1. **Tool-Augmented Reasoning**: The agent uses `cms_data_analyzer` to perform live Python-based analysis on CMS Open Data.
2. **Physics Validation**: Every response passes through a guardrail to ensure the AI doesn't "hallucinate" unphysical results.
3. **Hardware Awareness**: Through the MCP server, the agent can check the operational status of muon chambers.
