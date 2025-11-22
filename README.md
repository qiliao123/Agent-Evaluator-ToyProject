# Agent-Evaluator-ToyProject
This is a toy project for evaluating RAG agents built using GitHub Copilot Agent.

**Duration**: ~2–3 hours

**Tech Stack**: Python, ddgs (DuckDuckGo Search), OpenAI/Azure OpenAI, CLI, LLM-based Evaluator, Github Copilot Agent. 

## Overview
This project implements a **minimal Retrieval-Augmented Generation (RAG) agent** and an **LLM-based evaluator**, following the assignment instructions.
The solution includes:
- A search component using DuckDuckGo (ddgs)
- A lightweight RAG pipeline that accepts a query, searches the web and synthesizes response using LLM
- A CLI interface for interactive testing
- An automatic evaluator that assesses generated answers for different dimensions such as relevance, accuracy, and completeness, etc.
- A metrics redesign focusing on accuracy-based scoring instead of subjective preferences

The code is structured to be simple, and readable.

## Prerequisites
OpenAI API Key

VS Code

## Components
**Search (web_search.py)**
- Uses ddgs to query DuckDuckGo
- Returns structured results: title, body etc

**RAG agent (rag_agent.py)**

Pipeline: query → search results → retrieved context → LLM generation

Provides: 
- Context grounding
- Answer synthesis

**CLI agents with LLM evaluation**
- Accepts user query through CLI
- Allows users to evaluate each answers using LLM-based metrics

## Instructions 
You have three CLI agents to choose from with different evaluation metrics included:

**Option 1: Basic CLI (rag_cli.py)**

Has LLM-based evaluation with scores for relevance, accuracy, completeness, and clarity.

**Option 2: Improved CLI (rag_cli_improved.py)**

More advanced with objective metrics, uses GPT-4 for evaluation, and returns JSON-formatted results with detailed accuracy percentages.

**Option 3: Evaluator CLI (rag_evaluator.py)**

Comprehensive evaluation with citation coverage, factual grounding, and hallucination detection.

> **Each evaluator has its pros and cons. It's not a goal to decide which evaluator works better in this experiment.** 

All require setting your OpenAI API key.
Without OpenAI API key provided, you can still run your query and the agent will return search results only without evaluation. 

> Note: other .py files show the thought process steps, and not necessarily required to run the CLI agent.  

### How to 
Choose any of the three options above. Ask the RAG agent a question.
  ```bash
   python rag_cli.py
   ```
- You will be asked to enter your OpenAI API key.
- Enter your query.
- Agent returns the search result summary and the generated answer.
- You will be asked if you would like to evaluate the answer. Type 'y' to view the evaluation results. (OpenAI API key required)
- Continue with more questions or quit

Example: 
```
💭 Enter your query (or 'quit' to exit): What is Python?
🔍 Searching for: What is Python?
🤖 Generating answer...
🤖 AGENT RESPONSE
[Answer appears here]
📊 Run objective evaluation? (y/n): y
   ```

## Evaluation metrics
The evaluation focuses on accuracy and factual grounding, with supporting metrics such as query alignment, completeness etc. 

These evaluation metrics are preliminary and only support the experiment of this toy project. 

More tests and reevaluation need to be done in order to compare, and define the right metrics for the RAG agent fulfilling specific tasks. 

Improvement can be done to include different sets of evaluations in one agent for defined scenarios. 

## Demo
See demo.mp4


