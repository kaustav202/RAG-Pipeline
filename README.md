# RAG-Pipeline

An end-to-end AI chat agent interface, with Retrieval Augmented Generation based information retrieval system. Ask questions to the agent, analyse your data, create customised agents with intelligent behaviour for automating various tasks.

### Features

- Dynamically configurable Hybrid-Retrieval system, with a semantic search through vector similarity retriever and a keyword search through elastic retriever both working independently to fetch results.
- The results are combined using an ensemble retriever, to get the best of both semantics ( underlying meaning and context ) and keyword match.
- Retrieved Candidates are Reranked using a Cross-Encoder reranker, ensuring only the most relevant candidates in the final list of context.
- Customizable prompt augmentation with the input query, allowing creation of smart agents that can automate a given task or workflow.
- Usage of graph database in retrieval stage , utilizing graph-search algorithms for even better retrieval relevancy and also for predicting search krywords, topic recommendation to the user.
- LLM Memory chaining and Session based login, ensuring context is maintained across current session and also previous chat history of the user.
