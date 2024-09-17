# RAG-Pipeline

An end-to-end, production-ready, Retrieval Augmented Generation based Information Retrieval System, with a chat interface. Ask questions, find information from data, analyse your data, create customised agents with intelligent behaviour for automating various tasks.


Most LLM based information retrieval systems face two major challenges :

1. Finding the right context, when presented with a large knowledgebase ie 10s of thousands documents, which is what can be expected in practical applications.
2. Accuracy of answers based on proper interpretation of the context

The system attempts to tackle each of these problems in a very nuanced and robust way. Hybrid-Search based Grounding Mechanism, accompanied by a robust contextual tool augmentation, all ochestrated by an autonomous Agentic RAG pipeline that dynamically chooses the steps or actions to best suite the specific query. 

The chain-of-thought prompting of the agent is responsible for the dynamic RAG workflow, which can be customized as needed.


### Technical Implementation

- Dynamically configurable Hybrid-Retrieval system, with a semantic search through vector similarity retriever and a keyword search through elastic retriever both working independently to fetch results.
- The results are combined using an ensemble retriever, to get the best of both semantics ( underlying meaning and context ) and keyword match.
- Retrieved Candidates are Reranked using a Cross-Encoder reranker, ensuring only the most relevant candidates in the final list of context.
- Customizable prompt augmentation with the input query, allowing creation of smart agents that can automate a given task or workflow.
- Usage of graph database in retrieval stage , utilizing graph-search algorithms for even better retrieval relevancy and also for predicting search krywords, topic recommendation to the user.
- LLM Memory chaining and Session based login, ensuring context is maintained across current session and also previous chat history of the user.
- Advanced tokentization mechanism, using nlp based sentence tokenizer, preserving contextual meaning in each token leading to more relavant embeddings and improved retrieval.
- The graph database comes in handy to augment the other searches with a more nuanced understanding of the data, thus providing them loose guidelines. Also when need to keep track of the relationships is critical or part of the system requirements ( this can be extended to ai text to query service in relational databases)
- The graph database also comes with a semantic search on the graph itself, leveraging contextual meaning of nodes and relationships in the search.
- The knowledge graph helps in meta-level topic recognition from the data and keyword prediction from user search history patterns. This allows recommendations and further augmentation of the response.


### Features
- Secure User login and authentication system with encrypted using Identity and access management for isolated credentials storage.
- Supports Partial updates or full updates to the knowledge base, by triggering an entirely separate updation pipeline.
- Awesome UI with the ability to play around various configurations of the system, and dynamically change the system's technical behaviour to own requirements. Also has a prompt engineering suite.
- Fully configurable infrastructure, allowing custom choice of LLMs, Embedding Model, and Reranking Model. Choice of vectore database and search engine to be used. Also supports local inferencing engines using ollama and onnx runtime.
- The relationship can be with fixed schema exactly describing the e-r model, or schema less making it more flexible through llm generated relationships.
