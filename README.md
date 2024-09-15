# RAG-Pipeline

An end-to-end AI chat agent interface, with Retrieval Augmented Generation based information retrieval system. Ask questions to the agent, analyse your data, create customised agents with intelligent behaviour for automating various tasks.

### Features

- Dynamically configurable Hybrid-Retrieval system, with a semantic search through vector similarity retriever and a keyword search through elastic retriever both working independently to fetch results.
- The results are combined using an ensemble retriever, to get the best of both semantics ( underlying meaning and context ) and keyword match.
- Retrieved Candidates are Reranked using a Cross-Encoder reranker, ensuring only the most relevant candidates in the final list of context.
- Customizable prompt augmentation with the input query, allowing creation of smart agents that can automate a given task or workflow.
- Usage of graph database in retrieval stage , utilizing graph-search algorithms for even better retrieval relevancy and also for predicting search krywords, topic recommendation to the user.
- LLM Memory chaining and Session based login, ensuring context is maintained across current session and also previous chat history of the user.
- Secure User login and authentication system with encrypted using Keycloack Identity and access management for isolated credentials storage.
- Supports Partial updates or full updates to the knowledge base, by triggering an entirely separate updation pipeline.
- Advanced tokentization mechanism, using nlp based sentence tokenizer, preserving contextual meaning in each token leading to more relavant embeddings and improved retrieval.
- Awesome UI with the ability to play around various configurations of the system, and dynamically change the system's technical behaviour to own requirements. Also has a prompt engineering suite.
- Fully configurable infrastructure, allowing custom choice of LLMs, Embedding Model, and Reranking Model. Choice of vectore database and search engine to be used. Also supports local inferencing engines using ollama and onnx runtime.
- The graph database comes in handy to augment the other searches with a more nuanced understanding of the data, thus providing them loose guidelines. Also when need to keep track of the relationships is critical or part of the system requirements ( this can be extended to ai text to query service in relational databases)
- The graph database also comes with a schematic search on the graph itself, leveraging contextual meaning of nodes and relationships in the search.
- The knowledge graph helps in meta-level topic recognition from the data and keyword prediction from user search history patterns. This allows recommendations and further augmentation of the response.
