# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
import numpy as np
import os
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Dict, Iterable

from retriever.vector_retriever import VectorRetriever
from retriever.bm25_retriever import ElasticRetriever
from models.rerank_model import ReRanker
from retriever.ensemble_retriever import HybridRetriever

vec_retriever = VectorRetriever()
es_retriever = ElasticRetriever()

params = {"k": 10, "score_threshold": 0.5}

semantic_retriever = vec_retriever.config_retriever(index="test",search_kwargs=params)
bm25_retriever = es_retriever.config_retriever(index="test", content_field="content")

contextual_compressor = ReRanker()
semantic_retriever_reranked = contextual_compressor.apply_compression(semantic_retriever)

bm25_retriever_reranked = contextual_compressor.apply_compression(bm25_retriever)

hybrid = HybridRetriever(bm25_retriever,semantic_retriever_reranked)

ensemble_retriever = hybrid.create_ensemble_retriever(weights=[0.5,0.5])


# docs = bm25_retriever.invoke("Hi")

# print(docs)


template = """Answer the question in detail and elaborately based only on the following context :


{context}

Question: {question}
"""


prompt = ChatPromptTemplate.from_template(template)



from models.chat_model import AnthropicModel
anthropic_client = AnthropicModel()

llm = anthropic_client.get_llm()


from chain import Chain

chains = Chain(llm=llm, retriever=ensemble_retriever)

base_qna_chain = chains.qna_chain(prompt=prompt)

# inference = base_qna_chain.run("input")


qna_sources_chain = chains.qna_sources_chain(prompt=prompt)

# inference_s = qna_sources_chain("input")

#Meta class :

# control params

# invoke

# inference chain





def prompt_sp():
    s = '''
    '''
    return s


def prompt_gen():
    s = '''
    '''
    return s


def get_prompt():
    ps = prompt_sp()
    pg = prompt_gen()
    ins = '''
Use one of the following prompts, PROMPT1 or PROMPT2 , depending on the type of given input query. If the question asks about a specific person or entity use PROMPT1. If the question asked is a generic or situational use PROMPT2.
PROMPT1: {}

PROMPT2: {}
'''.format(ps, pg)
    return ins


from langchain.agents import AgentExecutor, create_react_agent , Tool, initialize_agent , create_tool_calling_agent

tools = [
    Tool(
        name = "Get_Information",
        func = base_qna_chain.run,
        description = "Use this Tool to lookup information on input query"
    ),
    Tool(
        name = "Retriever_Vec",
        func = semantic_retriever_reranked.invoke,
        description = "Use this retriever lookup information using Vector Retriever.Input to this tool must be a SINGLE JSON STRING"
    ),
    Tool(
        name = "Retriever_Key",
        func = bm25_retriever_reranked.invoke,
        description = "Use this retriever lookup information using Elastic Search Retriever.Input to this tool must be a SINGLE JSON STRING"
    )
]


from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    ensemble_retriever,
    "Native_Retriever",
    "Search for information using Native Retriever. Forinformation about any questions , you must use this tool!"
)



# tools = [retriever_tool]


system_context = '''You are an intelligent agent.
Answer the following questions as best you can. 
You have access to the following tools:

{tools}

You have access to the following tools to select the prompt dynamically based on the query:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
'''


prompt_agent = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_context,
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

agent_sys = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
'''

agent = create_react_agent(llm,
                           tools,
                           prompt_agent)

agent_executor = AgentExecutor(tools=tools,
                         agent=agent,
                         handle_parsing_errors=True,
                         verbose=True)


# agent = create_tool_calling_agent(llm, tools, prompt_agent)

# agent = initialize_agent(tools,
#                          llm,
#                          agent="zero-shot-react-description",
#                          verbose=True)


t = '''/n input_variables=['agent_scratchpad', 'input'] optional_variables=['chat_history'] 
input_types={'chat_history': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]], 
'agent_scratchpad': typing.List[typing.Union[langchain_core.messages.ai.AIMessage, langchain_core.messages.human.HumanMessage, langchain_core.messages.chat.ChatMessage, langchain_core.messages.system.SystemMessage, langchain_core.messages.function.FunctionMessage, langchain_core.messages.tool.ToolMessage]]} 
partial_variables={'chat_history': []} metadata={'lc_hub_owner': 'hwchase17', 'lc_hub_repo': 'openai-functions-agent', 'lc_hub_commit_hash': 'a1655024b06afbd95d17449f21316291e0726f13dcfaf990cc0d18087ad689a5'} 
messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], template='You are a helpful assistant')), MessagesPlaceholder(variable_name='chat_history', optional=True), 
HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['input'], template='{input}')), MessagesPlaceholder(variable_name='agent_scratchpad')]"
'''


prompt_exec = ChatPromptTemplate.from_template(t)



from langchain import hub


prompt_nr = hub.pull("hwchase17/openai-functions-agent")

agent_nr = create_tool_calling_agent(llm, tools, prompt_nr)

#from langchain.agents import AgentExecutor

executor_nr = AgentExecutor(agent=agent, tools=tools, verbose=True)

