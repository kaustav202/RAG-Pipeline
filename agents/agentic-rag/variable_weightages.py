from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from models.chat_model import AnthropicModel


def dynamic_weightage_logic(query):

    # Define the custom prompt
    prompt_template = """
    Given the following query, determine the appropriate weightages for semantic and elastic retrievers.
    The weightages should be integers and sum up to 100.

    Query: {query}

    Consider the following factors:
    1. If the query is more conceptual or requires understanding of context, lean towards a higher semantic weight.
    2. If the query is more factual or requires exact matching, lean towards a higher elastic weight.
    3. For a balanced approach, consider a 50-50 split.

    Output the weightages as two integers in a list format: [semantic_weight, elastic_weight]

    Weightages:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

    # Create the LLM chain
    anthropic_client = AnthropicModel()
    llm = anthropic_client.get_llm()
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain and parse the output
    result = chain.run(query=query)
    weightages = eval(result.strip())  # Convert string representation to list

    return weightages





from langchain.agents import AgentExecutor, create_react_agent , Tool, initialize_agent , create_tool_calling_agent

tools = [
    Tool(
        name = "Get_Dynamic_Weightages",
        func = dynamic_weightage_logic,
        description = "Use this Tool to find the appropriate percentages of respective weightages to the semantic retriever and the elastic retriever"
    ),
    Tool(
        name = "Invoke_Dynamic_Hybrid_Retriever",
        func = lambda weights: ensemble_retriever.invoke(weightages=[int, int]),
        description = "Use this Tool to lookup information using the dynamic hybrid retriever with two integer weights for semantic and elastic retrievers respectively"
    )
]

