#Control agent behaviour or series of actions, with chain-of-thought prompting

FORMAT_INSTRUCTIONS = '''
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Action: the action to take should be custom_logger, with the info retrieved as it's input
Action Input: the input to the action
... (this Thought/Action/Action Input/Observation/Action/Action Input can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important: After using any retrieval tool, use the custom_logger tool to log the retrieved information.

Custom Logger Format:
Action: custom_logger
Action Input: {{
    "tool_used": "name of the retrieval tool used",
    "query": "the query used for retrieval",
    "retrieved_info": "summary of the retrieved information"
}}

Begin!

Question: {input}
'''