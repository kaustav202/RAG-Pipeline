# def index_switch_logic():



def custom_information_logger(step_output: str):
    """
    Log the output of each step after an action is taken by the agent.
    
    Args:
    step_output (str): The output of the current step to be logged.
    
    Returns:
    str: A confirmation message that the output has been logged.
    """
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a log entry
    log_entry = f"[{timestamp}] Step Output: {step_output}\n"
    
    # Append the log entry to a file
    with open("agent_steps.log", "a") as log_file:
        log_file.write(log_entry)
    
    return f"Output logged successfully at {timestamp}"



from langchain.agents import AgentExecutor, create_react_agent , Tool, initialize_agent , create_tool_calling_agent

tools = [
    Tool(
        name = "Get_Information_A",
        func = root_qna.run(index='index_A'),
        description = "Use this Tool to lookup information on input query when the query asks about A"
    ),
    Tool(
        name = "Get_Information_B",
        func = root_qna.run(index='index_B'),
        description = "Use this Tool to lookup information on input query when the query asks about B"
    ),
    Tool(
        name = "Custom_Logger",
        func = custom_information_logger,
        description = "Use this tool to log output of the step after each action"
    )
]
