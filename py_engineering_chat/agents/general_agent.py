from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from py_engineering_chat.util.command_parser import parse_commands
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager 
from langchain_core.prompts import PromptTemplate
from py_engineering_chat.tools.custom_tools import get_tools

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    # Context is a string that holds additional context information
    context: str

# Initialize the graph builder
graph_builder = StateGraph(State)

prompt = PromptTemplate.from_template("""
You are a helpful assistant specialized in programming tasks. 
You operate using a "React" framework: Plan, Act, and Execute. 
You have access to a set of tools designed to interact with the project's directory structure, 
read and write files, and perform other project-specific operations:
{tools}

If any of the tools produce an error, don't repeat the tool call.  Just say there was an error.

Conversation history: {messages}

Additional contextual information to help you answer the user's questions: {context}

When a task is presented, you will:
1. Plan: Analyze the task and determine the necessary steps.
2. Act: Use the available tools to perform actions.
3. Execute: Deliver the results and provide feedback.
"""
)

# Define the LLM model and bind tools
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4o"
)
tools = get_tools()
llm_with_tools = llm.bind_tools(tools)  # Combine model with tools
llm_with_prompt = prompt | llm_with_tools

# Define the chatbot node
def chatbot(state: State):
    response = llm_with_prompt.invoke({
        "tools": tools,
        "messages": state["messages"],
        "context": state["context"]
    })  # Use the combined model
    return {"messages": [response]}

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

# Define the tools node
tool_node = ToolNode(tools=tools)

# Add nodes to the graph
graph_builder.add_node("tools", tool_node)

# Define the entry and exit points
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", "__end__": "__end__"})
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
graph = graph_builder.compile()

# Function to run the graph with continuous conversation
def run_continuous_conversation():
    state = {"messages": []}
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # Parse the user input to get the context
        context_data = parse_commands(user_input, ChatSettingsManager())
        state["context"] = context_data  # Add context to the state

        state["messages"].append(("user", user_input))
        
        for event in graph.stream(state, config):
            for value in event.values():
                if isinstance(value["messages"][-1], BaseMessage):
                    assistant_message = value["messages"][-1].content
                    print("Assistant:", assistant_message)
                    state["messages"].append(("assistant", assistant_message))

# Example usage
