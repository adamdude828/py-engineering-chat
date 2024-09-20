from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Annotated
from typing_extensions import TypedDict
from py_engineering_chat.tools.custom_tools import get_tools  # Updated import statement
from langgraph.prebuilt import ToolNode, tools_condition  # Import ToolNode and tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define the LLM model and bind tools
llm = ChatOpenAI(temperature=0)
tools = get_tools()
llm_with_tools = llm.bind_tools(tools)  # Combine model with tools

# Define the prompt template
template = """Your job is to get information from a user about what type of prompt template they want to create.
You should get the following information from them:
- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to
If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.
After you are able to discern all the information, call the relevant tool."""

def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages

def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tools.invoke(messages)  # Use the combined model
    return {"messages": [response]}

# Initialize the graph
memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)

# Add the tool node
tool_node = ToolNode(tools=tools)
workflow.add_node("tools", tool_node)

# Define state logic
def get_state(state) -> str:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "tools"  # Transition to tools node
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"

workflow.add_conditional_edges("info", get_state, {"tools": "tools", "__end__": "__end__"})
workflow.add_edge("tools", "info")  # Return to info after tool usage
workflow.add_edge(START, "info")

# Compile the graph
graph = workflow.compile(checkpointer=memory)

# Function to run the graph
def run_conversation_planning_agent():
    state = {"messages": []}
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        state["messages"].append(HumanMessage(content=user_input))
        for event in graph.stream(state, {}):
            for value in event.values():
                if isinstance(value["messages"][-1], AIMessage):
                    assistant_message = value["messages"][-1].content
                    print("Assistant:", assistant_message)
                    state["messages"].append(AIMessage(content=assistant_message))
                elif isinstance(value["messages"][-1], ToolMessage):  # Handle tool messages
                    tool_message = value["messages"][-1].content
                    print("Tool:", tool_message)
                    state["messages"].append(ToolMessage(content=tool_message))
