from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define the LLM model
llm = ChatOpenAI(temperature=0)

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
    response = llm.invoke(messages)
    return {"messages": [response]}

# Initialize the graph
memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)

# Define state logic
def get_state(state) -> str:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1], HumanMessage):
        return END
    return "info"

workflow.add_conditional_edges("info", get_state)
workflow.add_edge(START, "info")

# Compile the graph
graph = workflow.compile(checkpointer=memory)

# Function to run the graph
def run_conversation():
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

if __name__ == "__main__":
    run_conversation()
