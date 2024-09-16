from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from py_engineering_chat.util.command_parser import parse_commands
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager 
from langchain_core.prompts import PromptTemplate
from prompt_toolkit import PromptSession
from py_engineering_chat.util.file_completer import FileCompleter
from py_engineering_chat.util.enter_key_bindings import kb
from py_engineering_chat.util.context_model import ContextData
from py_engineering_chat.tools.custom_tools import get_tools
from py_engineering_chat.util.logger_util import get_configured_logger



class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

# Initialize the graph builder
graph_builder = StateGraph(State)

prompt = PromptTemplate.from_template("""
You are a helpful assistant specialized in programming tasks.
You operate using a "React" framework: Plan, Act, and Execute.
You have access to a set of tools designed to interact with the project's directory structure,
read and write files, and perform other project-specific operations:
{tools}

If any of the tools produce an error, don't repeat the tool call. Just say there was an error.

Conversation history: {messages}

Context included by user:
Additional contextual information to help you answer the user's questions: {context}

When a task is presented, you will:

1. **Plan**: Analyze the task and determine the necessary steps. Some messages will be simple questions that don't require planning

2. **Act**: Use the available tools to perform actions.

   - For **read actions**: ask the user if they want to proceed.
   - For **write actions**: before performing any write operation:
     - **Check Git Branch**: Determine if the current Git branch is the main branch or a feature branch.
     - **If on the main branch**:
       - Offer to switch to a new feature branch.
       - Suggest a branch name relevant to the task (e.g., `feature/add-login`).
       - Ask the user for permission to create and switch to the new branch.
     - **If on a feature branch**:
       - Confirm with the user before proceeding.
     - **Ask for Permission**: Before any write action, request the user's approval.

   - **Do not include raw tool outputs** in your responses.
   - **Summarize** the results of tool actions in a clear and concise manner.
   - **Avoid displaying any internal errors or stack traces**; provide a user-friendly error message if needed.

3. **Execute**: Deliver the results and provide feedback.

   - Summarize the actions taken.
   - Inform the user of any changes made.
   - If permission was denied, explain which actions were not performed.
""")

# Define the LLM model and bind tools
llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",
    verbose=False
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
    })
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)

# Remove the CustomToolNode class and use the original ToolNode
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Update the graph edges
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", "__end__": "__end__"}
)
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
graph = graph_builder.compile()

# The rest of your code remains mostly the same
# ...
# ANSI color codes
colors = [
    '\033[91m',  # Red
    '\033[92m',  # Green
    '\033[93m',  # Yellow
    '\033[94m',  # Blue
    '\033[95m',  # Magenta
    '\033[96m',  # Cyan
    '\033[97m',  # White
]
reset_color = '\033[0m'  # Reset color

# Update the run_continuous_conversation function if necessary
def run_continuous_conversation():
    state = {"messages": [], "tool_rejection": False, "user_feedback": ""}
    config = {"configurable": {"thread_id": "1"}}
    color_index = 0
    session = PromptSession(completer=FileCompleter(), key_bindings=kb)

    while True:
        try:
            user_input = session.prompt("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            context_data_list = parse_commands(user_input, ChatSettingsManager())
            context_strings = [context_data.toString() for context_data in context_data_list]
            state["context"] = " ".join(context_strings)
            state["messages"].append(("user", user_input))

            print("\n")
            color = colors[color_index % len(colors)]
            color_index += 1
            print(f"{color}You:{reset_color} {user_input}")

            for event in graph.stream(state, config):
                for value in event.values():
                    if isinstance(value["messages"][-1], AIMessage):
                        assistant_message = value["messages"][-1].content
                        color = colors[color_index % len(colors)]
                        color_index += 1
                        print("\n")
                        print(f"{color}Assistant:{reset_color} {assistant_message}")
                        state["messages"].append(("assistant", assistant_message))

        except KeyboardInterrupt:
            print("\nExiting...")
            break