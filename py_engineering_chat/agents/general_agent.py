from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Dict, Any
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
from py_engineering_chat.util.tiered_memory import TieredMemory
from sentence_transformers import SentenceTransformer
import time

class State(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

class GeneralAgent:
    def __init__(self):
        self.graph_builder = StateGraph(State)
        self.tiered_memory = TieredMemory()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = get_configured_logger(__name__)
        self.setup_graph()

    def setup_graph(self):
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

        llm = ChatOpenAI(temperature=0, model_name="gpt-4", verbose=False)
        tools = get_tools()
        llm_with_tools = llm.bind_tools(tools)
        llm_with_prompt = prompt | llm_with_tools

        def chatbot(state: State):
            response = llm_with_prompt.invoke({
                "tools": tools,
                "messages": state["messages"],
                "context": state["context"]
            })
            return {"messages": [response]}

        self.graph_builder.add_node("chatbot", chatbot)

        tool_node = ToolNode(tools=tools)
        self.graph_builder.add_node("tools", tool_node)

        self.graph_builder.add_edge(START, "chatbot")
        self.graph_builder.add_conditional_edges(
            "chatbot",
            tools_condition,
            {"tools": "tools", "__end__": "__end__"}
        )
        self.graph_builder.add_edge("tools", "chatbot")

        self.graph = self.graph_builder.compile()

    def add_to_memory(self, role: str, content: str):
        embedding = self.embedding_model.encode([content])[0].tolist()
        metadata = {"role": role, "timestamp": time.time()}
        self.tiered_memory.add_memory(content, metadata, embedding)

    def get_context(self, query: str, n_results: int = 5) -> str:
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        context = self.tiered_memory.get_context(query_embedding, n_results)
        return "\n".join([f"{item['metadata']['role']}: {item['content']}" for item in context])

    def run_conversation(self):
        state = {"messages": [], "context": ""}
        config = {"configurable": {"thread_id": "1"}}
        session = PromptSession(completer=FileCompleter(), key_bindings=kb)

        print("Welcome to the General Agent! Type 'exit' to end the conversation.")

        while True:
            try:
                user_input = session.prompt("You: ")
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break

                context_data_list = parse_commands(user_input, ChatSettingsManager())
                context_strings = [context_data.toString() for context_data in context_data_list]
                state["context"] = " ".join(context_strings)

                # Add user input to memory
                self.add_to_memory("user", user_input)

                # Get relevant context from memory
                memory_context = self.get_context(user_input)
                state["context"] += f"\nRelevant memory context:\n{memory_context}"

                state["messages"].append(HumanMessage(content=user_input))

                for event in self.graph.stream(state, config):
                    for value in event.values():
                        if isinstance(value["messages"][-1], AIMessage):
                            assistant_message = value["messages"][-1].content
                            print(f"Assistant: {assistant_message}")
                            state["messages"].append(AIMessage(content=assistant_message))
                            
                            # Add assistant response to memory
                            self.add_to_memory("assistant", assistant_message)

            except KeyboardInterrupt:
                print("\nExiting...")
                break

def run_continuous_conversation():
    agent = GeneralAgent()
    agent.run_conversation()

if __name__ == "__main__":
    run_continuous_conversation()
