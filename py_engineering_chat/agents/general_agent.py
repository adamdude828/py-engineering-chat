from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOpenAI
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
    edit_mode: bool

class GeneralAgent:
    def __init__(self):
        self.graph_builder = StateGraph(State)
        self.tiered_memory = TieredMemory()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = get_configured_logger(__name__)
        self.edit_mode = False  # Default to read-only mode
        self.setup_graph()

    def setup_graph(self):
        prompt = PromptTemplate.from_template("""
        You are a helpful assistant specialized in programming tasks. You have access to the following tools:
        {tools}

        Conversation history: {messages}

        Context: {context}

        Instructions:
        1. Analyze the user's request and respond appropriately.
        2. Use tools when necessary, but don't mention them explicitly in your response.
        3. Be concise in your explanations and actions.
        4. If edit_mode is False, do not perform any write operations.
        5. For read operations, summarize the results briefly.
        6. For write operations (when edit_mode is True):
        7. Avoid displaying raw tool outputs or error messages.

        Current edit mode: {edit_mode}

        Respond to the user's request:
        """)

        llm = ChatOpenAI(temperature=0, model_name="gpt-4", verbose=False)
        tools = get_tools()
        llm_with_tools = llm.bind_tools(tools)
        llm_with_prompt = prompt | llm_with_tools

        def chatbot(state: State):
            response = llm_with_prompt.invoke({
                "tools": tools,
                "messages": state["messages"],
                "context": state["context"],
                "edit_mode": state["edit_mode"]
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

    def toggle_edit_mode(self):
        self.edit_mode = not self.edit_mode
        mode = "enabled" if self.edit_mode else "disabled"
        print(f"Edit mode {mode}")

    def run_conversation(self):
        state = {"messages": [], "context": "", "edit_mode": self.edit_mode}
        config = {"configurable": {"thread_id": "1"}}
        session = PromptSession(completer=FileCompleter(), key_bindings=kb)

        print("Welcome to the General Agent! Type 'exit' to end the conversation.")
        print("Type '/toggle_edit' to switch between read-only and edit modes.")

        while True:
            try:
                user_input = session.prompt("You: ")
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == '/toggle_edit':
                    self.toggle_edit_mode()
                    state["edit_mode"] = self.edit_mode
                    continue

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
