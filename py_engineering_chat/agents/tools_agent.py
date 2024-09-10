from typing import Annotated, TypedDict, Optional, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from py_engineering_chat.tools.custom_tools import get_tools
from py_engineering_chat.agents.base_agent import BaseAgent
import os

class State(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], "The conversation history"]
    next: Annotated[Optional[Literal["agent", "tool"]], "The next node to call"]

class ToolsAgent(BaseAgent):
    def __init__(self, openai_api_key):
        super().__init__()
        
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4",
            openai_api_key=openai_api_key
        )
        
        self.tools = get_tools()
        self.tool_executor = ToolExecutor(self.tools)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following tools to assist the user: {tool_names}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.agent = (
            self.prompt
            | self.llm.bind(functions=[tool.schema for tool in self.tools])
            | OpenAIFunctionsAgentOutputParser()
        )

        workflow = StateGraph(State)
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tool", self.tool_node)
        workflow.set_entry_point("agent")
        
        workflow.add_conditional_edges(
            "agent",
            lambda x: "tool" if x.get("next") == "tool" else END,
            {
                "tool": "tool",
                END: END,
            },
        )
        workflow.add_edge("tool", "agent")

        self.graph = workflow.compile()

    def agent_node(self, state: State, input: Optional[str] = None):
        messages = state["messages"]
        if input:
            messages.append(HumanMessage(content=input))
        
        tool_names = ", ".join([tool.name for tool in self.tools])
        agent_scratchpad = format_to_openai_function_messages(state.get("intermediate_steps", []))
        
        last_message = messages[-1].content if messages else input
        if not last_message:
            return {
                "messages": messages,
                "next": None,
            }
        
        output = self.agent.invoke({
            "chat_history": messages,
            "input": last_message,
            "tool_names": tool_names,
            "agent_scratchpad": agent_scratchpad,
        })
        
        if output.tool:
            return {
                "messages": messages,
                "next": "tool",
                "tool_input": {
                    "name": output.tool,
                    "arguments": output.tool_input,
                },
            }
        else:
            messages.append(AIMessage(content=output.return_values["output"]))
            return {
                "messages": messages,
                "next": None,
            }

    def tool_node(self, state: State):
        tool_input = state["tool_input"]
        messages = state["messages"]
        
        output = self.tool_executor.invoke(tool_input)
        messages.append(AIMessage(content=str(output)))
        
        return {
            "messages": messages,
            "next": "agent",
            "intermediate_steps": state.get("intermediate_steps", []) + [(tool_input, output)],
        }

    def run(self, user_input):
        state = {"messages": [], "next": "agent"}
        try:
            while state["next"]:
                state = self.graph.invoke(state, {"input": user_input})
                user_input = None  # Clear input after first iteration
            
            if state["messages"]:
                return state["messages"][-1].content
            else:
                return "I'm sorry, but I couldn't generate a response. Could you please try again?"
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return "I encountered an error while processing your request. Could you please try again or rephrase your question?"

def chat_with_tools_agent():
    agent = ToolsAgent(os.getenv("OPENAI_API_KEY"))
    print("Welcome to the Tools Agent. Type 'exit', 'quit', or 'bye' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        response = agent.run(user_input)
        print(f"Agent: {response}")

if __name__ == "__main__":
    chat_with_tools_agent()
