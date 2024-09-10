from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from py_engineering_chat.tools.custom_tools import get_tools
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from py_engineering_chat.agents.base_agent import BaseAgent

class ToolsAgent(BaseAgent):
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o",
            openai_api_key=openai_api_key
        ) 
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use tools that are available to assist the user are {tools} and the naes are {tool_names}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        tools = get_tools()
        agent = create_react_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )

    def run(self, user_input):
        return self.agent_executor.invoke({"input": user_input})["output"]

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
