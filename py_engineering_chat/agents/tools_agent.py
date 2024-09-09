import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory
from .base_agent import BaseAgent
from py_engineering_chat.tools.custom_tools import FakeWeatherTool
from py_engineering_chat.tools.shell_command_tool import SafeShellCommandTool

class ToolsAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
        self.is_windows = self.check_if_windows()
        self.ai_shadow_directory = self.create_ai_shadow_directory()
        self.tools = self.get_tools()
        self.agent_executor = self.create_chain()

    def check_if_windows(self):
        return os.getenv('IS_WINDOWS', 'false').lower() == 'true'

    def create_ai_shadow_directory(self):
        ai_shadow_dir = os.getenv('AI_SHADOW_DIRECTORY', './ai_shadow')
        if self.is_windows:
            ai_shadow_dir = ai_shadow_dir.replace('/', '\\')
        os.makedirs(ai_shadow_dir, exist_ok=True)
        return ai_shadow_dir

    def get_file_tools(self):
        toolkit = FileManagementToolkit(root_dir=self.ai_shadow_directory)
        return toolkit.get_tools()

    def get_tools(self):
        #file_tools = self.get_file_tools()
        custom_tools = [SafeShellCommandTool()]
        return  custom_tools

    def create_chain(self):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are an AI assistant with file management capabilities. Help the user with their tasks. The current system is {'Windows' if self.is_windows else 'not Windows'}. Always use the following format:\n\nThought: your thoughts here\nAction: the action to take, should be one of {{tool_names}}\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question"),
            ("human", "tool names: {tool_names}\n\nAction Input: {tools}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            ("human", "Thought: {agent_scratchpad}"),
        ])

        agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(agent=agent, tools=self.tools, memory=memory, verbose=True)

    def chat(self):
        print(f"Welcome to the {self.__class__.__name__} Chat!")
        print("Type 'exit' to end the conversation.")

        session_id = f"{self.__class__.__name__.lower()}_session"

        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            try:
                response = self.agent_executor.invoke({"input": user_input})
                print(f"Agent: {response['output']}")
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")

def chat_with_tools_agent():
    agent = ToolsAgent()
    agent.chat()

if __name__ == "__main__":
    chat_with_tools_agent()
