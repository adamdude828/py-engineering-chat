from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_community.tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def tools_agent():
    llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
    
    # Define file management tools
    read_file = ReadFileTool()
    write_file = WriteFileTool()
    list_directory = ListDirectoryTool()
    
    tools = [
        Tool(
            name="ReadFile",
            func=read_file.run,
            description="Read the contents of a file"
        ),
        Tool(
            name="WriteFile",
            func=write_file.run,
            description="Write content to a file"
        ),
        Tool(
            name="ListDirectory",
            func=list_directory.run,
            description="List files and directories in a specified path"
        ),
    ]
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant capable of using file management tools. Use these tools to help the user with their file-related tasks. Available tools: {tools}"),
        ("human", "The names of the tools available to you are {tool_names}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    tool_names = [tool.name for tool in tools]
    prompt = prompt.partial(tools=", ".join(tool_names))
    
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=False)

def chat_with_tools_agent():
    agent = tools_agent()
    print("Welcome to the Tools Agent Chat!")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        try:
            response = agent.inoke(user_input)
            print(f"Agent: {response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
