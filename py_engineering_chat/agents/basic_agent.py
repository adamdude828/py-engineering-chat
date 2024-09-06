from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def basic_agent():
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    parser = StrOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the following query:"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | parser
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )
    
    return with_message_history

def chat_with_basic_agent():
    agent = basic_agent()
    print("Welcome to the Basic Agent Chat!")
    print("Type 'exit' to end the conversation.")
    
    session_id = "basic_agent_session"
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        try:
            response = agent.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            print(f"Agent: {response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")