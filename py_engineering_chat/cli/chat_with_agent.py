import time
from langchain.agents import AgentExecutor
from langchain.schema import HumanMessage, AIMessage

def chat_with_agent(agent_executor: AgentExecutor):
    print("Welcome to the LangChain Agent Chat!")
    print("Type 'exit' to end the conversation.")
    
    message_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        human_message = HumanMessage(content=user_input)
        
        try:
            response = agent_executor.invoke(human_message, config={"session_id": "local_session"})
            
            if isinstance(response, dict) and 'output' in response:
                agent_response = response['output']
            else:
                agent_response = str(response)
            
            print(f"Agent: {agent_response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")