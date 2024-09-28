import os
from abc import ABC, abstractmethod
import re
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from pathlib import Path
from py_engineering_chat.util.chat_settings_manager import ChatSettingsManager
from py_engineering_chat.research.scan_codebase import scan_codebase
from langchain_core.messages import AIMessage, HumanMessage
from py_engineering_chat.util.command_parser import parse_commands

class BaseAgent(ABC):
    def __init__(self):
        self.store = {}
        
        # Load environment variables
        root_dir = Path(__file__).resolve().parents[2]
        env_path = root_dir / '.env'
        print(env_path)
        load_dotenv(dotenv_path=env_path)
        
        # Get AI_SHADOW_DIRECTORY from environment variables
        self.shadow_dir = os.getenv('AI_SHADOW_DIRECTORY')
        if not self.shadow_dir:
            raise ValueError("AI_SHADOW_DIRECTORY environment variable is not set")
        
        self.shadow_path = Path(self.shadow_dir)
        self.shadow_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        chroma_db_path = self.shadow_path / '.chroma_db'
        self.client = chromadb.PersistentClient(path=str(chroma_db_path))
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.settings_manager = ChatSettingsManager()

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def search_context(self, collection_name, query):
        try:
            collection = self.client.get_collection(name=collection_name)
            query_embedding = self.model.encode([query]).tolist()
            results = collection.query(query_embeddings=query_embedding, n_results=3)
            return results['documents'][0]
        except Exception as e:
            print(f"Error searching context: {str(e)}")
            return []

    def process_input(self, inputs):
        user_input = inputs['input']
        context_data = parse_commands(user_input, self.settings_manager)
        inputs['context'] = context_data['context']
        return inputs

    def create_prompt(self, prompt_type: str, inputs: Dict[str, Any]) -> str:
        # Basic implementation for creating prompts
        if prompt_type == "default":
            return f"Context: {inputs.get('context', '')}\nQuery: {inputs.get('input', '')}"
        else:
            return f"Unsupported prompt type: {prompt_type}"

    def select_model(self, task: str) -> ChatOpenAI:
        # Default model selection
        return ChatOpenAI(temperature=0, model="gpt-4-0125-preview")

    def process_structured_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Basic implementation for processing structured input
        return input_data

    def process_unstructured_input(self, input_text: str) -> Dict[str, Any]:
        # Basic implementation for processing unstructured input
        return {"input": input_text}

    def create_chain(self):
        llm = self.select_model("default")
        parser = StrOutputParser()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the query if provided:"),
            ("system", "{context}"),
            ("human", "{input}")
        ])
        
        return prompt | llm | parser

    def agent_function(self, x, config=None):
        chain = self.create_chain()
        with_message_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
        
        if config:
            return with_message_history.invoke(self.process_input(x), config=config)
        else:
            return with_message_history.invoke(self.process_input(x))

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
                response = self.agent_function(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                
                print(f"Agent: {response}")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

# Add this if you want to be able to run the BaseAgent directly
if __name__ == "__main__":
    agent = BaseAgent()
    agent.chat()