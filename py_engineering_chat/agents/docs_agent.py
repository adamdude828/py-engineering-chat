from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .base_agent import BaseAgent

class DocsAgent(BaseAgent):
    def create_chain(self):
        llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
        parser = StrOutputParser()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the query if provided:"),
            ("system", "{context}"),
            ("human", "{input}")
        ])
        
        return prompt | llm | parser

def chat_with_docs_agent():
    agent = DocsAgent()
    agent.chat()

if __name__ == "__main__":
    chat_with_docs_agent()