from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from py_engineering_chat.agents.base_agent import BaseAgent

class DocsAgent(BaseAgent):
    def create_chain(self):
        llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("You are a helpful assistant. Use the following context to answer the query."),
            SystemMessagePromptTemplate.from_template("{context}"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])
        return LLMChain(llm=llm, prompt=prompt)

def chat_with_docs_agent():
    agent = DocsAgent()
    agent.chat()
