from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class TextSummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", verbose=False)
        self.summary_prompt = ChatPromptTemplate.from_template(
            "You are an AI assistant focused on summarizing text for programming tasks. "
            "Summarize the following text, highlighting key concepts, important details, "
            "and any relevant code examples or syntax:\n\n{content}"
        )
        self.summary_chain = self.summary_prompt | self.llm

    def summarize(self, content: str) -> str:
        result = self.summary_chain.invoke({"content": content})
        
        if isinstance(result, dict) and 'content' in result:
            return result['content']
        elif hasattr(result, 'content'):
            return result.content
        else:
            print(f"Unexpected result type: {type(result)}")
            return str(result)

    def __call__(self, content: str) -> str:
        return self.summarize(content)