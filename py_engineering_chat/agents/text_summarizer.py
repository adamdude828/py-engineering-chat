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
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_prompt, verbose=False)

    def summarize(self, content: str) -> str:
        return self.summary_chain.run(content=content)

    def __call__(self, content: str) -> str:
        return self.summarize(content)