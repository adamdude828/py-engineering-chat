from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from py_engineering_chat.util.chroma_db import ChromaDB
from py_engineering_chat.util.logger_util import get_configured_logger
import time

class ConversationSummarizer:
    def __init__(self):
        self.logger = get_configured_logger(__name__)
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
        self.chroma_db = ChromaDB()
        
        summarize_prompt = ChatPromptTemplate.from_template(
            "Summarize the following conversation in a concise manner, "
            "capturing the main points and any important details:\n\n{conversation}"
        )
        self.summarize_chain = LLMChain(llm=self.llm, prompt=summarize_prompt)

    def summarize_conversation(self, conversation_id: str, conversation: str):
        self.logger.debug(f"Summarizing conversation: {conversation_id}")
        try:
            summary = self.summarize_chain.run(conversation=conversation)
            
            # Generate a simple embedding for the summary (using average of word vectors)
            words = summary.split()
            embedding = [sum(ord(c) for c in word)/len(word) for word in words]
            avg_embedding = [sum(x)/len(embedding) for x in zip(*embedding)] if embedding else [0] * 100
            
            # Store the summary in Chroma
            self.chroma_db.add_conversation(
                conversation_id=conversation_id,
                content=summary,
                metadata={"timestamp": time.time(), "type": "summary"},
                embedding=avg_embedding
            )
            self.logger.info(f"Conversation {conversation_id} summarized and stored in Chroma")
            return summary
        except Exception as e:
            self.logger.error(f"Error summarizing conversation {conversation_id}: {str(e)}")
            return None

    def get_summary(self, conversation_id: str):
        self.logger.debug(f"Retrieving summary for conversation: {conversation_id}")
        return self.chroma_db.get_conversation(conversation_id)

def background_summarization_process():
    summarizer = ConversationSummarizer()
    logger = get_configured_logger(__name__)
    
    while True:
        try:
            # In a real implementation, you would fetch recent conversations that need summarization
            # For now, we'll just log that the process is running
            logger.info("Background summarization process running...")
            time.sleep(300)  # Sleep for 5 minutes before the next check
        except Exception as e:
            logger.error(f"Error in background summarization process: {str(e)}")
            time.sleep(60)  # Sleep for 1 minute before retrying

if __name__ == "__main__":
    background_summarization_process()
