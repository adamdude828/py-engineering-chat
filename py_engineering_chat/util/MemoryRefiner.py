import asyncio
import time
from typing import List, Dict, Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from py_engineering_chat.util.logger_util import get_configured_logger
from py_engineering_chat.tiered_memory import TieredMemory

class MemoryRefiner:
    def __init__(self, tiered_memory: TieredMemory, llm: ChatOpenAI, batch_size: int = 5, rate_limit_delay: float = 1.0, refinement_interval: int = 300):
        """
        Initialize the MemoryRefiner.

        Inputs:
        - tiered_memory: TieredMemory - An instance of the TieredMemory class.
        - llm: ChatOpenAI - An instance of the ChatOpenAI class for LLM interactions.
        - batch_size: int - Number of memories to process in each batch.
        - rate_limit_delay: float - Delay in seconds between batches to handle rate limiting.
        - refinement_interval: int - Interval in seconds between refinement cycles.
        """
        self.tiered_memory = tiered_memory
        self.llm = llm
        self.logger = get_configured_logger(__name__)
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.refinement_interval = refinement_interval
        self.refinement_metadata = {}  # Metadata about the refinement process

    async def refine_memories(self, tier: str) -> List[Dict[str, Any]]:
        """
        Refine memories from a specific tier.

        Inputs:
        - tier: str - The tier from which to retrieve memories.

        Outputs:
        - List[Dict[str, Any]] - List of updated memories with refined content.
        """
        self.logger.info(f"Starting refinement of memories from tier: {tier}")
        memories = self.tiered_memory.chroma_db.get_conversations_by_metadata({"tier": tier})
        self.logger.info(f"Retrieved {len(memories)} memories from tier {tier} for refinement")

        refined_memories = []
        total_memories = len(memories)
        if total_memories == 0:
            self.logger.info(f"No memories to refine in tier {tier}")
            return refined_memories

        # Batch the memories
        for i in range(0, total_memories, self.batch_size):
            batch = memories[i:i+self.batch_size]
            self.logger.debug(f"Processing batch {i//self.batch_size +1}/{(total_memories -1)//self.batch_size +1}")
            messages_list = []
            ids_list = []
            for memory in batch:
                prompt = self._create_prompt(memory)
                messages = [HumanMessage(content=prompt)]
                messages_list.append(messages)
                ids_list.append(memory['id'])
            try:
                responses = await self.llm.agenerate(messages_list)
                for memory_id, generation in zip(ids_list, responses.generations):
                    refined_content = generation[0].text.strip()
                    self.update_refined_memory(memory_id, refined_content)
                    refined_memories.append({
                        'id': memory_id,
                        'refined_content': refined_content
                    })
            except Exception as e:
                self.logger.error(f"Error in refining batch starting at index {i}: {e}")
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
        self.logger.info(f"Completed refinement of memories from tier: {tier}")
        return refined_memories

    async def summarize_memory(self, memory: Dict[str, Any]) -> str:
        """
        Use the LLM to generate a concise summary focusing on problem-solving approaches.

        Inputs:
        - memory: Dict[str, Any] - The memory to be summarized.

        Outputs:
        - str - The refined summary of the memory.
        """
        prompt = self._create_prompt(memory)
        try:
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.agenerate([messages])
            refined_content = response.generations[0][0].text.strip()
            return refined_content
        except Exception as e:
            self.logger.error(f"Error in summarizing memory {memory.get('id', 'unknown')}: {e}")
            return ""

    def update_refined_memory(self, memory_id: str, refined_content: str):
        """
        Update the TieredMemory with the refined content.

        Inputs:
        - memory_id: str - The ID of the memory to update.
        - refined_content: str - The refined summary content.
        """
        try:
            # Retrieve the memory from the database
            memory = self.tiered_memory.chroma_db.get_conversation(memory_id)
            if not memory:
                self.logger.error(f"Memory with ID {memory_id} not found.")
                return
            # Update the content while maintaining metadata
            self.tiered_memory.chroma_db.update_conversation(
                memory_id,
                refined_content,
                memory['metadata'],
                memory['embedding']  # Assuming embedding stays the same
            )
            self.logger.info(f"Updated memory {memory_id} with refined content.")
        except Exception as e:
            self.logger.error(f"Error updating memory {memory_id}: {e}")

    async def background_refinement_task(self):
        self.logger.info("Starting background refinement task.")
        while True:
            try:
                for tier in self.tiered_memory.tiers:
                    self.logger.info(f"Starting refinement for tier: {tier}")
                    refined_memories = await self.refine_memories(tier)
                    self.logger.info(f"Refined {len(refined_memories)} memories in tier {tier}")
                    await asyncio.sleep(1)
                self.logger.info(f"Completed refinement cycle for all tiers. Waiting for {self.refinement_interval} seconds before next cycle.")
                await asyncio.sleep(self.refinement_interval)
            except asyncio.CancelledError:
                self.logger.info("Background refinement task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in background refinement task: {e}")
                await asyncio.sleep(60)

    def _create_prompt(self, memory: Dict[str, Any]) -> str:
        """
        Create a prompt for the LLM to summarize the memory.

        Inputs:
        - memory: Dict[str, Any] - The memory content and metadata.

        Outputs:
        - str - The prompt string.
        """
        prompt = f"""
Please read the following text and extract the key problem-solving techniques, key insights, and general approaches used.
Focus on summarizing the methods and strategies rather than specific details.

Text:
{memory['content']}

Summary:
"""
        return prompt

    def update_refinement_params(self, batch_size: int = None, rate_limit_delay: float = None, refinement_interval: int = None):
        if batch_size is not None:
            self.batch_size = batch_size
        if rate_limit_delay is not None:
            self.rate_limit_delay = rate_limit_delay
        if refinement_interval is not None:
            self.refinement_interval = refinement_interval
        self.logger.info(f"Updated refinement parameters: batch_size={self.batch_size}, rate_limit_delay={self.rate_limit_delay}, refinement_interval={self.refinement_interval}")
