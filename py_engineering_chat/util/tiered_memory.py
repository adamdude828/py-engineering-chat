import time
from typing import List, Dict, Any
from py_engineering_chat.util.chroma_db import ChromaDB
from py_engineering_chat.util.logger_util import get_configured_logger

class TieredMemory:
    def __init__(self):
        self.logger = get_configured_logger(__name__)
        self.chroma_db = ChromaDB()
        self.tiers = {
            "recent": {"max_age": 60 * 60, "max_items": 50},  # 1 hour, 50 items
            "medium": {"max_age": 24 * 60 * 60, "max_items": 200},  # 1 day, 200 items
            "long_term": {"max_age": None, "max_items": 1000}  # No time limit, 1000 items
        }

    def add_memory(self, content: str, metadata: Dict[str, Any], embedding: List[float]):
        """Add a new memory to the recent tier."""
        conversation_id = f"memory_{int(time.time())}"
        self.chroma_db.add_conversation(conversation_id, content, metadata, embedding)
        self.logger.info(f"Added new memory to recent tier: {conversation_id}")
        self._manage_tiers()

    def get_context(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve context from all tiers based on relevance."""
        results = self.chroma_db.search_conversations(query_embedding, n_results * 3)  # Get more results to filter
        return self._filter_results(results, n_results)

    def _manage_tiers(self):
        """Manage the movement of memories between tiers."""
        self.logger.debug("Starting _manage_tiers method")
        current_time = time.time()
        for tier, config in self.tiers.items():
            self.logger.debug(f"Processing tier: {tier}")
            if tier == "long_term":
                self.logger.debug("Skipping long_term tier")
                continue  # Long-term memories are not automatically moved or deleted

            memories = self.chroma_db.get_conversations_by_metadata({"tier": tier})
            self.logger.debug(f"Retrieved {len(memories) if memories else 0} memories for tier {tier}")

            if not memories:
                self.logger.debug(f"No memories found for tier {tier}")
                continue

            for memory in memories:
                age = current_time - memory['metadata']['timestamp']
                self.logger.debug(f"Memory age: {age}, max_age: {config['max_age']}")
                if age > config['max_age']:
                    self.logger.debug(f"Moving memory {memory['id']} to next tier")
                    self._move_to_next_tier(memory, tier)

            # Prune excess memories if the tier is over capacity
            self._prune_tier(tier)

        self.logger.debug("Finished _manage_tiers method")

    def _move_to_next_tier(self, memory: Dict[str, Any], current_tier: str):
        """Move a memory to the next tier."""
        next_tier = "medium" if current_tier == "recent" else "long_term"
        self.chroma_db.update_conversation(
            memory['id'],
            memory['content'],
            {**memory['metadata'], "tier": next_tier},
            memory['embedding']
        )
        self.logger.info(f"Moved memory {memory['id']} from {current_tier} to {next_tier} tier")

    def _prune_tier(self, tier: str):
        """Remove excess memories from a tier."""
        self.logger.debug(f"Starting _prune_tier for {tier}")
        memories = self.chroma_db.get_conversations_by_metadata({"tier": tier})
        self.logger.debug(f"Retrieved {len(memories) if memories else 0} memories for pruning in tier {tier}")
        if memories and len(memories) > self.tiers[tier]['max_items']:
            memories_to_remove = sorted(memories, key=lambda x: x['metadata']['timestamp'])[:-self.tiers[tier]['max_items']]
            for memory in memories_to_remove:
                self.chroma_db.delete_conversation(memory['id'])
                self.logger.info(f"Pruned memory {memory['id']} from {tier} tier")
        self.logger.debug(f"Finished _prune_tier for {tier}")

    def _filter_results(self, results: List[Dict[str, Any]], n_results: int) -> List[Dict[str, Any]]:
        """Filter and rank results based on relevance and recency."""
        self.logger.debug(f"Filtering {len(results)} results to {n_results}")
        # Implement a scoring system that considers both similarity and recency
        scored_results = []
        for result in results:
            age = time.time() - result['metadata']['timestamp']
            recency_score = 1 / (1 + age / 86400)  # Decay over 24 hours
            relevance_score = 1 - result['distance']  # Assuming distance is between 0 and 1
            combined_score = (relevance_score + recency_score) / 2
            scored_results.append((combined_score, result))

        # Sort by combined score and return top n_results
        filtered_results = [result for _, result in sorted(scored_results, key=lambda x: x[0], reverse=True)[:n_results]]
        self.logger.debug(f"Returned {len(filtered_results)} filtered results")
        return filtered_results
