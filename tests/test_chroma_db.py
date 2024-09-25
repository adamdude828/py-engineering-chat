import pytest
import os
import shutil
from py_engineering_chat.util.chroma_db import ChromaDB

class TestChromaDB:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.test_db_path = os.path.join(os.path.dirname(__file__), 'test_chroma_db')
        os.environ['AI_SHADOW_DIRECTORY'] = self.test_db_path
        self.chroma_db = ChromaDB()
        yield
        shutil.rmtree(self.test_db_path, ignore_errors=True)

    def test_add_and_get_conversation(self):
        conversation_id = "test_conv_1"
        content = "This is a test conversation"
        metadata = {"tier": "recent", "timestamp": 1234567890}
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        self.chroma_db.add_conversation(conversation_id, content, metadata, embedding)
        
        retrieved = self.chroma_db.get_conversation(conversation_id)
        assert retrieved is not None
        assert retrieved['id'] == conversation_id
        assert retrieved['content'] == content
        assert retrieved['metadata'] == metadata
        assert retrieved['embedding'] == [round(val, 6) for val in embedding]

    def test_update_conversation(self):
        conversation_id = "test_conv_2"
        content = "Initial content"
        metadata = {"tier": "recent", "timestamp": 1234567890}
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        self.chroma_db.add_conversation(conversation_id, content, metadata, embedding)
        
        updated_content = "Updated content"
        updated_metadata = {"tier": "medium", "timestamp": 1234567891}
        updated_embedding = [0.2, 0.3, 0.4, 0.5, 0.6]

        self.chroma_db.update_conversation(conversation_id, updated_content, updated_metadata, updated_embedding)
        
        retrieved = self.chroma_db.get_conversation(conversation_id)
        assert retrieved['content'] == updated_content
        assert retrieved['metadata'] == updated_metadata
        assert retrieved['embedding'] == updated_embedding

    def test_delete_conversation(self):
        conversation_id = "test_conv_3"
        content = "Content to be deleted"
        metadata = {"tier": "recent", "timestamp": 1234567890}
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        self.chroma_db.add_conversation(conversation_id, content, metadata, embedding)
        self.chroma_db.delete_conversation(conversation_id)
        
        retrieved = self.chroma_db.get_conversation(conversation_id)
        assert retrieved is None

    def test_search_conversations(self):
        for i in range(10):
            self.chroma_db.add_conversation(
                f"test_conv_{i}",
                f"Content {i}",
                {"tier": "recent", "timestamp": 1234567890 + i},
                [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i]
            )

        query_embedding = [0.5, 1.0, 1.5, 2.0, 2.5]
        results = self.chroma_db.search_conversations(query_embedding, n_results=3)
        
        assert len(results) == 3
        assert all('id' in result for result in results)
        assert all('content' in result for result in results)
        assert all('metadata' in result for result in results)
        assert all('distance' in result for result in results)

    def test_get_conversations_by_metadata(self):
        for i in range(5):
            self.chroma_db.add_conversation(
                f"test_conv_{i}",
                f"Content {i}",
                {"tier": "recent" if i < 3 else "medium", "timestamp": 1234567890 + i},
                [0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i]
            )

        recent_conversations = self.chroma_db.get_conversations_by_metadata({"tier": "recent"})
        assert len(recent_conversations) == 3

        medium_conversations = self.chroma_db.get_conversations_by_metadata({"tier": "medium"})
        assert len(medium_conversations) == 2