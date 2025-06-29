import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any
import numpy as np

# Import the modules to test
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.pdf_utils import PDFFileLoader, extract_text_from_pdf
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.rag_pipeline import RAGPipeline, create_rag_pipeline
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.embedding import EmbeddingModel


class TestTextProcessing:
    """Test text loading and splitting functionality."""
    
    def test_text_file_loader_with_text_file(self):
        """Test loading a simple text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nIt has multiple lines.")
            temp_path = f.name
        
        try:
            loader = TextFileLoader(temp_path)
            documents = loader.load_documents()
            assert len(documents) == 1
            assert "This is a test document" in documents[0]
        finally:
            os.unlink(temp_path)
    
    def test_character_text_splitter(self):
        """Test text splitting functionality."""
        splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "This is a long text that should be split into multiple chunks."
        
        chunks = splitter.split(text)
        assert len(chunks) > 1
        assert len(chunks[0]) <= 20
        
        # Test splitting multiple texts
        texts = [text, "Another text to split"]
        all_chunks = splitter.split_texts(texts)
        assert len(all_chunks) > len(chunks)
    
    def test_text_file_loader_invalid_path(self):
        """Test error handling for invalid paths."""
        with pytest.raises(ValueError):
            loader = TextFileLoader("nonexistent.txt")
            loader.load_documents()


class TestPDFProcessing:
    """Test PDF processing functionality."""
    
    @pytest.mark.skipif(
        not os.path.exists("tests/test_data/sample.pdf"),
        reason="No test PDF available"
    )
    def test_pdf_file_loader(self):
        """Test PDF loading functionality (requires test PDF)."""
        loader = PDFFileLoader("tests/test_data/sample.pdf")
        documents = loader.load_documents()
        assert len(documents) > 0
        assert isinstance(documents[0], str)
        
        metadata = loader.get_metadata()
        assert len(metadata) > 0
        assert 'total_pages' in metadata[0]
    
    def test_pdf_loader_invalid_path(self):
        """Test error handling for invalid PDF paths."""
        with pytest.raises(ValueError):
            loader = PDFFileLoader("nonexistent.pdf")
            loader.load_documents()


class TestVectorDatabase:
    """Test vector database functionality."""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        mock_model = Mock(spec=EmbeddingModel)
        mock_model.get_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_model.async_get_embeddings = AsyncMock(return_value=[
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7]
        ])
        return mock_model
    
    def test_vector_database_initialization(self, mock_embedding_model):
        """Test vector database initialization."""
        db = VectorDatabase(embedding_model=mock_embedding_model)
        assert db.embedding_model == mock_embedding_model
        assert len(db.vectors) == 0
        assert len(db.metadata) == 0
    
    def test_vector_database_insert(self, mock_embedding_model):
        """Test inserting vectors with metadata."""
        db = VectorDatabase(embedding_model=mock_embedding_model)
        
        vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        metadata = {"source": "test", "category": "example"}
        
        db.insert("test document", vector, metadata)
        
        assert len(db.vectors) == 1
        assert "test document" in db.vectors
        assert "test document" in db.metadata
        assert db.metadata["test document"]["source"] == "test"
        assert "inserted_at" in db.metadata["test document"]
    
    def test_vector_database_search(self, mock_embedding_model):
        """Test vector similarity search."""
        db = VectorDatabase(embedding_model=mock_embedding_model)
        
        # Insert test vectors
        vectors = [
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
            np.array([0.3, 0.4, 0.5, 0.6, 0.7])
        ]
        
        texts = ["document 1", "document 2", "document 3"]
        metadata = [
            {"category": "A"},
            {"category": "B"},
            {"category": "A"}
        ]
        
        for text, vector, meta in zip(texts, vectors, metadata):
            db.insert(text, vector, meta)
        
        # Test search
        query_vector = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
        results = db.search(query_vector, k=2, return_metadata=True)
        
        assert len(results) == 2
        assert all(len(result) == 3 for result in results)  # text, score, metadata
    
    def test_vector_database_metadata_filtering(self, mock_embedding_model):
        """Test metadata filtering in search."""
        db = VectorDatabase(embedding_model=mock_embedding_model)
        
        # Insert test vectors with metadata
        db.insert("doc A1", np.array([0.1, 0.2, 0.3]), {"category": "A"})
        db.insert("doc B1", np.array([0.2, 0.3, 0.4]), {"category": "B"})
        db.insert("doc A2", np.array([0.3, 0.4, 0.5]), {"category": "A"})
        
        # Search with metadata filter
        query_vector = np.array([0.2, 0.3, 0.4])
        results = db.search(
            query_vector, k=5, 
            metadata_filter={"category": "A"},
            return_metadata=True
        )
        
        assert len(results) == 2  # Only category A documents
        for text, score, metadata in results:
            assert metadata["category"] == "A"
    
    @pytest.mark.asyncio
    async def test_vector_database_build_from_list(self, mock_embedding_model):
        """Test building database from list of texts."""
        db = VectorDatabase(embedding_model=mock_embedding_model)
        
        texts = ["document 1", "document 2", "document 3"]
        metadata = [
            {"source": "test1"},
            {"source": "test2"},
            {"source": "test3"}
        ]
        
        result_db = await db.abuild_from_list(texts, metadata)
        
        assert result_db == db
        assert len(db.vectors) == 3
        assert len(db.metadata) == 3
        
        # Verify metadata was added correctly
        for text in texts:
            assert text in db.metadata
            assert "chunk_index" in db.metadata[text]
    
    def test_vector_database_stats(self, mock_embedding_model):
        """Test database statistics."""
        db = VectorDatabase(embedding_model=mock_embedding_model)
        
        # Insert test data
        db.insert("test doc", np.array([0.1, 0.2, 0.3]), {"category": "test"})
        
        stats = db.get_stats()
        
        assert stats["total_documents"] == 1
        assert stats["embedding_dimension"] == 3
        assert "metadata_keys" in stats
        assert "average_text_length" in stats


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock language model."""
        mock_llm = Mock(spec=ChatOpenAI)
        mock_llm.run.return_value = "This is a test response based on the provided context."
        return mock_llm
    
    @pytest.fixture
    def mock_vector_db(self):
        """Create a mock vector database."""
        mock_db = Mock(spec=VectorDatabase)
        mock_db.search_by_text.return_value = [
            ("Sample document text", 0.95, {"source": "test.pdf", "page": 1}),
            ("Another document text", 0.87, {"source": "test.pdf", "page": 2})
        ]
        mock_db.get_stats.return_value = {
            "total_documents": 100,
            "embedding_dimension": 1536
        }
        return mock_db
    
    def test_rag_pipeline_initialization(self, mock_llm, mock_vector_db):
        """Test RAG pipeline initialization."""
        pipeline = RAGPipeline(
            llm=mock_llm,
            vector_db=mock_vector_db,
            response_style="detailed",
            include_scores=True
        )
        
        assert pipeline.llm == mock_llm
        assert pipeline.vector_db == mock_vector_db
        assert pipeline.response_style == "detailed"
        assert pipeline.include_scores == True
    
    def test_rag_pipeline_search_documents(self, mock_llm, mock_vector_db):
        """Test document search functionality."""
        pipeline = RAGPipeline(llm=mock_llm, vector_db=mock_vector_db)
        
        results = pipeline.search_documents("test query", k=2)
        
        mock_vector_db.search_by_text.assert_called_once_with(
            query_text="test query",
            k=2,
            return_metadata=True,
            metadata_filter=None
        )
        assert len(results) == 2
    
    def test_rag_pipeline_format_context(self, mock_llm, mock_vector_db):
        """Test context formatting."""
        pipeline = RAGPipeline(
            llm=mock_llm, 
            vector_db=mock_vector_db, 
            include_scores=True
        )
        
        search_results = [
            ("Document 1 text", 0.95, {"source": "doc1"}),
            ("Document 2 text", 0.87, {"source": "doc2"})
        ]
        
        context, metadata_info = pipeline.format_context(search_results)
        
        assert "[Source 1]: Document 1 text" in context
        assert "[Source 2]: Document 2 text" in context
        assert "Source 1: 0.950" in metadata_info
        assert "Source 2: 0.870" in metadata_info
    
    def test_rag_pipeline_generate_response(self, mock_llm, mock_vector_db):
        """Test response generation."""
        pipeline = RAGPipeline(llm=mock_llm, vector_db=mock_vector_db)
        
        response = pipeline.generate_response(
            query="test query",
            context="test context",
            metadata_info="test metadata"
        )
        
        assert response == "This is a test response based on the provided context."
        mock_llm.run.assert_called_once()
        
        # Verify the messages passed to the LLM
        call_args = mock_llm.run.call_args[0][0]  # Get the messages argument
        assert len(call_args) == 2  # system and user messages
        assert call_args[0]["role"] == "system"
        assert call_args[1]["role"] == "user"
    
    def test_rag_pipeline_run_pipeline(self, mock_llm, mock_vector_db):
        """Test complete pipeline run."""
        pipeline = RAGPipeline(
            llm=mock_llm, 
            vector_db=mock_vector_db,
            include_scores=True
        )
        
        result = pipeline.run_pipeline("What is the test query?", k=2)
        
        assert "response" in result
        assert "context" in result
        assert "context_count" in result
        assert "similarity_scores" in result
        assert result["context_count"] == 2
        assert len(result["similarity_scores"]) == 2
    
    def test_rag_pipeline_batch_process(self, mock_llm, mock_vector_db):
        """Test batch processing of queries."""
        pipeline = RAGPipeline(llm=mock_llm, vector_db=mock_vector_db)
        
        queries = ["Query 1", "Query 2", "Query 3"]
        results = pipeline.batch_process(queries, k=2)
        
        assert len(results) == 3
        for result in results:
            assert "response" in result
            assert "context_count" in result
    
    def test_rag_pipeline_with_metadata_filter(self, mock_llm, mock_vector_db):
        """Test pipeline with metadata filtering."""
        pipeline = RAGPipeline(llm=mock_llm, vector_db=mock_vector_db)
        
        metadata_filter = {"category": "research"}
        result = pipeline.run_pipeline(
            "test query", 
            k=2, 
            metadata_filter=metadata_filter
        )
        
        mock_vector_db.search_by_text.assert_called_with(
            query_text="test query",
            k=2,
            return_metadata=True,
            metadata_filter=metadata_filter
        )
        assert result["metadata_filter"] == metadata_filter


class TestRAGIntegration:
    """Integration tests for the complete RAG system."""
    
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OpenAI API key not available"
    )
    @pytest.mark.asyncio
    async def test_end_to_end_rag_pipeline(self):
        """Test complete RAG pipeline with real components (requires API key)."""
        # Create sample documents
        texts = [
            "Python is a versatile programming language used for web development, data science, and automation.",
            "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
            "Natural language processing enables computers to understand and interpret human language."
        ]
        
        metadata = [
            {"topic": "programming", "category": "technology"},
            {"topic": "AI", "category": "technology"},
            {"topic": "NLP", "category": "technology"}
        ]
        
        # Build vector database
        vector_db = VectorDatabase()
        vector_db = await vector_db.abuild_from_list(texts, metadata)
        
        # Create LLM
        llm = ChatOpenAI()
        
        # Create pipeline
        pipeline = RAGPipeline(
            llm=llm,
            vector_db=vector_db,
            response_style="concise",
            include_scores=True
        )
        
        # Test query
        result = pipeline.run_pipeline("What is Python used for?", k=2)
        
        assert "response" in result
        assert result["context_count"] > 0
        assert "programming" in result["response"].lower() or "python" in result["response"].lower()
    
    def test_create_rag_pipeline_convenience_function(self):
        """Test the convenience function for creating RAG pipelines."""
        with patch('aimakerspace.rag_pipeline.ChatOpenAI') as mock_llm_class, \
             patch('aimakerspace.rag_pipeline.VectorDatabase') as mock_db_class:
            
            mock_llm = Mock()
            mock_db = Mock()
            mock_llm_class.return_value = mock_llm
            mock_db_class.return_value = mock_db
            
            pipeline = create_rag_pipeline()
            
            assert isinstance(pipeline, RAGPipeline)
            mock_llm_class.assert_called_once()
            mock_db_class.assert_called_once()


if __name__ == "__main__":
    # Run tests
    print("Running RAG System Tests...")
    
    # Simple test runner for debugging
    import sys
    
    test_classes = [
        TestTextProcessing,
        TestVectorDatabase,
        TestRAGPipeline,
        TestRAGIntegration
    ]
    
    for test_class in test_classes:
        print(f"\n{'='*50}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*50}")
        
        instance = test_class()
        methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for method_name in methods:
            try:
                print(f"  • {method_name}... ", end="")
                method = getattr(instance, method_name)
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()
                print("✓ PASSED")
            except Exception as e:
                print(f"✗ FAILED: {e}")
    
    print(f"\n{'='*50}")
    print("Test run completed!")
    print(f"{'='*50}") 