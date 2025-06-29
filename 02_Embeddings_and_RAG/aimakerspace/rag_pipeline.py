from typing import Dict, List, Any, Optional, Tuple
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
from aimakerspace.openai_utils.prompts import SystemRolePrompt, UserRolePrompt
import logging


class RAGPipeline:
    """
    A comprehensive Retrieval Augmented Generation pipeline that combines
    semantic search with AI-powered response generation.
    """
    
    def __init__(
        self,
        llm: ChatOpenAI,
        vector_db: VectorDatabase,
        response_style: str = "detailed",
        include_scores: bool = False,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            llm: The language model for generating responses
            vector_db: Vector database for document retrieval
            response_style: Style of responses ('concise', 'detailed', 'comprehensive')
            include_scores: Whether to include similarity scores in responses
            system_template: Custom system prompt template
            user_template: Custom user prompt template
            logger: Optional logger instance
        """
        self.llm = llm
        self.vector_db = vector_db
        self.response_style = response_style
        self.include_scores = include_scores
        self.logger = logger or logging.getLogger(__name__)
        
        # Default system template
        self.system_template = system_template or """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response."""

        # Default user template
        self.user_template = user_template or """Context Information:
{context}

{metadata_info}

Question: {user_query}

Please provide your answer based solely on the context above."""

        # Initialize prompt templates
        self.system_prompt = SystemRolePrompt(
            self.system_template,
            strict=False,
            defaults={"response_style": self.response_style, "response_length": "appropriate"}
        )
        
        self.user_prompt = UserRolePrompt(
            self.user_template,
            strict=False,
            defaults={"metadata_info": "", "context": ""}
        )

    def search_documents(
        self,
        query: str,
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
        return_metadata: bool = True
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for relevant documents using the vector database.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            metadata_filter: Optional metadata filter
            return_metadata: Whether to return metadata
            
        Returns:
            List of (text, score, metadata) tuples
        """
        try:
            self.logger.debug(f"Searching for documents with query: {query}")
            results = self.vector_db.search_by_text(
                query_text=query,
                k=k,
                return_metadata=return_metadata,
                metadata_filter=metadata_filter
            )
            self.logger.debug(f"Found {len(results)} relevant documents")
            return results
        except Exception as e:
            self.logger.error(f"Error searching documents: {e}")
            raise

    def format_context(
        self,
        search_results: List[Tuple[str, float, Dict[str, Any]]]
    ) -> Tuple[str, str]:
        """
        Format search results into context string and metadata info.
        
        Args:
            search_results: List of (text, score, metadata) tuples
            
        Returns:
            Tuple of (formatted_context, metadata_info)
        """
        context_parts = []
        metadata_parts = []
        
        for i, (text, score, metadata) in enumerate(search_results, 1):
            context_parts.append(f"[Source {i}]: {text}")
            
            if self.include_scores:
                metadata_parts.append(f"Source {i}: {score:.3f}")
            
        context = "\n\n".join(context_parts)
        
        metadata_info = ""
        if metadata_parts:
            metadata_info = f"Relevance scores: {', '.join(metadata_parts)}"
        
        return context, metadata_info

    def generate_response(
        self,
        query: str,
        context: str,
        metadata_info: str = "",
        **system_kwargs
    ) -> str:
        """
        Generate AI response using the LLM.
        
        Args:
            query: User query
            context: Formatted context from search results
            metadata_info: Metadata information string
            **system_kwargs: Additional system prompt arguments
            
        Returns:
            Generated response string
        """
        try:
            # Create system message
            system_params = {
                "response_style": self.response_style,
                "response_length": system_kwargs.get("response_length", "detailed")
            }
            system_message = self.system_prompt.create_message(**system_params)
            
            # Create user message
            user_params = {
                "user_query": query,
                "context": context,
                "metadata_info": metadata_info
            }
            user_message = self.user_prompt.create_message(**user_params)
            
            # Generate response
            self.logger.debug("Generating AI response")
            response = self.llm.run([system_message, user_message])
            self.logger.debug("AI response generated successfully")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise

    def run_pipeline(
        self,
        user_query: str,
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **system_kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline.
        
        Args:
            user_query: User's question
            k: Number of documents to retrieve
            metadata_filter: Optional metadata filter for search
            **system_kwargs: Additional arguments for system prompt
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            self.logger.info(f"Running RAG pipeline for query: {user_query}")
            
            # Step 1: Search for relevant documents
            search_results = self.search_documents(
                query=user_query,
                k=k,
                metadata_filter=metadata_filter,
                return_metadata=True
            )
            
            if not search_results:
                return {
                    "response": "I don't know - no relevant documents found.",
                    "context": [],
                    "context_count": 0,
                    "similarity_scores": None
                }
            
            # Step 2: Format context
            context, metadata_info = self.format_context(search_results)
            
            # Step 3: Generate response
            response = self.generate_response(
                query=user_query,
                context=context,
                metadata_info=metadata_info,
                **system_kwargs
            )
            
            # Step 4: Prepare return data
            similarity_scores = None
            if self.include_scores:
                similarity_scores = [f"Source {i+1}: {score:.3f}" 
                                   for i, (_, score, _) in enumerate(search_results)]
            
            result = {
                "response": response,
                "context": search_results,
                "context_count": len(search_results),
                "similarity_scores": similarity_scores,
                "metadata_filter": metadata_filter,
                "query": user_query
            }
            
            self.logger.info("RAG pipeline completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"RAG pipeline failed: {e}")
            raise

    def batch_process(
        self,
        queries: List[str],
        k: int = 4,
        metadata_filter: Optional[Dict[str, Any]] = None,
        **system_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            k: Number of documents to retrieve per query
            metadata_filter: Optional metadata filter
            **system_kwargs: Additional system prompt arguments
            
        Returns:
            List of result dictionaries
        """
        results = []
        for query in queries:
            try:
                result = self.run_pipeline(
                    user_query=query,
                    k=k,
                    metadata_filter=metadata_filter,
                    **system_kwargs
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process query '{query}': {e}")
                results.append({
                    "response": f"Error processing query: {e}",
                    "context": [],
                    "context_count": 0,
                    "similarity_scores": None,
                    "query": query,
                    "error": str(e)
                })
        return results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline's vector database."""
        return self.vector_db.get_stats()


# For backward compatibility
class RetrievalAugmentedQAPipeline(RAGPipeline):
    """Backward compatibility alias for RAGPipeline."""
    pass


# Convenience function for quick setup
def create_rag_pipeline(
    llm: Optional[ChatOpenAI] = None,
    vector_db: Optional[VectorDatabase] = None,
    **kwargs
) -> RAGPipeline:
    """
    Create a RAG pipeline with default configurations.
    
    Args:
        llm: Optional ChatOpenAI instance
        vector_db: Optional VectorDatabase instance
        **kwargs: Additional arguments for RAGPipeline
        
    Returns:
        Configured RAGPipeline instance
    """
    if llm is None:
        llm = ChatOpenAI()
    
    if vector_db is None:
        vector_db = VectorDatabase()
    
    return RAGPipeline(llm=llm, vector_db=vector_db, **kwargs) 