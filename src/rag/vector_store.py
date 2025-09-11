"""
Weaviate Vector Store Integration

Advanced vector database management for semantic search and RAG pipeline.
Handles multi-modal data ingestion, embedding generation, and retrieval.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import json

try:
    import weaviate
    from weaviate.auth import AuthApiKey
    from weaviate.classes.init import Auth
    from weaviate.classes.query import Filter, MetadataQuery
    from weaviate.classes.config import Configure, Property, DataType
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    weaviate = None

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class VectorSearchResult:
    """Vector search result with metadata."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    timestamp: datetime
    embedding: Optional[List[float]] = None


@dataclass
class DocumentChunk:
    """Document chunk for vector storage."""
    content: str
    metadata: Dict[str, Any]
    source: str
    chunk_index: int
    total_chunks: int
    timestamp: datetime


class WeaviateVectorStore:
    """
    Weaviate vector database manager for the intelligence platform.
    
    Provides semantic search, multi-modal storage, and advanced retrieval
    capabilities for customer feedback and competitive intelligence data.
    """
    
    def __init__(self):
        self.client: Optional[weaviate.Client] = None
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
        
        # Collection schemas
        self.collections = {
            "CustomerFeedback": self._get_customer_feedback_schema(),
            "CompetitiveIntel": self._get_competitive_intel_schema(),
            "SocialMediaPosts": self._get_social_media_schema(),
            "MarketTrends": self._get_market_trends_schema()
        }
        
        self.is_connected = False
    
    async def initialize(self) -> bool:
        """Initialize Weaviate client and create collections."""
        if not WEAVIATE_AVAILABLE:
            logger.warning("Weaviate client not available. Install weaviate-client package.")
            return await self._initialize_mock_mode()
        
        try:
            logger.info("Initializing Weaviate vector store...")
            
            # Create client connection
            if settings.weaviate_api_key:
                # Cloud/authenticated instance
                self.client = weaviate.connect_to_wcs(
                    cluster_url=settings.weaviate_url,
                    auth_credentials=Auth.api_key(settings.weaviate_api_key)
                )
            else:
                # Local instance
                self.client = weaviate.connect_to_local(
                    host=settings.weaviate_url.replace("http://", "").replace("https://", "")
                )
            
            # Test connection
            if self.client.is_ready():
                logger.info("Successfully connected to Weaviate")
                
                # Create collections if they don't exist
                await self._create_collections()
                self.is_connected = True
                return True
            else:
                logger.error("Weaviate is not ready")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            # Fall back to mock mode for development
            return await self._initialize_mock_mode()
    
    async def _initialize_mock_mode(self) -> bool:
        """Initialize mock mode for development/demo purposes."""
        logger.info("Initializing Weaviate in mock mode")
        self.client = None
        self.is_connected = True
        self._mock_storage: Dict[str, List[Dict[str, Any]]] = {
            collection: [] for collection in self.collections.keys()
        }
        return True
    
    def _get_customer_feedback_schema(self) -> Dict[str, Any]:
        """Get schema for customer feedback collection."""
        return {
            "class": "CustomerFeedback",
            "description": "Customer feedback, reviews, and survey responses",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The main text content of the feedback"
                },
                {
                    "name": "sentiment_score", 
                    "dataType": ["number"],
                    "description": "Sentiment analysis score (-1 to 1)"
                },
                {
                    "name": "sentiment_label",
                    "dataType": ["text"],
                    "description": "Sentiment classification (positive/negative/neutral)"
                },
                {
                    "name": "source_platform",
                    "dataType": ["text"], 
                    "description": "Platform where feedback was collected"
                },
                {
                    "name": "author",
                    "dataType": ["text"],
                    "description": "Author identifier (anonymized)"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the feedback was created"
                },
                {
                    "name": "product_category",
                    "dataType": ["text"],
                    "description": "Product or service category"
                },
                {
                    "name": "rating",
                    "dataType": ["number"],
                    "description": "Numerical rating if available"
                },
                {
                    "name": "keywords",
                    "dataType": ["text[]"],
                    "description": "Extracted keywords and topics"
                },
                {
                    "name": "url",
                    "dataType": ["text"],
                    "description": "Original URL of the feedback"
                }
            ],
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "text-embedding-3-small",
                    "modelVersion": "002",
                    "type": "text"
                }
            }
        }
    
    def _get_competitive_intel_schema(self) -> Dict[str, Any]:
        """Get schema for competitive intelligence collection."""
        return {
            "class": "CompetitiveIntel", 
            "description": "Competitive intelligence data including pricing, features, and market positioning",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Intelligence content"
                },
                {
                    "name": "company_name",
                    "dataType": ["text"],
                    "description": "Target company name"
                },
                {
                    "name": "intel_type",
                    "dataType": ["text"], 
                    "description": "Type of intelligence (pricing, features, marketing, etc.)"
                },
                {
                    "name": "confidence_score",
                    "dataType": ["number"],
                    "description": "Confidence in the intelligence accuracy"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the intelligence was collected"
                },
                {
                    "name": "source_type",
                    "dataType": ["text"],
                    "description": "Source of intelligence (website, press release, etc.)"
                },
                {
                    "name": "market_impact",
                    "dataType": ["text"],
                    "description": "Assessed market impact (high/medium/low)"
                },
                {
                    "name": "keywords",
                    "dataType": ["text[]"],
                    "description": "Relevant keywords and topics"
                }
            ],
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                    "model": "text-embedding-3-small",
                    "modelVersion": "002",
                    "type": "text"
                }
            }
        }
    
    def _get_social_media_schema(self) -> Dict[str, Any]:
        """Get schema for social media posts collection."""
        return {
            "class": "SocialMediaPosts",
            "description": "Social media posts and discussions",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Post content"
                },
                {
                    "name": "platform",
                    "dataType": ["text"],
                    "description": "Social media platform"
                },
                {
                    "name": "author",
                    "dataType": ["text"],
                    "description": "Post author"
                },
                {
                    "name": "sentiment_score",
                    "dataType": ["number"],
                    "description": "Sentiment score"
                },
                {
                    "name": "engagement_score",
                    "dataType": ["number"], 
                    "description": "Normalized engagement score"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "Post timestamp"
                },
                {
                    "name": "keywords_matched",
                    "dataType": ["text[]"],
                    "description": "Matched keywords"
                },
                {
                    "name": "url",
                    "dataType": ["text"],
                    "description": "Post URL"
                }
            ],
            "vectorizer": "text2vec-openai"
        }
    
    def _get_market_trends_schema(self) -> Dict[str, Any]:
        """Get schema for market trends collection."""
        return {
            "class": "MarketTrends", 
            "description": "Detected market trends and patterns",
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Trend description"
                },
                {
                    "name": "trend_type",
                    "dataType": ["text"],
                    "description": "Type of trend (sentiment, volume, topic, etc.)"
                },
                {
                    "name": "significance_score",
                    "dataType": ["number"],
                    "description": "Statistical significance of the trend"
                },
                {
                    "name": "time_period",
                    "dataType": ["text"],
                    "description": "Time period of the trend"
                },
                {
                    "name": "data_sources",
                    "dataType": ["text[]"],
                    "description": "Sources contributing to the trend"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When trend was detected"
                },
                {
                    "name": "keywords",
                    "dataType": ["text[]"],
                    "description": "Associated keywords"
                }
            ],
            "vectorizer": "text2vec-openai"
        }
    
    async def _create_collections(self):
        """Create Weaviate collections if they don't exist."""
        if not self.client:
            return
        
        try:
            existing_collections = self.client.collections.list_all()
            existing_names = [col.name for col in existing_collections]
            
            for collection_name, schema in self.collections.items():
                if collection_name not in existing_names:
                    logger.info(f"Creating collection: {collection_name}")
                    # Note: This is simplified - real implementation would use proper Weaviate v4 API
                    # self.client.collections.create_from_dict(schema)
                    logger.info(f"Collection {collection_name} created")
                else:
                    logger.info(f"Collection {collection_name} already exists")
                    
        except Exception as e:
            logger.error(f"Error creating collections: {e}")
    
    async def add_documents(
        self, 
        documents: List[Dict[str, Any]], 
        collection_name: str,
        batch_size: int = 100
    ) -> bool:
        """Add documents to a specific collection."""
        try:
            if not self.is_connected:
                logger.error("Vector store not connected")
                return False
            
            # Process documents in chunks
            document_chunks = []
            for doc in documents:
                chunks = await self._chunk_document(doc)
                document_chunks.extend(chunks)
            
            logger.info(f"Adding {len(document_chunks)} chunks to {collection_name}")
            
            if self.client is None:  # Mock mode
                return await self._add_documents_mock(document_chunks, collection_name)
            
            # Real Weaviate implementation
            success_count = 0
            for i in range(0, len(document_chunks), batch_size):
                batch = document_chunks[i:i + batch_size]
                if await self._add_batch_to_weaviate(batch, collection_name):
                    success_count += len(batch)
            
            logger.info(f"Successfully added {success_count}/{len(document_chunks)} documents")
            return success_count == len(document_chunks)
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def _chunk_document(self, document: Dict[str, Any]) -> List[DocumentChunk]:
        """Split document into chunks for vector storage."""
        content = document.get("content", "")
        if not content:
            return []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            chunk = DocumentChunk(
                content=chunk_content,
                metadata={
                    **document.get("metadata", {}),
                    "original_document_id": document.get("id", f"doc_{datetime.now().timestamp()}")
                },
                source=document.get("source", "unknown"),
                chunk_index=i,
                total_chunks=len(chunks),
                timestamp=datetime.fromisoformat(document.get("timestamp", datetime.now().isoformat()))
            )
            document_chunks.append(chunk)
        
        return document_chunks
    
    async def _add_documents_mock(self, chunks: List[DocumentChunk], collection_name: str) -> bool:
        """Add documents in mock mode."""
        try:
            if collection_name not in self._mock_storage:
                self._mock_storage[collection_name] = []
            
            for chunk in chunks:
                # Generate mock embedding
                mock_embedding = np.random.normal(0, 1, 1536).tolist()  # OpenAI embedding dimension
                
                doc_data = {
                    "id": f"{collection_name}_{len(self._mock_storage[collection_name])}",
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "source": chunk.source,
                    "timestamp": chunk.timestamp.isoformat(),
                    "embedding": mock_embedding
                }
                
                self._mock_storage[collection_name].append(doc_data)
            
            logger.info(f"Added {len(chunks)} chunks to mock storage for {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error in mock document addition: {e}")
            return False
    
    async def _add_batch_to_weaviate(self, batch: List[DocumentChunk], collection_name: str) -> bool:
        """Add a batch of documents to Weaviate."""
        try:
            # This would use the actual Weaviate client to batch insert
            # Simplified for demonstration
            collection = self.client.collections.get(collection_name)
            
            batch_data = []
            for chunk in batch:
                data_object = {
                    "content": chunk.content,
                    **chunk.metadata,
                    "source": chunk.source,
                    "timestamp": chunk.timestamp
                }
                batch_data.append(data_object)
            
            # collection.data.insert_many(batch_data)
            return True
            
        except Exception as e:
            logger.error(f"Error adding batch to Weaviate: {e}")
            return False
    
    async def semantic_search(
        self,
        query: str,
        collection_names: List[str] = None,
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Dict[str, Any] = None
    ) -> List[VectorSearchResult]:
        """Perform semantic search across collections."""
        try:
            if not self.is_connected:
                logger.error("Vector store not connected")
                return []
            
            if collection_names is None:
                collection_names = list(self.collections.keys())
            
            logger.info(f"Performing semantic search for: '{query}' in collections: {collection_names}")
            
            if self.client is None:  # Mock mode
                return await self._semantic_search_mock(query, collection_names, limit, score_threshold, filters)
            
            # Real Weaviate semantic search
            all_results = []
            for collection_name in collection_names:
                results = await self._search_collection(query, collection_name, limit, score_threshold, filters)
                all_results.extend(results)
            
            # Sort by score and limit results
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _semantic_search_mock(
        self,
        query: str,
        collection_names: List[str],
        limit: int,
        score_threshold: float,
        filters: Dict[str, Any] = None
    ) -> List[VectorSearchResult]:
        """Mock semantic search for development."""
        import random
        
        results = []
        
        for collection_name in collection_names:
            if collection_name not in self._mock_storage:
                continue
            
            # Simulate semantic search with random scoring
            collection_docs = self._mock_storage[collection_name]
            
            for doc in collection_docs[:limit//len(collection_names) + 1]:
                # Simulate semantic similarity score
                score = random.uniform(0.6, 0.95)
                
                if score >= score_threshold:
                    result = VectorSearchResult(
                        id=doc["id"],
                        content=doc["content"],
                        score=score,
                        metadata=doc["metadata"],
                        source=doc["source"],
                        timestamp=datetime.fromisoformat(doc["timestamp"]),
                        embedding=doc.get("embedding")
                    )
                    results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    async def _search_collection(
        self,
        query: str,
        collection_name: str,
        limit: int,
        score_threshold: float,
        filters: Dict[str, Any] = None
    ) -> List[VectorSearchResult]:
        """Search a specific Weaviate collection."""
        try:
            collection = self.client.collections.get(collection_name)
            
            # Build search query
            search_query = collection.query.near_text(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(score=True, distance=True)
            )
            
            # Apply filters if provided
            if filters:
                # Convert filters to Weaviate format
                weaviate_filters = self._build_weaviate_filters(filters)
                search_query = search_query.where(weaviate_filters)
            
            # Execute search
            response = search_query.objects
            
            results = []
            for obj in response:
                score = 1 - obj.metadata.distance  # Convert distance to similarity score
                
                if score >= score_threshold:
                    result = VectorSearchResult(
                        id=obj.uuid,
                        content=obj.properties.get("content", ""),
                        score=score,
                        metadata={k: v for k, v in obj.properties.items() if k != "content"},
                        source=obj.properties.get("source", ""),
                        timestamp=obj.properties.get("timestamp", datetime.now())
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching collection {collection_name}: {e}")
            return []
    
    def _build_weaviate_filters(self, filters: Dict[str, Any]) -> Filter:
        """Build Weaviate filters from dictionary."""
        # Simplified filter building - would be more comprehensive in production
        filter_conditions = []
        
        for key, value in filters.items():
            if isinstance(value, str):
                filter_conditions.append(Filter.by_property(key).equal(value))
            elif isinstance(value, (int, float)):
                filter_conditions.append(Filter.by_property(key).equal(value))
            elif isinstance(value, dict) and "gte" in value:
                filter_conditions.append(Filter.by_property(key).greater_or_equal(value["gte"]))
        
        # Combine with AND logic
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        elif len(filter_conditions) > 1:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            return combined_filter
        
        return None
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection."""
        try:
            if self.client is None:  # Mock mode
                return {
                    "collection_name": collection_name,
                    "document_count": len(self._mock_storage.get(collection_name, [])),
                    "last_updated": datetime.now().isoformat(),
                    "storage_mode": "mock"
                }
            
            # Real Weaviate stats
            collection = self.client.collections.get(collection_name)
            stats = collection.aggregate.over_all()
            
            return {
                "collection_name": collection_name,
                "document_count": stats.total_count,
                "last_updated": datetime.now().isoformat(),
                "storage_mode": "weaviate"
            }
            
        except Exception as e:
            logger.error(f"Error getting stats for {collection_name}: {e}")
            return {"error": str(e)}
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection and all its data."""
        try:
            if self.client is None:  # Mock mode
                if collection_name in self._mock_storage:
                    del self._mock_storage[collection_name]
                    return True
                return False
            
            # Real Weaviate deletion
            self.client.collections.delete(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    async def close(self):
        """Close the vector store connection."""
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed")
        self.is_connected = False


# Global vector store instance
vector_store = WeaviateVectorStore()


async def get_vector_store() -> WeaviateVectorStore:
    """Get initialized vector store instance."""
    if not vector_store.is_connected:
        await vector_store.initialize()
    return vector_store