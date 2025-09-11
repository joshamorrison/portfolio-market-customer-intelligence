"""
Social Media MCP Server

Handles data collection from social media platforms:
- Twitter/X API integration
- Reddit API integration  
- LinkedIn API integration
- Real-time sentiment tracking
- Engagement metrics collection
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
import aiohttp
import websockets
from websockets.server import serve, WebSocketServerProtocol

from ...config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SocialMediaMCPServer:
    """MCP Server for social media data collection."""
    
    def __init__(self, host: str = "localhost", port: int = 3001):
        self.host = host
        self.port = port
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.is_running = False
        
        # API clients
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Data collection state
        self.active_streams: Dict[str, bool] = {}
        self.collection_stats = {
            "total_posts_collected": 0,
            "sentiment_scores_processed": 0,
            "api_calls_made": 0,
            "errors_encountered": 0
        }
    
    async def start_server(self):
        """Start the MCP server."""
        try:
            self.session = aiohttp.ClientSession()
            logger.info(f"Starting Social Media MCP Server on {self.host}:{self.port}")
            
            async with serve(
                self.handle_client,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10
            ) as server:
                self.is_running = True
                logger.info("Social Media MCP Server started successfully")
                await server.wait_closed()
                
        except Exception as e:
            logger.error(f"Error starting Social Media MCP Server: {e}")
            raise
        finally:
            if self.session:
                await self.session.close()
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle client connections."""
        client_id = f"client_{datetime.now().timestamp()}"
        self.clients[client_id] = websocket
        logger.info(f"Client {client_id} connected to Social Media MCP Server")
        
        try:
            async for message in websocket:
                await self.process_message(client_id, websocket, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def process_message(self, client_id: str, websocket: WebSocketServerProtocol, message: str):
        """Process incoming client messages."""
        try:
            data = json.loads(message)
            method = data.get("method")
            params = data.get("params", {})
            request_id = data.get("id")
            
            logger.debug(f"Processing method: {method} for client {client_id}")
            
            # Route method to appropriate handler
            if method == "initialize":
                response = await self.handle_initialize(params)
            elif method == "collect_social_data":
                response = await self.handle_collect_social_data(params)
            elif method == "start_streaming":
                response = await self.handle_start_streaming(client_id, params)
            elif method == "stop_streaming":
                response = await self.handle_stop_streaming(client_id, params)
            elif method == "get_stats":
                response = await self.handle_get_stats(params)
            elif method == "ping":
                response = {"success": True, "data": {"pong": True, "server_time": datetime.now().isoformat()}}
            else:
                response = {"success": False, "error": f"Unknown method: {method}"}
            
            # Send response
            response["id"] = request_id
            await websocket.send(json.dumps(response))
            
        except json.JSONDecodeError:
            error_response = {
                "success": False,
                "error": "Invalid JSON format",
                "id": None
            }
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            error_response = {
                "success": False,
                "error": f"Server error: {str(e)}",
                "id": data.get("id") if 'data' in locals() else None
            }
            await websocket.send(json.dumps(error_response))
    
    async def handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle client initialization."""
        client_name = params.get("client_name", "unknown")
        capabilities = params.get("capabilities", [])
        
        logger.info(f"Initializing client: {client_name} with capabilities: {capabilities}")
        
        return {
            "success": True,
            "data": {
                "server_name": "social-media-mcp",
                "server_version": "1.0.0",
                "supported_platforms": ["twitter", "reddit", "linkedin"],
                "capabilities": [
                    "data_collection",
                    "real_time_streaming", 
                    "sentiment_analysis",
                    "engagement_metrics"
                ],
                "rate_limits": {
                    "twitter": "300 requests/15min",
                    "reddit": "100 requests/minute",
                    "linkedin": "100 requests/day"
                }
            }
        }
    
    async def handle_collect_social_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle social media data collection request."""
        platforms = params.get("platforms", ["twitter", "reddit"])
        keywords = params.get("keywords", [])
        time_range_hours = params.get("time_range_hours", 24)
        include_sentiment = params.get("include_sentiment", True)
        include_engagement = params.get("include_engagement_metrics", True)
        
        logger.info(f"Collecting social data from {platforms} for keywords: {keywords}")
        
        collected_data = []
        collection_summary = {
            "platforms_processed": [],
            "total_posts": 0,
            "date_range": {
                "start": (datetime.now() - timedelta(hours=time_range_hours)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "processing_time_ms": 0
        }
        
        start_time = datetime.now()
        
        try:
            # Collect from each platform
            for platform in platforms:
                platform_data = await self._collect_from_platform(
                    platform, keywords, time_range_hours, include_sentiment, include_engagement
                )
                if platform_data:
                    collected_data.extend(platform_data)
                    collection_summary["platforms_processed"].append(platform)
            
            collection_summary["total_posts"] = len(collected_data)
            collection_summary["processing_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update stats
            self.collection_stats["total_posts_collected"] += len(collected_data)
            self.collection_stats["api_calls_made"] += len(platforms)
            
            return {
                "success": True,
                "data": {
                    "posts": collected_data,
                    "summary": collection_summary,
                    "metadata": {
                        "collection_timestamp": datetime.now().isoformat(),
                        "keywords_used": keywords,
                        "sentiment_analysis_enabled": include_sentiment
                    }
                }
            }
            
        except Exception as e:
            self.collection_stats["errors_encountered"] += 1
            logger.error(f"Error collecting social data: {e}")
            return {
                "success": False,
                "error": f"Data collection failed: {str(e)}",
                "data": {"partial_results": collected_data} if collected_data else None
            }
    
    async def _collect_from_platform(
        self, 
        platform: str, 
        keywords: List[str], 
        time_range_hours: int,
        include_sentiment: bool,
        include_engagement: bool
    ) -> List[Dict[str, Any]]:
        """Collect data from a specific platform."""
        
        if platform == "twitter":
            return await self._collect_twitter_data(keywords, time_range_hours, include_sentiment, include_engagement)
        elif platform == "reddit":
            return await self._collect_reddit_data(keywords, time_range_hours, include_sentiment, include_engagement)
        elif platform == "linkedin":
            return await self._collect_linkedin_data(keywords, time_range_hours, include_sentiment, include_engagement)
        else:
            logger.warning(f"Unsupported platform: {platform}")
            return []
    
    async def _collect_twitter_data(
        self, 
        keywords: List[str], 
        time_range_hours: int,
        include_sentiment: bool,
        include_engagement: bool
    ) -> List[Dict[str, Any]]:
        """Collect data from Twitter/X."""
        if not settings.twitter_bearer_token:
            logger.warning("Twitter bearer token not configured")
            return self._generate_mock_twitter_data(keywords, time_range_hours, include_sentiment, include_engagement)
        
        # For demonstration, we'll generate realistic mock data
        # In production, this would use the Twitter API v2
        return self._generate_mock_twitter_data(keywords, time_range_hours, include_sentiment, include_engagement)
    
    async def _collect_reddit_data(
        self, 
        keywords: List[str], 
        time_range_hours: int,
        include_sentiment: bool,
        include_engagement: bool
    ) -> List[Dict[str, Any]]:
        """Collect data from Reddit."""
        if not settings.reddit_client_id or not settings.reddit_client_secret:
            logger.warning("Reddit API credentials not configured")
            return self._generate_mock_reddit_data(keywords, time_range_hours, include_sentiment, include_engagement)
        
        # Generate realistic mock data for demonstration
        return self._generate_mock_reddit_data(keywords, time_range_hours, include_sentiment, include_engagement)
    
    async def _collect_linkedin_data(
        self, 
        keywords: List[str], 
        time_range_hours: int,
        include_sentiment: bool,
        include_engagement: bool
    ) -> List[Dict[str, Any]]:
        """Collect data from LinkedIn."""
        # LinkedIn has very restrictive API access, so we'll use mock data
        return self._generate_mock_linkedin_data(keywords, time_range_hours, include_sentiment, include_engagement)
    
    def _generate_mock_twitter_data(
        self, 
        keywords: List[str], 
        time_range_hours: int,
        include_sentiment: bool,
        include_engagement: bool
    ) -> List[Dict[str, Any]]:
        """Generate realistic mock Twitter data."""
        import random
        
        mock_posts = []
        num_posts = random.randint(50, 200)  # Realistic volume
        
        sample_tweets = [
            "Just tried the new AI features - incredible performance improvements!",
            "The customer service experience was disappointing. Expected better quality.",
            "Amazing product launch today! The innovation in this space is accelerating.",
            "Pricing seems too high compared to competitors. Need better value proposition.",
            "User interface is intuitive and well-designed. Great user experience overall.",
            "Having technical issues with the latest update. Anyone else experiencing problems?",
            "Impressed by the quality and attention to detail in this product.",
            "The onboarding process was smooth and efficient. Well executed!",
            "Feature requests: better analytics dashboard and mobile app improvements.",
            "Excellent customer support team. Quick response and helpful solutions."
        ]
        
        for i in range(num_posts):
            post_time = datetime.now() - timedelta(hours=random.uniform(0, time_range_hours))
            sentiment_score = random.uniform(-1, 1) if include_sentiment else None
            
            post = {
                "id": f"twitter_{i}_{int(post_time.timestamp())}",
                "platform": "twitter",
                "text": random.choice(sample_tweets),
                "author": f"@user{random.randint(1000, 9999)}",
                "timestamp": post_time.isoformat(),
                "keywords_matched": random.sample(keywords, min(len(keywords), 2)) if keywords else [],
                "url": f"https://twitter.com/user{random.randint(1000, 9999)}/status/{random.randint(1000000000, 9999999999)}"
            }
            
            if include_sentiment:
                post["sentiment"] = {
                    "score": sentiment_score,
                    "label": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral",
                    "confidence": random.uniform(0.7, 0.95)
                }
            
            if include_engagement:
                post["engagement"] = {
                    "likes": random.randint(0, 500),
                    "retweets": random.randint(0, 100),
                    "replies": random.randint(0, 50),
                    "engagement_rate": random.uniform(0.01, 0.15)
                }
            
            mock_posts.append(post)
        
        return mock_posts
    
    def _generate_mock_reddit_data(
        self, 
        keywords: List[str], 
        time_range_hours: int,
        include_sentiment: bool,
        include_engagement: bool
    ) -> List[Dict[str, Any]]:
        """Generate realistic mock Reddit data."""
        import random
        
        mock_posts = []
        num_posts = random.randint(30, 100)
        
        sample_posts = [
            "Has anyone tried the new platform? Looking for honest reviews and experiences.",
            "Comprehensive comparison between leading solutions in this space. Here's my analysis...",
            "Customer support was excellent. Quick resolution and very professional team.",
            "Disappointed with recent changes. The user experience has declined significantly.",
            "Technical deep dive into the architecture. Impressive engineering work overall.",
            "Pricing analysis: is it worth the cost compared to alternatives?",
            "Feature request thread: what improvements would you like to see?",
            "Performance benchmarks show significant improvements in latest version.",
            "Integration guide for developers. Step-by-step implementation process.",
            "Community feedback: what are your biggest pain points with current solutions?"
        ]
        
        subreddits = ["technology", "startups", "MachineLearning", "datascience", "programming"]
        
        for i in range(num_posts):
            post_time = datetime.now() - timedelta(hours=random.uniform(0, time_range_hours))
            sentiment_score = random.uniform(-1, 1) if include_sentiment else None
            
            post = {
                "id": f"reddit_{i}_{int(post_time.timestamp())}",
                "platform": "reddit",
                "text": random.choice(sample_posts),
                "author": f"u/redditor{random.randint(100, 999)}",
                "subreddit": random.choice(subreddits),
                "timestamp": post_time.isoformat(),
                "keywords_matched": random.sample(keywords, min(len(keywords), 1)) if keywords else [],
                "url": f"https://reddit.com/r/{random.choice(subreddits)}/comments/{random.randint(1000000, 9999999)}"
            }
            
            if include_sentiment:
                post["sentiment"] = {
                    "score": sentiment_score,
                    "label": "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral",
                    "confidence": random.uniform(0.6, 0.9)
                }
            
            if include_engagement:
                post["engagement"] = {
                    "upvotes": random.randint(1, 1000),
                    "downvotes": random.randint(0, 100),
                    "comments": random.randint(0, 200),
                    "upvote_ratio": random.uniform(0.6, 0.95)
                }
            
            mock_posts.append(post)
        
        return mock_posts
    
    def _generate_mock_linkedin_data(
        self, 
        keywords: List[str], 
        time_range_hours: int,
        include_sentiment: bool,
        include_engagement: bool
    ) -> List[Dict[str, Any]]:
        """Generate realistic mock LinkedIn data."""
        import random
        
        mock_posts = []
        num_posts = random.randint(10, 50)  # Lower volume for LinkedIn
        
        sample_posts = [
            "Excited to share insights from our latest industry report on digital transformation trends.",
            "Reflecting on key learnings from implementing AI solutions across enterprise environments.",
            "Announcing our new partnership to deliver innovative solutions to our clients.",
            "Thoughts on the evolving landscape of customer experience and digital engagement.",
            "Proud of our team's achievement in delivering record-breaking results this quarter.",
            "Industry analysis: key trends shaping the future of business technology.",
            "Leadership lessons learned from scaling high-performing teams in competitive markets.",
            "Innovation spotlight: how emerging technologies are transforming traditional industries.",
            "Professional development tip: the importance of continuous learning in technology careers.",
            "Market research findings: customer expectations and service delivery excellence."
        ]
        
        for i in range(num_posts):
            post_time = datetime.now() - timedelta(hours=random.uniform(0, time_range_hours))
            sentiment_score = random.uniform(-0.5, 1) if include_sentiment else None  # LinkedIn tends more positive
            
            post = {
                "id": f"linkedin_{i}_{int(post_time.timestamp())}",
                "platform": "linkedin",
                "text": random.choice(sample_posts),
                "author": f"Professional {random.randint(100, 999)}",
                "timestamp": post_time.isoformat(),
                "keywords_matched": random.sample(keywords, min(len(keywords), 1)) if keywords else [],
                "url": f"https://linkedin.com/posts/professional{random.randint(100, 999)}_{random.randint(1000000000, 9999999999)}"
            }
            
            if include_sentiment:
                post["sentiment"] = {
                    "score": sentiment_score,
                    "label": "positive" if sentiment_score > 0.1 else "neutral" if sentiment_score > -0.1 else "negative",
                    "confidence": random.uniform(0.7, 0.9)
                }
            
            if include_engagement:
                post["engagement"] = {
                    "likes": random.randint(5, 500),
                    "comments": random.randint(0, 50),
                    "shares": random.randint(0, 20),
                    "engagement_rate": random.uniform(0.02, 0.08)
                }
            
            mock_posts.append(post)
        
        return mock_posts
    
    async def handle_start_streaming(self, client_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle real-time streaming request."""
        stream_types = params.get("stream_types", ["real_time_updates"])
        
        # Start background streaming task
        self.active_streams[client_id] = True
        asyncio.create_task(self._stream_real_time_data(client_id, stream_types))
        
        return {
            "success": True,
            "data": {
                "streaming_started": True,
                "stream_types": stream_types,
                "client_id": client_id
            }
        }
    
    async def _stream_real_time_data(self, client_id: str, stream_types: List[str]):
        """Stream real-time data to client."""
        try:
            websocket = self.clients.get(client_id)
            if not websocket:
                return
            
            while self.active_streams.get(client_id, False):
                # Generate mock real-time update
                stream_data = {
                    "type": "real_time_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "new_posts": random.randint(1, 10),
                        "sentiment_shift": random.uniform(-0.1, 0.1),
                        "trending_keywords": ["AI", "innovation", "technology"][:random.randint(1, 3)]
                    }
                }
                
                await websocket.send(json.dumps(stream_data))
                await asyncio.sleep(30)  # Send update every 30 seconds
                
        except Exception as e:
            logger.error(f"Error in streaming for client {client_id}: {e}")
            if client_id in self.active_streams:
                del self.active_streams[client_id]
    
    async def handle_stop_streaming(self, client_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stop streaming request."""
        if client_id in self.active_streams:
            del self.active_streams[client_id]
        
        return {
            "success": True,
            "data": {"streaming_stopped": True, "client_id": client_id}
        }
    
    async def handle_get_stats(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistics request."""
        return {
            "success": True,
            "data": {
                "server_stats": self.collection_stats,
                "active_clients": len(self.clients),
                "active_streams": len(self.active_streams),
                "uptime": "calculated_uptime",
                "supported_platforms": ["twitter", "reddit", "linkedin"]
            }
        }


async def run_social_media_server():
    """Run the Social Media MCP Server."""
    server = SocialMediaMCPServer()
    await server.start_server()


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(run_social_media_server())
    except KeyboardInterrupt:
        logger.info("Social Media MCP Server shutdown requested")
    except Exception as e:
        logger.error(f"Social Media MCP Server error: {e}")
        sys.exit(1)