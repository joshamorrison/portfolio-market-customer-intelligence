"""
MCP (Model Context Protocol) Client

Manages connections to multiple MCP servers for different data sources:
- Social Media MCP Server (Twitter, LinkedIn, Reddit)  
- Review Platforms MCP Server (G2, TrustPilot, Google Reviews)
- Competitive Intelligence MCP Server (Pricing, Features, Campaigns)
- Survey Data MCP Server (Customer Feedback, NPS)
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from ..config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class MCPRequest:
    """MCP request structure."""
    method: str
    params: Dict[str, Any]
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.request_id is None:
            self.request_id = f"req_{datetime.now().timestamp()}"
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass 
class MCPResponse:
    """MCP response structure."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    request_id: Optional[str] = None
    response_time_ms: Optional[float] = None


class MCPClient:
    """
    MCP Client for connecting to multiple data source servers.
    
    Handles connection management, request routing, and response processing
    for all MCP servers in the intelligence platform.
    """
    
    def __init__(self):
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.server_endpoints = {
            "social_media": settings.mcp_social_media_endpoint,
            "review_platforms": settings.mcp_review_platforms_endpoint, 
            "competitive_intel": settings.mcp_competitive_intel_endpoint
        }
        self.is_initialized = False
        self.request_timeout = 30.0  # seconds
        
    async def initialize(self) -> bool:
        """Initialize connections to all MCP servers."""
        try:
            logger.info("Initializing MCP client connections...")
            
            # Connect to all MCP servers
            connection_tasks = []
            for server_name, endpoint in self.server_endpoints.items():
                task = self._connect_to_server(server_name, endpoint)
                connection_tasks.append(task)
            
            # Wait for all connections
            results = await asyncio.gather(*connection_tasks, return_exceptions=True)
            
            # Check results
            successful_connections = 0
            for i, result in enumerate(results):
                server_name = list(self.server_endpoints.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to connect to {server_name}: {result}")
                else:
                    successful_connections += 1
                    logger.info(f"Successfully connected to {server_name}")
            
            self.is_initialized = successful_connections > 0
            logger.info(f"MCP client initialized with {successful_connections}/{len(self.server_endpoints)} servers")
            
            return self.is_initialized
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            return False
    
    async def _connect_to_server(self, server_name: str, endpoint: str) -> bool:
        """Connect to a specific MCP server."""
        try:
            # Create WebSocket connection
            websocket = await websockets.connect(
                endpoint,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connections[server_name] = websocket
            
            # Send initialization handshake
            init_request = MCPRequest(
                method="initialize",
                params={
                    "client_name": "market-intelligence-platform",
                    "client_version": "1.0.0",
                    "capabilities": ["data_collection", "real_time_streaming"]
                }
            )
            
            response = await self._send_request(server_name, init_request)
            if not response.success:
                logger.error(f"Failed to initialize {server_name}: {response.error}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to {server_name} at {endpoint}: {e}")
            return False
    
    async def _send_request(self, server_name: str, request: MCPRequest) -> MCPResponse:
        """Send request to specific MCP server."""
        if server_name not in self.connections:
            return MCPResponse(
                success=False,
                error=f"No connection to server: {server_name}",
                request_id=request.request_id
            )
        
        try:
            websocket = self.connections[server_name]
            start_time = datetime.now()
            
            # Send request
            request_data = {
                "method": request.method,
                "params": request.params,
                "id": request.request_id,
                "timestamp": request.timestamp.isoformat()
            }
            
            await websocket.send(json.dumps(request_data))
            
            # Wait for response
            response_raw = await asyncio.wait_for(
                websocket.recv(),
                timeout=self.request_timeout
            )
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            response_data = json.loads(response_raw)
            
            return MCPResponse(
                success=response_data.get("success", True),
                data=response_data.get("data"),
                error=response_data.get("error"),
                request_id=response_data.get("id"),
                response_time_ms=response_time
            )
            
        except asyncio.TimeoutError:
            return MCPResponse(
                success=False,
                error=f"Request timeout ({self.request_timeout}s)",
                request_id=request.request_id
            )
        except (ConnectionClosed, WebSocketException) as e:
            logger.error(f"WebSocket error for {server_name}: {e}")
            # Attempt reconnection
            await self._reconnect_server(server_name)
            return MCPResponse(
                success=False,
                error=f"Connection error: {e}",
                request_id=request.request_id
            )
        except Exception as e:
            logger.error(f"Unexpected error sending request to {server_name}: {e}")
            return MCPResponse(
                success=False,
                error=f"Unexpected error: {e}",
                request_id=request.request_id
            )
    
    async def collect_social_media_data(
        self, 
        platforms: List[str] = None,
        keywords: List[str] = None,
        time_range_hours: int = 24
    ) -> MCPResponse:
        """Collect data from social media platforms."""
        if platforms is None:
            platforms = ["twitter", "reddit", "linkedin"]
            
        request = MCPRequest(
            method="collect_social_data",
            params={
                "platforms": platforms,
                "keywords": keywords or [],
                "time_range_hours": time_range_hours,
                "include_sentiment": True,
                "include_engagement_metrics": True
            }
        )
        
        return await self._send_request("social_media", request)
    
    async def collect_review_data(
        self,
        platforms: List[str] = None,
        company_names: List[str] = None,
        product_names: List[str] = None
    ) -> MCPResponse:
        """Collect review data from review platforms."""
        if platforms is None:
            platforms = ["g2", "trustpilot", "google_reviews"]
            
        request = MCPRequest(
            method="collect_review_data",
            params={
                "platforms": platforms,
                "company_names": company_names or [],
                "product_names": product_names or [],
                "include_ratings": True,
                "include_full_text": True
            }
        )
        
        return await self._send_request("review_platforms", request)
    
    async def collect_competitive_intel(
        self,
        competitor_companies: List[str],
        intel_types: List[str] = None
    ) -> MCPResponse:
        """Collect competitive intelligence data."""
        if intel_types is None:
            intel_types = ["pricing", "features", "marketing_campaigns", "job_postings"]
            
        request = MCPRequest(
            method="collect_competitive_intel",
            params={
                "companies": competitor_companies,
                "intel_types": intel_types,
                "include_historical": True,
                "include_analysis": True
            }
        )
        
        return await self._send_request("competitive_intel", request)
    
    async def stream_real_time_data(
        self,
        data_sources: List[str],
        callback_func: callable
    ) -> bool:
        """Start real-time data streaming from multiple sources."""
        try:
            # Start streaming from each requested source
            streaming_tasks = []
            
            for source in data_sources:
                if source in self.connections:
                    task = self._start_stream(source, callback_func)
                    streaming_tasks.append(task)
            
            if streaming_tasks:
                await asyncio.gather(*streaming_tasks)
                return True
            else:
                logger.warning("No valid streaming sources found")
                return False
                
        except Exception as e:
            logger.error(f"Error starting real-time streams: {e}")
            return False
    
    async def _start_stream(self, server_name: str, callback_func: callable):
        """Start streaming from a specific server."""
        request = MCPRequest(
            method="start_streaming",
            params={
                "stream_types": ["real_time_updates", "sentiment_changes", "trend_alerts"],
                "callback_endpoint": "internal"
            }
        )
        
        response = await self._send_request(server_name, request)
        if response.success:
            # Start listening for streaming data
            websocket = self.connections[server_name]
            
            try:
                async for message in websocket:
                    stream_data = json.loads(message)
                    await callback_func(server_name, stream_data)
            except Exception as e:
                logger.error(f"Error in stream from {server_name}: {e}")
    
    async def _reconnect_server(self, server_name: str) -> bool:
        """Attempt to reconnect to a specific server."""
        try:
            endpoint = self.server_endpoints[server_name]
            if server_name in self.connections:
                await self.connections[server_name].close()
                del self.connections[server_name]
            
            return await self._connect_to_server(server_name, endpoint)
        except Exception as e:
            logger.error(f"Failed to reconnect to {server_name}: {e}")
            return False
    
    async def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all MCP servers."""
        status = {}
        
        for server_name in self.server_endpoints.keys():
            if server_name in self.connections:
                # Send ping to check connection
                request = MCPRequest(method="ping", params={})
                response = await self._send_request(server_name, request)
                
                status[server_name] = {
                    "connected": response.success,
                    "response_time_ms": response.response_time_ms,
                    "endpoint": self.server_endpoints[server_name],
                    "last_error": response.error if not response.success else None
                }
            else:
                status[server_name] = {
                    "connected": False,
                    "endpoint": self.server_endpoints[server_name],
                    "error": "No connection established"
                }
        
        return status
    
    async def close_all_connections(self):
        """Close all MCP server connections."""
        logger.info("Closing all MCP connections...")
        
        close_tasks = []
        for server_name, websocket in self.connections.items():
            close_tasks.append(self._close_connection(server_name, websocket))
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self.connections.clear()
        self.is_initialized = False
        
    async def _close_connection(self, server_name: str, websocket):
        """Close a specific connection."""
        try:
            await websocket.close()
            logger.info(f"Closed connection to {server_name}")
        except Exception as e:
            logger.error(f"Error closing connection to {server_name}: {e}")


# Global MCP client instance
mcp_client = MCPClient()


async def get_mcp_client() -> MCPClient:
    """Get initialized MCP client instance."""
    if not mcp_client.is_initialized:
        await mcp_client.initialize()
    return mcp_client