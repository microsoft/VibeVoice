"""
VEMI AI - Tools Module for Real-time Knowledge
===============================================

Provides web search and utility functions for VEMI AI voice assistant.
Created by Alvion Global Solutions.

Features:
- DuckDuckGo web search for real-time information
- Time queries for any timezone
- Fast, low-latency tool execution
- Async support for non-blocking operations
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import re
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Thread pool for running blocking operations
_executor = ThreadPoolExecutor(max_workers=3)


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    data: str
    tool_name: str
    latency_ms: float = 0.0


class WebSearchTool:
    """DuckDuckGo web search tool for real-time information"""
    
    def __init__(self, max_results: int = 3, timeout: int = 8, max_retries: int = 2):
        self.max_results = max_results
        self.timeout = timeout
        self.max_retries = max_retries
        self._ddgs = None
    
    def _get_ddgs(self, force_new: bool = False):
        """Lazy load DDGS to avoid import errors if not installed"""
        if self._ddgs is None or force_new:
            try:
                from duckduckgo_search import DDGS
                self._ddgs = DDGS(timeout=self.timeout)
            except ImportError:
                logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
                return None
        return self._ddgs
    
    def search(self, query: str, max_results: Optional[int] = None) -> ToolResult:
        """
        Perform a web search using DuckDuckGo with retry logic.
        
        Args:
            query: Search query
            max_results: Maximum number of results (default: 3)
            
        Returns:
            ToolResult with search results summary
        """
        import time as time_module
        start = time_module.time()
        
        results = None
        last_error = None
        
        # Try with retries for rate limiting
        for attempt in range(self.max_retries + 1):
            ddgs = self._get_ddgs(force_new=(attempt > 0))
            if ddgs is None:
                return ToolResult(
                    success=False,
                    data="Web search is not available.",
                    tool_name="web_search",
                    latency_ms=0
                )
            
            try:
                results = list(ddgs.text(
                    query, 
                    max_results=max_results or self.max_results,
                    safesearch="moderate"
                ))
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "ratelimit" in error_str or "202" in error_str:
                    if attempt < self.max_retries:
                        logger.warning(f"Rate limited, retrying ({attempt + 1}/{self.max_retries})...")
                        time_module.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                else:
                    break  # Non-rate-limit error, don't retry
        
        if results is None:
            logger.error(f"Web search failed after {self.max_retries + 1} attempts: {last_error}")
            return ToolResult(
                success=False,
                data="Search temporarily unavailable. Please try again.",
                tool_name="web_search",
                latency_ms=(time_module.time() - start) * 1000
            )
        
        if not results:
            return ToolResult(
                success=True,
                data=f"No results found for '{query}'.",
                tool_name="web_search",
                latency_ms=(time_module.time() - start) * 1000
            )
        
        # Format results for LLM consumption
        summary_parts = []
        for i, r in enumerate(results[:self.max_results], 1):
            title = r.get('title', 'No title')
            body = r.get('body', '')[:200]  # Limit body length
            summary_parts.append(f"{i}. {title}: {body}")
        
        summary = "\n".join(summary_parts)
        
        latency = (time_module.time() - start) * 1000
        logger.info(f"Web search for '{query}' completed in {latency:.0f}ms")
        
        return ToolResult(
            success=True,
            data=summary,
            tool_name="web_search",
            latency_ms=latency
        )
    
    async def search_async(self, query: str, max_results: Optional[int] = None) -> ToolResult:
        """Async version of search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.search, query, max_results)


class TimeTool:
    """Tool for getting current time in various timezones"""
    
    # Common timezone mappings
    TIMEZONE_ALIASES = {
        "new york": "America/New_York",
        "nyc": "America/New_York",
        "los angeles": "America/Los_Angeles",
        "la": "America/Los_Angeles",
        "chicago": "America/Chicago",
        "london": "Europe/London",
        "paris": "Europe/Paris",
        "berlin": "Europe/Berlin",
        "tokyo": "Asia/Tokyo",
        "sydney": "Australia/Sydney",
        "mumbai": "Asia/Kolkata",
        "india": "Asia/Kolkata",
        "dubai": "Asia/Dubai",
        "singapore": "Asia/Singapore",
        "hong kong": "Asia/Hong_Kong",
        "shanghai": "Asia/Shanghai",
        "beijing": "Asia/Shanghai",
        "moscow": "Europe/Moscow",
        "toronto": "America/Toronto",
        "vancouver": "America/Vancouver",
        "seattle": "America/Los_Angeles",
        "san francisco": "America/Los_Angeles",
        "miami": "America/New_York",
        "boston": "America/New_York",
        "denver": "America/Denver",
        "phoenix": "America/Phoenix",
        "utc": "UTC",
        "gmt": "UTC",
    }
    
    def get_time(self, location: str = "UTC") -> ToolResult:
        """
        Get current time for a location.
        
        Args:
            location: City name or timezone
            
        Returns:
            ToolResult with formatted time
        """
        import time
        start = time.time()
        
        try:
            # Try to find timezone
            location_lower = location.lower().strip()
            tz_name = self.TIMEZONE_ALIASES.get(location_lower)
            
            if tz_name is None:
                # Try direct timezone name
                tz_name = location if "/" in location else None
            
            if tz_name:
                try:
                    from zoneinfo import ZoneInfo
                    tz = ZoneInfo(tz_name)
                    now = datetime.now(tz)
                    time_str = now.strftime("%I:%M %p on %A, %B %d, %Y")
                    result = f"The current time in {location.title()} is {time_str}."
                except Exception:
                    # Fallback to UTC offset calculation
                    result = self._get_time_fallback(location)
            else:
                result = self._get_time_fallback(location)
            
            latency = (time.time() - start) * 1000
            logger.info(f"Time query for '{location}' completed in {latency:.0f}ms")
            
            return ToolResult(
                success=True,
                data=result,
                tool_name="time",
                latency_ms=latency
            )
            
        except Exception as e:
            logger.error(f"Time query error: {e}")
            return ToolResult(
                success=False,
                data=f"Could not get time for {location}.",
                tool_name="time",
                latency_ms=(time.time() - start) * 1000
            )
    
    def _get_time_fallback(self, location: str) -> str:
        """Fallback time calculation using UTC"""
        now = datetime.utcnow()
        time_str = now.strftime("%I:%M %p UTC")
        return f"I don't have timezone data for {location}, but the current UTC time is {time_str}."
    
    async def get_time_async(self, location: str = "UTC") -> ToolResult:
        """Async version of get_time"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.get_time, location)


class ToolManager:
    """
    Manages tool detection and execution for VEMI AI.
    Detects when user queries need external tools and executes them.
    """
    
    def __init__(self):
        self.web_search = WebSearchTool(max_results=3, timeout=5)
        self.time_tool = TimeTool()
        
        # Patterns for detecting tool needs
        self.time_patterns = [
            r"what(?:'s| is) the (?:current )?time (?:in|at) (.+?)(?:\?|$)",
            r"what time is it (?:in|at) (.+?)(?:\?|$)",
            r"current time (?:in|at) (.+?)(?:\?|$)",
            r"time (?:in|at) (.+?)(?:\?|$)",
            r"what(?:'s| is) the (?:current )?time(?:\?|$)",
            r"what time is it(?:\?|$)",
            r"tell me the time (?:in|at) (.+?)(?:\?|$)",
        ]
        
        self.search_patterns = [
            r"(?:do you know|tell me) (?:about|what is) (.+?)(?:\?|$)",
            r"what(?:'s| is) (.+?)(?:\?|$)",
            r"who(?:'s| is) (.+?)(?:\?|$)",
            r"search (?:for )?(.+?)(?:\?|$)",
            r"look up (.+?)(?:\?|$)",
            r"find (?:information (?:about|on) )?(.+?)(?:\?|$)",
        ]
        
        # Keywords that indicate web search is needed
        self.search_keywords = [
            "latest", "today", "yesterday", "recent",
            "news", "update", "price", "stock", "weather",
            "vivawise", "viva wise", "viva vice",  # Common ASR variations
        ]
        
        # Keywords to exclude from search (handled by LLM)
        self.exclude_keywords = [
            "your name", "who are you", "help me", "how to",
            "headache", "pain", "symptom", "medicine", "doctor",
            "car", "tire", "engine", "brake", "oil",
            "time",  # Exclude time queries from web search
        ]
    
    def detect_tool_need(self, user_message: str) -> Optional[Dict[str, Any]]:
        """
        Detect if a user message requires a tool.
        
        Args:
            user_message: The user's message
            
        Returns:
            Dict with tool info if tool needed, None otherwise
        """
        message_lower = user_message.lower().strip()
        
        # Check for time queries first (highest priority)
        for pattern in self.time_patterns:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                location = match.group(1).strip() if match.lastindex else "UTC"
                return {
                    "tool": "time",
                    "query": location,
                    "original_message": user_message
                }
        
        # Check for explicit search requests
        if any(kw in message_lower for kw in ["search for", "look up", "find information"]):
            for pattern in self.search_patterns:
                match = re.search(pattern, message_lower, re.IGNORECASE)
                if match:
                    query = match.group(1).strip() if match.lastindex else user_message
                    return {
                        "tool": "web_search",
                        "query": query,
                        "original_message": user_message
                    }
        
        # Check if query contains search keywords but not excluded topics
        has_search_keyword = any(kw in message_lower for kw in self.search_keywords)
        has_excluded = any(kw in message_lower for kw in self.exclude_keywords)
        
        if has_search_keyword and not has_excluded:
            # Extract the main query
            for pattern in self.search_patterns:
                match = re.search(pattern, message_lower, re.IGNORECASE)
                if match:
                    query = match.group(1).strip() if match.lastindex else user_message
                    return {
                        "tool": "web_search",
                        "query": query,
                        "original_message": user_message
                    }
        
        return None
    
    def execute_tool(self, tool_info: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool based on detected need.
        
        Args:
            tool_info: Dict from detect_tool_need
            
        Returns:
            ToolResult from execution
        """
        tool_name = tool_info.get("tool")
        query = tool_info.get("query", "")
        
        if tool_name == "time":
            return self.time_tool.get_time(query)
        elif tool_name == "web_search":
            return self.web_search.search(query)
        else:
            return ToolResult(
                success=False,
                data="Unknown tool requested.",
                tool_name="unknown"
            )
    
    async def execute_tool_async(self, tool_info: Dict[str, Any]) -> ToolResult:
        """Async version of execute_tool"""
        tool_name = tool_info.get("tool")
        query = tool_info.get("query", "")
        
        if tool_name == "time":
            return await self.time_tool.get_time_async(query)
        elif tool_name == "web_search":
            return await self.web_search.search_async(query)
        else:
            return ToolResult(
                success=False,
                data="Unknown tool requested.",
                tool_name="unknown"
            )


# Convenience functions
_tool_manager = None

def get_tool_manager() -> ToolManager:
    """Get or create the global tool manager"""
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
    return _tool_manager


def detect_and_execute_tool(user_message: str) -> Optional[ToolResult]:
    """
    Convenience function to detect and execute tool in one call.
    
    Args:
        user_message: User's message
        
    Returns:
        ToolResult if tool was executed, None if no tool needed
    """
    manager = get_tool_manager()
    tool_info = manager.detect_tool_need(user_message)
    
    if tool_info:
        logger.info(f"Tool detected: {tool_info['tool']} for query: {tool_info['query']}")
        return manager.execute_tool(tool_info)
    
    return None


async def detect_and_execute_tool_async(user_message: str) -> Optional[ToolResult]:
    """Async version of detect_and_execute_tool"""
    manager = get_tool_manager()
    tool_info = manager.detect_tool_need(user_message)
    
    if tool_info:
        logger.info(f"Tool detected: {tool_info['tool']} for query: {tool_info['query']}")
        return await manager.execute_tool_async(tool_info)
    
    return None
