import os
import sys
import json

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()


# Initialize FastMCP server
server = FastMCP(
    "tavily-search-mcp",
    instructions="""
# Tavily Search MCP Server

Tavily is an AI-optimized search engine that provides accurate, relevant search results specifically designed for AI applications and LLMs.

## Available Tools

### 1. tavily_search
Search with Tavily and get AI-optimized search results with high-quality, relevant content from the web.

### 2. tavily_qna_search
Get a quick answer to a question using Tavily's Q&A mode, which provides a direct answer along with supporting sources.

## Output Format

All search results will be formatted as text with clear sections for each
result item, including:

- Tavily Search: Title, URL, Content, Score
- Tavily Q&A: Direct answer with source URLs

If the API key is missing or invalid, appropriate error messages will be returned.
""",
)


@server.tool()
async def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_domains: list[str] = None,
    exclude_domains: list[str] = None
) -> str:
    """Search with Tavily and get AI-optimized search results with high-quality, relevant content.

    Args:
        query: Search query (required)
        search_depth: Depth of search - "basic" (faster) or "advanced" (more comprehensive). Default is "basic"
        max_results: Number of results to return (1-10, default 5)
        include_domains: List of domains to specifically include in search (optional)
        exclude_domains: List of domains to exclude from search (optional)
    """
    # Get API key from environment
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

    if not tavily_api_key:
        return (
            "Error: Tavily API key is not configured. Please set the "
            "TAVILY_API_KEY environment variable.\n"
            "Get your API key from: https://tavily.com"
        )

    # API endpoint
    endpoint = "https://api.tavily.com/search"

    try:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False
        }

        # Add optional filters
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint, json=payload, timeout=30.0
            )

            response.raise_for_status()
            data = response.json()

            if "results" not in data or not data["results"]:
                return "No results found."

            results = []
            for idx, result in enumerate(data["results"], 1):
                results.append(
                    f"Result {idx}:\n"
                    f"Title: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"Content: {result.get('content', 'N/A')}\n"
                    f"Score: {result.get('score', 'N/A')}"
                )

            return "\n\n".join(results)

    except httpx.HTTPStatusError as e:
        return f"Tavily API HTTP error occurred: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        return f"Error communicating with Tavily API: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


@server.tool()
async def tavily_qna_search(query: str, search_depth: str = "advanced") -> str:
    """Get a quick answer to a question using Tavily's Q&A mode.

    Args:
        query: Question to answer (required)
        search_depth: Depth of search - "basic" or "advanced". Default is "advanced" for better answers
    """
    # Get API key from environment
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "")

    if not tavily_api_key:
        return (
            "Error: Tavily API key is not configured. Please set the "
            "TAVILY_API_KEY environment variable.\n"
            "Get your API key from: https://tavily.com"
        )

    # API endpoint
    endpoint = "https://api.tavily.com/search"

    try:
        payload = {
            "api_key": tavily_api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": True,
            "max_results": 3
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                endpoint, json=payload, timeout=30.0
            )

            response.raise_for_status()
            data = response.json()

            # Format response
            result_parts = []

            # Add answer if available
            if "answer" in data and data["answer"]:
                result_parts.append(f"Answer: {data['answer']}\n")

            # Add source URLs
            if "results" in data and data["results"]:
                sources = [f"- {r.get('url', 'N/A')}" for r in data["results"]]
                result_parts.append("Sources:\n" + "\n".join(sources))
            
            if not result_parts:
                return "No answer found."

            return "\n".join(result_parts)

    except httpx.HTTPStatusError as e:
        return f"Tavily API HTTP error occurred: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        return f"Error communicating with Tavily API: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def main():
    """Initialize and run the MCP server."""

    # Check for required environment variables
    if "TAVILY_API_KEY" not in os.environ:
        print(
            "Error: TAVILY_API_KEY environment variable is required",
            file=sys.stderr,
        )
        print(
            "Get a Tavily API key from: https://tavily.com",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Starting Tavily Search MCP server...", file=sys.stderr)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
