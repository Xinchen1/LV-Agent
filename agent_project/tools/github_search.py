"""GitHub Search Tool - Search GitHub code, issues, pull requests, repos."""

import json
from typing import Any, Dict, List, Optional

import requests
from . import BaseTool, ToolResult, TOOLS_REGISTRY


class GitHubSearchTool(BaseTool):
    name = "github_search"
    description = (
        "Search GitHub using the public search API. "
        "Use this when you need to find real code, repos, issues, or PRs."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "GitHub search query, like 'language:python agent tool' "
                    "or 'repo:openai/openai-python'. Combine qualifiers freely."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["code", "repositories", "issues", "pull_requests"],
                "default": "code",
                "description": "What to search: code, repositories, issues, or pull requests.",
            },
            "per_page": {
                "type": "integer",
                "description": "Results per page (default 10, max 30).",
                "default": 10,
                "minimum": 1,
                "maximum": 30,
            }
        },
        "required": ["query"],
    }

    def execute(
        self,
        query: str,
        kind: str = "code",
        per_page: int = 10,
    ) -> ToolResult:
        try:
            results = self._search(kind=kind, query=query, per_page=per_page)
            output = json.dumps(results, indent=2, ensure_ascii=False)
            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "kind": kind,
                    "query": query,
                    "num_results": len(results),
                },
            )
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            body = ""
            if e.response is not None:
                try:
                    body = e.response.text[:300]
                except Exception:
                    body = ""
            return ToolResult(
                success=False,
                output="",
                error=f"GitHub search failed: {status} {body}".strip(),
            )
        except requests.RequestException as e:
            return ToolResult(success=False, output="", error=f"GitHub search failed: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"GitHub search error: {str(e)}")

    def _endpoint(self, kind: str) -> str:
        return {
            "code": "https://api.github.com/search/code",
            "repositories": "https://api.github.com/search/repositories",
            "issues": "https://api.github.com/search/issues",
            "pull_requests": "https://api.github.com/search/issues",
        }[kind]

    def _search(self, kind: str, query: str, per_page: int) -> List[Dict[str, Any]]:
        params = {"q": query, "per_page": int(per_page)}
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "OpenMythosAgent",
        }
        url = self._endpoint(kind)
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", []) or []

        results: List[Dict[str, Any]] = []
        for item in items:
            if kind == "code":
                results.append(
                    {
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "repository": item.get("repository", {}).get("full_name"),
                        "html_url": item.get("html_url"),
                        "score": item.get("score"),
                    }
                )
            elif kind == "repositories":
                results.append(
                    {
                        "full_name": item.get("full_name"),
                        "description": item.get("description"),
                        "stars": item.get("stargazers_count"),
                        "url": item.get("html_url"),
                        "language": item.get("language"),
                        "score": item.get("score"),
                    }
                )
            else:
                results.append(
                    {
                        "title": item.get("title"),
                        "html_url": item.get("html_url"),
                        "state": item.get("state"),
                        "repository_url": item.get("repository_url"),
                        "score": item.get("score"),
                    }
                )
        return results


if not TOOLS_REGISTRY.get("github_search"):
    TOOLS_REGISTRY.register(GitHubSearchTool())
