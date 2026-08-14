"""GitHub Search Tool - Search GitHub repos, code, issues, PRs, users.

Thoroughly wired GitHub search with:
- repo search with stars/created sorting and date filters
- code / issues / PR / user search
- optional GITHUB_TOKEN for higher rate limits
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests
from . import BaseTool, ToolResult, TOOLS_REGISTRY


class GitHubSearchTool(BaseTool):
    name = "github_search"
    description = (
        "Search GitHub using the public REST API. Supports repos, code, issues, "
        "pull requests, and users. Examples: "
        "repos created:>2026-07-01 stars:>100 sort=stars ; "
        "code 'language:python agent'. Set GITHUB_TOKEN for higher rate limits."
    )

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "GitHub search query. Repos: 'created:>2026-07-01 stars:>100 agent'. "
                    "Code: 'language:python agent tool'. Combine qualifiers freely."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["repositories", "code", "issues", "pull_requests", "users"],
                "default": "repositories",
                "description": "What to search.",
            },
            "sort": {
                "type": "string",
                "enum": ["stars", "best-match", "updated", "forks", "help-wanted-issues"],
                "default": "best-match",
                "description": "Sort field (repositories: stars/updated/forks; best-match otherwise).",
            },
            "order": {
                "type": "string",
                "enum": ["desc", "asc"],
                "default": "desc",
                "description": "Sort order.",
            },
            "per_page": {
                "type": "integer",
                "description": "Results per page (default 10, max 30).",
                "default": 10,
                "minimum": 1,
                "maximum": 30,
            },
            "page": {
                "type": "integer",
                "description": "Page number.",
                "default": 1,
                "minimum": 1,
            },
        },
        "required": ["query"],
    }

    def execute(
        self,
        query: str,
        kind: str = "repositories",
        sort: str = "best-match",
        order: str = "desc",
        per_page: int = 10,
        page: int = 1,
    ) -> ToolResult:
        try:
            results = self._search(
                kind=kind,
                query=query,
                sort=sort,
                order=order,
                per_page=per_page,
                page=page,
            )
            output = json.dumps(results, indent=2, ensure_ascii=False)
            return ToolResult(
                success=True,
                output=output,
                metadata={
                    "kind": kind,
                    "query": query,
                    "sort": sort,
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
            hint = ""
            if status == 403:
                hint = " (rate limited: set GITHUB_TOKEN or wait)"
            elif status == 401:
                hint = " (invalid GITHUB_TOKEN)"
            return ToolResult(
                success=False,
                output="",
                error=f"GitHub search failed: {status} {body[:200]}{hint}".strip(),
            )
        except requests.RequestException as e:
            return ToolResult(success=False, output="", error=f"GitHub search failed: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"GitHub search error: {str(e)}")

    def _endpoint(self, kind: str) -> str:
        return {
            "repositories": "https://api.github.com/search/repositories",
            "code": "https://api.github.com/search/code",
            "issues": "https://api.github.com/search/issues",
            "pull_requests": "https://api.github.com/search/issues",
            "users": "https://api.github.com/search/users",
        }[kind]

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "LvAgent",
        }
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"token {token}"
        return headers

    def _search(
        self,
        kind: str,
        query: str,
        sort: str,
        order: str,
        per_page: int,
        page: int,
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "q": query,
            "per_page": int(per_page),
            "page": int(page),
        }
        # repositories support stars/updated/forks sort; others best-match
        if kind == "repositories" and sort in ("stars", "updated", "forks"):
            params["sort"] = sort
            params["order"] = order
        elif kind != "repositories" and sort not in ("best-match",):
            params["sort"] = sort
            params["order"] = order

        url = self._endpoint(kind)
        response = requests.get(url, params=params, headers=self._headers(), timeout=20)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", []) or []

        results: List[Dict[str, Any]] = []
        for item in items:
            if kind == "repositories":
                results.append(
                    {
                        "full_name": item.get("full_name"),
                        "description": item.get("description"),
                        "stars": item.get("stargazers_count"),
                        "forks": item.get("forks_count"),
                        "language": item.get("language"),
                        "created_at": item.get("created_at"),
                        "updated_at": item.get("updated_at"),
                        "topics": (item.get("topics") or [])[:5],
                        "url": item.get("html_url"),
                    }
                )
            elif kind == "code":
                results.append(
                    {
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "repository": item.get("repository", {}).get("full_name"),
                        "html_url": item.get("html_url"),
                        "score": item.get("score"),
                    }
                )
            elif kind == "users":
                results.append(
                    {
                        "login": item.get("login"),
                        "type": item.get("type"),
                        "html_url": item.get("html_url"),
                        "score": item.get("score"),
                    }
                )
            else:  # issues / pull_requests
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
