"""
API Call Tool - 安全HTTP请求（仅允许预定义hosts）
"""

import json
import requests
from typing import Dict, Any, Optional
from . import BaseTool, ToolResult, TOOLS_REGISTRY


class ApiCallTool(BaseTool):
    name = "api_call"
    description = "Make HTTP requests to allowed API endpoints. Useful for interacting with REST APIs."

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to make request to (must match allowed hosts)"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "default": "GET",
                "description": "HTTP method"
            },
            "headers": {
                "type": "object",
                "description": "Request headers (JSON object)",
                "default": {}
            },
            "data": {
                "type": "object",
                "description": "Request body for POST/PUT (JSON object, will be serialized)"
            }
        },
        "required": ["url", "method"]
    }

    def __init__(self, allowed_hosts: list = None, timeout: int = 30):
        self.allowed_hosts = set(allowed_hosts) if allowed_hosts else None
        self.timeout = timeout

    def execute(self, url: str, method: str = "GET", headers: Dict = None, data: Dict = None) -> ToolResult:
        try:
            # 安全检查：验证URL是否在允许的hosts内
            if not self._is_allowed_url(url):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"URL host not in allowed list. Allowed: {', '.join(self.allowed_hosts)}"
                )

            # 准备请求
            request_headers = headers or {}
            request_kwargs = {
                "method": method.upper(),
                "url": url,
                "headers": request_headers,
                "timeout": self.timeout
            }

            if data is not None and method.upper() in ["POST", "PUT", "PATCH"]:
                request_kwargs["json"] = data

            # 发送请求
            response = requests.request(**request_kwargs)

            # 尝试解析JSON
            try:
                output = json.dumps(response.json(), indent=2)
            except ValueError:
                output = response.text

            return ToolResult(
                success=response.ok,
                output=output,
                metadata={
                    "status_code": response.status_code,
                    "method": method.upper(),
                    "url": url
                }
            )

        except requests.RequestException as e:
            return ToolResult(success=False, output="", error=f"HTTP request failed: {str(e)}")
        except Exception as e:
            return ToolResult(success=False, output="", error=f"Error: {str(e)}")

    def _is_allowed_url(self, url: str) -> bool:
        """检查URL是否允许访问。无白名单时允许所有外部URL，阻止内网SSRF。"""
        from urllib.parse import urlparse
        import ipaddress
        try:
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()
            if ':' in hostname:
                hostname = hostname.split(':')[0]
            if self.allowed_hosts:
                return hostname in self.allowed_hosts
            if hostname in ('localhost', '127.0.0.1', '::1', '0.0.0.0'):
                return False
            try:
                ip = ipaddress.ip_address(hostname)
                return not (ip.is_private or ip.is_loopback or ip.is_link_local)
            except ValueError:
                return True
        except Exception:
            return False


# 注册工具
TOOLS_REGISTRY.register(ApiCallTool())
