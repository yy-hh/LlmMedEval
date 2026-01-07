import os
import re
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP

SGLANG_BASE_URL = os.getenv("SGLANG_BASE_URL", "http://127.0.0.1:1217")
SGLANG_API_KEY = os.getenv("SGLANG_API_KEY", "")   # 如果你的 SGLang 网关需要
DEFAULT_MODEL = os.getenv("SGLANG_MODEL", "auto")

mcp = FastMCP("SGLang-MCP-Gateway", json_response=True)

# 并发限制：避免 MCP 多客户端打爆 SGLang
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "4"))
sem = asyncio.Semaphore(MAX_CONCURRENCY)

_think_block = re.compile(r"(?s)<think>.*?</think>\s*")

def strip_think(text: str) -> str:
    # 兼容两种：<think>...</think> 或只出现 </think>
    if "</think>" in text:
        _, _, after = text.partition("</think>")
        return after.lstrip()
    return _think_block.sub("", text).lstrip()

def headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if SGLANG_API_KEY:
        h["Authorization"] = f"Bearer {SGLANG_API_KEY}"
    return h

@mcp.tool()
async def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    stream: bool = False,
    strip_think_output: bool = True,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """Call SGLang /v1/chat/completions and return normalized result."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }

    url = f"{SGLANG_BASE_URL}/v1/chat/completions"

    async with sem:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(url, headers=headers(), json=payload)
            r.raise_for_status()
            data = r.json()

    raw = data["choices"][0]["message"]["content"]
    content = strip_think(raw) if strip_think_output else raw

    return {
        "content": content,
        "raw_content": raw if not strip_think_output else None,
        "model": data.get("model", model),
        "usage": data.get("usage", {}),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }

@mcp.resource("model://info")
async def model_info() -> str:
    # 兼容不同实现：/model_info 或 /v1/models
    async with httpx.AsyncClient(timeout=10) as client:
        for path in ("/model_info", "/v1/models"):
            try:
                r = await client.get(f"{SGLANG_BASE_URL}{path}", headers=headers())
                if r.status_code == 200:
                    return r.text
            except Exception:
                pass
    return "model info endpoint not available"

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")  # stdio 或 streamable-http
    # streamable-http 时可加 host/port（具体参数跟你安装的 fastmcp 版本有关）
    mcp.run(transport=transport)
