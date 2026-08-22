#!/usr/bin/env python3
"""
Contextual Search for LV Agent
Pipeline: Local files -> Memory -> Web Search
"""
import os
from pathlib import Path
from typing import List, Dict

VAULT_PATH = Path("/Users/mac/Downloads/Obsidian Vault")
AGENT_PATH = Path("/Users/mac/Desktop/agent_project")

def local_search(query: str, limit: int = 5) -> List[Dict]:
    results = []
    keywords = query.split()
    for root, dirs, files in os.walk(VAULT_PATH):
        for f in files:
            if f.endswith(('.md', '.txt')):
                p = Path(root) / f
                try:
                    txt = p.read_text(encoding='utf-8', errors='ignore')
                except:
                    continue
                if any(k.lower() in txt.lower() for k in keywords):
                    results.append({'path': str(p), 'snippet': txt[:300]})
    return results[:limit]

def build_contextual_query(query: str, local_hits: List[Dict]) -> str:
    context = []
    for hit in local_hits:
        context.append(os.path.basename(hit['path']))
    if context:
        return f"{query} context:{' '.join(context)}"
    return query

def contextual_search(query: str):
    print(f"[本地检索] 查询: {query}")
    local = local_search(query)
    print(f"命中本地文件: {len(local)}")
    for r in local:
        print("-", r['path'])
    refined = build_contextual_query(query, local)
    print(f"[Web 检索] 优化后查询: {refined}")
    # 这里可接入 web_search
    return {'local': local, 'refined_query': refined}

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "LV Agent 核心产品"
    contextual_search(q)
