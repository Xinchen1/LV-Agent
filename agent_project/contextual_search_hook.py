"""
Hook to make LV Agent search context-aware
Integrate into super_agent.py command handling
"""
from pathlib import Path
import os
from typing import List, Dict

VAULT_PATH = Path("/Users/mac/Downloads/Obsidian Vault")
AGENT_PATH = Path("/Users/mac/Desktop/agent_project")

def local_hits(query: str, limit: int = 5) -> List[Dict]:
    keywords = query.lower().split()
    hits = []
    for root, dirs, files in os.walk(VAULT_PATH):
        for f in files:
            if f.lower().endswith(('.md','.txt')):
                p = Path(root)/f
                try:
                    txt = p.read_text(encoding='utf-8', errors='ignore')
                except:
                    continue
                if any(k in txt.lower() for k in keywords):
                    hits.append({'path': str(p), 'snippet': txt[:200]})
    return hits[:limit]

def contextual_search(query: str):
    hits = local_hits(query)
    context_files = [os.path.basename(h['path']) for h in hits]
    enhanced = f"{query} | local_context:{','.join(context_files)}"
    return {
        'local_hits': hits,
        'enhanced_query': enhanced
    }

# Example usage in super_agent.py:
# from agent_project.contextual_search_hook import contextual_search
# result = contextual_search(user_query)
# print(result['local_hits'])
# # then call web_search with result['enhanced_query']
