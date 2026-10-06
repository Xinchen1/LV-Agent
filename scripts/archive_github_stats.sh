#!/bin/bash
# 每日归档 GitHub 流量数据（clones/views/stars/forks），防止 14 天过期丢失
set -u
REPO="Xinchen1/LV-Agent"
OUT="/Users/mac/Desktop/agent_project/data/github_traffic.jsonl"
mkdir -p "$(dirname "$OUT")"
export CLONES=$(gh api "repos/$REPO/traffic/clones" 2>/dev/null)
export VIEWS=$(gh api "repos/$REPO/traffic/views" 2>/dev/null)
export STARS=$(gh api "repos/$REPO" --jq '.stargazers_count' 2>/dev/null)
export FORKS=$(gh api "repos/$REPO" --jq '.forks_count' 2>/dev/null)
export REPO STAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python3 -c "
import json,os,datetime
def load(k):
    try: return json.loads(os.environ.get(k) or '{}')
    except Exception: return {}
rec={'archived_at':os.environ['STAMP'],'repo':os.environ['REPO'],
     'stars':int(os.environ.get('STARS') or 0),'forks':int(os.environ.get('FORKS') or 0),
     'clones_14d':load('CLONES'),'views_14d':load('VIEWS')}
print(json.dumps(rec,ensure_ascii=False))
" >> "$OUT"
