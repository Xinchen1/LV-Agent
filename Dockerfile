FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git jq && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "openai>=1.0.0" \
    "anthropic>=0.40.0" \
    "pyyaml>=6.0" \
    "pydantic>=2.5.0" \
    "requests>=2.31.0" \
    "python-dotenv>=1.0.0" \
    "rich>=13.7.0" \
    "tqdm>=4.66.0" \
    "tiktoken>=0.5.0" \
    "beautifulsoup4>=4.12.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.30.0" \
    "websockets>=12.0" \
    "aiohttp>=3.9.0" \
    "lxml>=5.0.0" \
    "playwright>=1.40.0"

RUN playwright install chromium --with-deps

COPY agent_project /app/agent_project
COPY web/server.py /app/web/server.py
COPY web/frontend /app/web/frontend
COPY config.yaml /app/config.yaml
COPY config.example.yaml /app/config.example.yaml

RUN mkdir -p /app/data/web_workspaces /app/logs

ENV PORT=8080
EXPOSE 8080

CMD ["python", "web/server.py"]