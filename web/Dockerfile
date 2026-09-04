FROM python:3.12-slim

WORKDIR /app

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
    "aiohttp>=3.9.0"

COPY agent_project /app/agent_project
COPY web /app/web
COPY config.yaml /app/config.yaml
COPY config.example.yaml /app/config.example.yaml

RUN mkdir -p /app/data/web_workspaces /app/logs

ENV PORT=8080
EXPOSE 8080

CMD ["python", "web/server.py"]