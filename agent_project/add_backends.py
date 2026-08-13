"""Add openrouter and anthropic backend support to agent.py"""
with open("agent_project/agent.py", "r") as f:
    lines = f.readlines()

# Find "return backend" after openmythos section (around line 265)
insert_idx = None
for i in range(250, 280):
    if i < len(lines) and "return backend" in lines[i].strip():
        insert_idx = i + 1  # insert AFTER this line
        break

if insert_idx is None:
    raise SystemExit("Could not find insertion point")

print(f"Inserting at index {insert_idx} (after line {insert_idx})")

new_backend_code = """
        elif self.config.backend == "openrouter":
            # OpenRouter (OpenAI-compatible gateway to many models)
            or_cfg = self.config.openrouter
            api_key = or_cfg.get('api_key') or os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                raise ValueError(
                    "OpenRouter backend selected but no API key provided. "
                    "Set agent.openrouter.api_key in config.yaml or OPENROUTER_API_KEY env var."
                )
            print("\\033[2musing OpenRouter backend\\033[0m")
            backend = OpenAIBackend(
                api_key=api_key,
                base_url=or_cfg.get('base_url', 'https://openrouter.ai/api/v1'),
                model=or_cfg.get('model', 'anthropic/claude-sonnet-4'),
                temperature=or_cfg.get('temperature', self.config.temperature),
                top_p=or_cfg.get('top_p', 0.9),
                max_tokens=or_cfg.get('max_tokens', 4096),
                timeout=or_cfg.get('timeout', 120),
            )
            backend.tokenizer = self._create_simple_tokenizer()
            return backend

        elif self.config.backend == "anthropic":
            # Anthropic Messages API (direct)
            ant_cfg = self.config.anthropic
            api_key = ant_cfg.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "Anthropic backend selected but no API key provided. "
                    "Set agent.anthropic.api_key in config.yaml or ANTHROPIC_API_KEY env var."
                )
            print("\\033[2musing Anthropic Messages API backend\\033[0m")
            try:
                from .model_backends import AnthropicBackend
            except ImportError:
                raise ImportError(
                    "anthropic backend requires 'anthropic' package. "
                    "Install with: pip install anthropic"
                )
            backend = AnthropicBackend(
                api_key=api_key,
                base_url=ant_cfg.get('base_url', 'https://api.anthropic.com/v1'),
                model=ant_cfg.get('model', 'claude-sonnet-4-20250514'),
                temperature=ant_cfg.get('temperature', self.config.temperature),
                max_tokens=ant_cfg.get('max_tokens', 4096),
                timeout=ant_cfg.get('timeout', 120),
            )
            return backend
"""

lines.insert(insert_idx, new_backend_code)

with open("agent_project/agent.py", "w") as f:
    f.writelines(lines)

print(f"Inserted new backend handlers. File now has {len(lines)} lines.")
