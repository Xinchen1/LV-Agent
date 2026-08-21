# Contributing to LV Agent

Thanks for your interest in contributing to LV Agent! This project is
open-source under the MIT license, and we welcome all forms of contribution:
bug reports, feature ideas, documentation, and code.

## Ground Rules

1. **By submitting a pull request, you agree** that your contribution is
   licensed under the MIT license (same as the rest of the project).
2. **You certify that you have the right** to submit the code (you wrote it,
   or you have permission from the original author).
3. **Do not include** real API keys, tokens, passwords, or personal data
   in code, configs, or commit messages.
4. **Respect the trademarks**: do not use "LV Agent", "Lux Vita", or the
   cleveris research name/logo on derivative works without permission.

## How to Contribute

### Reporting Issues
- Search existing issues first to avoid duplicates
- Use a clear, descriptive title
- Include: steps to reproduce, expected vs actual behavior, and your
  environment (OS, Python version, backend in use)

### Submitting Code
1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make your changes, keeping them focused
4. Add tests if you change behavior (`python -m pytest tests/ -v`)
5. Commit with a clear message (see style below)
6. Push and open a Pull Request

### Commit Message Style
We use conventional commits:

```
type: short description

Optional detailed explanation (why the change, what it affects)
```

Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `chore`, `security`

Example:
```
fix: resolve Enter key non-response in interactive prompt

The raw input reader conflicted with readline's bracketed paste mode.
Disabled readline so termios+cbreak handles input directly.
```

## Development Setup

```bash
git clone https://github.com/Xinchen1/LV-Agent.git
cd LV-Agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m pytest tests/ -v
```

## Code of Conduct

Be respectful and constructive. Harassment, trolling, and personal attacks
are not tolerated. We're building something together — help make it
welcoming for everyone.
