# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security issue in LV Agent,
**please do not open a public issue** — report it privately.

**How to report:**
- Open a GitHub security advisory at:
  `https://github.com/Xinchen1/LV-Agent/security/advisories/new`
- Or email the maintainers (linked in the GitHub profile)

**What to include:**
- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Impact assessment

**Response timeline:**
- We acknowledge reports within 48 hours
- We aim to provide a fix or mitigation within 7 days

## Security Notes

- API keys and tokens are read from environment variables or `config.yaml`.
  Never commit real keys to the repository.
- The `bash_exec` and `python_exec` tools run arbitrary commands. Keep them
  disabled unless you trust the environment (see `config.yaml`).
- The Telegram bot token in `config.example.yaml` is a placeholder. Replace
  it with your own before enabling the bot.
