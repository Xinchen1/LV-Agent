"""敏感输入掩码读取回归: 粘贴时逐字符显示掩码, 而不是完全不回显."""

import pytest

import super_agent
from super_agent import _masked_read_loop


def _run(script):
    it = iter(script)
    out = []
    value = _masked_read_loop(lambda: next(it, ""), out.append)
    return value, "".join(out)


def test_each_typed_char_echoes_one_mask():
    value, echo = _run(["s", "k", "-", "a", "b", "\r"])
    assert value == "sk-ab"
    assert echo == "•" * 5, "每个字符都应回显一个掩码符号"


def test_pasted_key_shows_masks_without_escape_noise():
    # 终端 bracketed paste 会插入 \x1b[200~ ... \x1b[201~ 包裹符
    value, echo = _run([
        "\x1b", "[", "2", "0", "0", "~",
        "n", "v", "a", "p", "i", "-",
        "\x1b", "[", "2", "0", "1", "~",
        "\n",
    ])
    assert value == "nvapi-"
    assert echo == "•" * 6


def test_backspace_erases_one_mask():
    value, echo = _run(["a", "b", "\x7f", "c", "\n"])
    assert value == "ac"
    assert echo == "••" + "\b \b" + "•"


def test_eof_and_interrupt_signals():
    with pytest.raises(EOFError):
        _run([""])                     # 空行 EOF -> 取消
    assert _run(["a", ""])[0] == "a"   # 有内容时 EOF 视为提交
    with pytest.raises(KeyboardInterrupt):
        _run(["a", "\x03"])


def test_ask_secret_uses_masked_reader(monkeypatch):
    seen = {}

    def fake(prompt):
        seen["prompt"] = prompt
        return "sk-ant-test"

    monkeypatch.setattr(super_agent, "read_secret_masked", fake)
    cli = super_agent.SuperAgentCLI.__new__(super_agent.SuperAgentCLI)
    assert cli._ask_secret("API Key (sk-ant-...)") == "sk-ant-test"
    assert seen["prompt"] == "API Key (sk-ant-...): "


def test_ask_secret_cancelled_returns_empty(monkeypatch):
    def boom(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr(super_agent, "read_secret_masked", boom)
    cli = super_agent.SuperAgentCLI.__new__(super_agent.SuperAgentCLI)
    assert cli._ask_secret("API Key") == ""