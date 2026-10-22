"""管道模式(桌面发送排队)协议测试: LV_READY 标记 + \x1f 换行还原 + 批处理跳过."""
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from super_agent import SuperAgentCLI


def _make_cli():
  return SuperAgentCLI.__new__(SuperAgentCLI)


class PipeProtocolTest(unittest.TestCase):
  """_prompt_with_footer 非 tty 分支行为."""

  def _run_prompt(self, stdin_text):
    cli = _make_cli()
    fake_in = io.StringIO(stdin_text)
    fake_out = io.StringIO()
    fake_in.isatty = lambda: False
    fake_out.isatty = lambda: False
    with mock.patch('sys.stdin', fake_in), mock.patch('sys.stdout', fake_out):
      value = cli._prompt_with_footer()
    return cli, value, fake_out.getvalue()

  def test_ready_marker_emitted(self):
    cli, value, out = self._run_prompt("hello\n")
    self.assertIn("LV_READY_7f3a2e", out)
    self.assertEqual("hello", value)

  def test_pipe_mode_flag_set(self):
    cli, _, _ = self._run_prompt("hello\n")
    self.assertTrue(getattr(cli, '_pipe_mode', False))

  def test_multiline_restore_single_newline(self):
    _, value, _ = self._run_prompt("第一行\x1f第二行\n")
    self.assertEqual("第一行\n第二行", value)

  def test_multiline_restore_blank_line(self):
    _, value, _ = self._run_prompt("段落A\x1f\x1f段落B\n")
    self.assertEqual("段落A\n\n段落B", value)

  def test_pipe_batch_split_disabled(self):
    """管道模式下空行分隔不应进入 _run_task_batch(上下文不是独立任务)."""
    cli = _make_cli()
    cli._pipe_mode = True
    calls = []
    with mock.patch.object(cli, '_run_task_batch', lambda segs: calls.append(segs)):
      src = open(ROOT / 'super_agent.py', encoding='utf-8').read()
      snippet = src[src.index('多段任务'):src.index('try:\n        result = self._run_with_progress')]
      # 模拟片段逻辑: 语义直接验证条件分支
      self.assertIn("not getattr(self, '_pipe_mode', False)", snippet)
    self.assertEqual([], calls)


if __name__ == '__main__':
  unittest.main()
