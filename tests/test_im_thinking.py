"""IM 渠道思考过程流: AgentWorker.ask_stream + 飞书 _ThinkingProgress 聚合刷新."""
import queue
import threading
import time
import unittest

from agent_project.im_bridge.base import AgentWorker


class FakeAgent:
  def __init__(self, events):
    self._events = events
    self.seen_kind = []

  def run(self, task, code_mode=False, stream_callback=None, **kw):
    self.stream_cb = stream_callback
    for kind, text in self._events:
      self.seen_kind.append((kind, text))
      if stream_callback:
        stream_callback(kind, text)
    return {"final_answer": "最终答案", "turns": []}


class AskStreamTest(unittest.TestCase):
  def test_ask_stream_forwards_events(self):
    events = [
      ('status', 'thinking'),
      ('status', 'chain-of-thought: step 2/4'),
      ('tool_call', 'file_ops(read_file)'),
      ('reasoning', '先分析目录结构'),
      ('content', ' [TOOL: file_ops] junk'),
      ('content', '最终内容片段'),
    ]
    worker = AgentWorker(agent=None)
    got = []
    orig_ensure = worker._ensure_agent
    worker._ensure_agent = lambda: worker.__dict__.setdefault('_agent', FakeAgent(events))
    worker._agent = FakeAgent(events)
    answer = worker.ask_stream('任务', on_event=lambda k, t: got.append((k, t)))
    self.assertEqual('最终答案', answer)
    kinds = {kind for kind, _ in got}
    self.assertIn('status', kinds)
    self.assertIn('tool_call', kinds)
    self.assertIn('reasoning', kinds)
    # content 流不透传给 IM(答案完成时整条回复), TOOL: 行不开头丢弃
    self.assertNotIn('content', kinds)
    for kind, text in got:
      self.assertFalse(text.startswith('[TOOL:'))

  def test_ask_without_stream_no_callback(self):
    worker = AgentWorker(agent=FakeAgent([('status', 'x')]))
    self.assertEqual('最终答案', worker.ask('任务'))

  def test_event_callback_error_does_not_break_task(self):
    worker = AgentWorker(agent=FakeAgent([('status', 'boom'), ('reasoning', 'r')]))
    def _bad(kind, text):
      raise RuntimeError('platform send fail')
    self.assertEqual('最终答案', worker.ask_stream('任务', on_event=_bad))


class FakeBot:
  def __init__(self):
    self.sent = 0
    self.updated = 0
    self._mid = 'om_fake_msg'

  def _send_message(self, chat_id, text):
    self.sent += 1
    return self._mid

  def _update_message(self, message_id, text):
    self.updated += 1


class ThinkingProgressTest(unittest.TestCase):
  def _cls(self):
    import agent_project.im_bridge.feishu as feishu
    return feishu.FeishuBridge._ThinkingProgress

  def _bot_and_progress(self):
    cls = self._cls()
    bot = FakeBot()
    prog = cls(bot, 'chat1')
    prog._last_flush = 0.0
    return bot, prog

  def test_first_event_creates_message(self):
    bot, prog = self._bot_and_progress()
    prog.on_event('status', 'chain-of-thought: step 1/4')
    self.assertEqual(1, bot.sent)
    self.assertEqual('om_fake_msg', prog._message_id)

  def test_throttle_updates(self):
    bot, prog = self._bot_and_progress()
    prog.on_event('status', 's1')
    prog.on_event('tool_call', 'tool A')
    prog.on_event('status', 'step 2')
    # 2s 节流: 首条发了消息, 后续若在节流窗口内不应重复编辑
    self.assertEqual(1, bot.sent)
    self.assertLessEqual(bot.updated, 1)

  def test_finish_marks_done(self):
    bot, prog = self._bot_and_progress()
    prog.on_event('tool_call', 'tool A')
    finish_body_holder = {}
    orig_update = bot._update_message
    def upd(mid, text):
      finish_body_holder['b'] = text
      orig_update(mid, text)
    bot._update_message = upd
    prog.finish()
    self.assertTrue(prog._done)
    self.assertIn('思考过程 ✅', finish_body_holder['b'])

  def test_render_caps_size(self):
    bot, prog = self._bot_and_progress()
    for i in range(80):
      prog.on_event('status', f'step {i}')
    body = prog._render()
    self.assertLessEqual(len(body.encode('utf-8')), 3800)

  def test_thinking_status_suppressed(self):
    bot, prog = self._bot_and_progress()
    prog.on_event('status', 'thinking')
    self.assertEqual(0, bot.sent)  # 'thinking' 泛提示不触发首条消息


if __name__ == '__main__':
  unittest.main()
