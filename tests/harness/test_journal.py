"""Journal durability, replay, torn-tail tolerance and forks."""

import json

from agent_project.harness import events as ev
from agent_project.harness.journal import ForkedJournal, Journal


def test_append_assigns_monotonic_seq_and_replays(tmp_path):
    j = Journal(tmp_path / "s.jsonl")
    stamped = [j.append(ev.TurnStarted(turn_index=i)) for i in range(3)]
    assert [e.seq for e in stamped] == [1, 2, 3]

    replayed = Journal(tmp_path / "s.jsonl").read_all()
    assert [type(e) for e in replayed] == [ev.TurnStarted] * 3
    assert [e.turn_index for e in replayed] == [0, 1, 2]


def test_reopened_journal_continues_seq(tmp_path):
    path = tmp_path / "s.jsonl"
    Journal(path).append(ev.SessionStarted(task="x"))
    j2 = Journal(path)
    assert j2.last_seq == 1
    assert j2.append(ev.FinalAnswer(text="y")).seq == 2


def test_torn_tail_is_ignored(tmp_path):
    path = tmp_path / "s.jsonl"
    j = Journal(path)
    j.append(ev.SessionStarted(task="ok"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "TurnStarted", "data": {"turn_index": 9')  # truncated
    events = Journal(path).read_all()
    assert len(events) == 1 and isinstance(events[0], ev.SessionStarted)


def test_unknown_event_kind_stops_replay_without_crashing(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        json.dumps({"kind": "FutureEventV99", "data": {}}) + "\n"
        + json.dumps({"kind": "SessionStarted", "data": {"task": "t", "seq": 2}}) + "\n",
        encoding="utf-8",
    )
    assert Journal(path).read_all() == []


def test_read_since_and_fork(tmp_path):
    parent = Journal(tmp_path / "p.jsonl")
    parent.append(ev.SessionStarted(task="root"))
    parent.append(ev.TurnStarted(turn_index=0))
    parent.append(ev.TurnStarted(turn_index=1))

    assert [e.seq for e in parent.read_since(1)] == [2, 3]

    child = ForkedJournal.from_parent(parent, at_seq=2, path=tmp_path / "c.jsonl")
    child_events = child.read_all()
    assert [e.seq for e in child_events] == [1, 2]
    assert isinstance(child_events[0], ev.SessionStarted)
    child.append(ev.FinalAnswer(text="branch"))
    assert child.last_seq == 3
    assert len(parent.read_all()) == 3  # parent untouched
