"""Append-only event journal with crash-safe replay.

Storage format is JSON Lines: one event per line, ``kind`` + payload.
Appends are ``os.fsync``-ed so a power loss can only lose the last line,
and readers tolerate a truncated tail (the classic WAL partial-write case).

The journal assigns ``seq`` on append -- events arrive with ``seq=-1`` and
are returned stamped. Reads replay in seq order; a session rebuilds its
logical state with :func:`agent_project.harness.events.fold`.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterator, List, Optional

from .events import EVENT_REGISTRY, Event


class Journal:
    """Single-session append-only log.

    Thread-safe for appends. One Journal instance per open file; cheap to
    construct, so callers usually keep one per Session.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = self._recover_seq()

    # ---------- writes ----------

    def append(self, event: Event) -> Event:
        """Stamp *event* with the next seq and persist it durably."""

        with self._lock:
            self._seq += 1
            stamped = event.with_seq(self._seq)
            line = json.dumps(
                {"kind": stamped.kind, "data": stamped.model_dump(mode="json")},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            return stamped

    def append_many(self, events: List[Event]) -> List[Event]:
        return [self.append(e) for e in events]

    # ---------- reads ----------

    def read_all(self) -> List[Event]:
        return list(self.iter_events())

    def iter_events(self) -> Iterator[Event]:
        """Yield events in seq order, skipping a torn tail line."""

        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cls = EVENT_REGISTRY[obj["kind"]]
                    yield cls.model_validate(obj["data"])
                except (json.JSONDecodeError, KeyError, ValueError):
                    # Torn write or unknown event kind: stop replay here.
                    # Unknown kinds from *newer* code must not crash old readers.
                    break

    def read_since(self, seq: int) -> List[Event]:
        """Events strictly after *seq* -- used by forks and tail-followers."""

        return [e for e in self.iter_events() if e.seq > seq]

    @property
    def last_seq(self) -> int:
        return self._seq

    # ---------- internals ----------

    def _recover_seq(self) -> int:
        last = 0
        for ev in self.iter_events():
            last = max(last, ev.seq)
        return last


class ForkedJournal(Journal):
    """A journal that starts as a copy of a parent journal up to a seq."""

    @classmethod
    def from_parent(
        cls, parent: Journal, at_seq: int, path: Path | str
    ) -> "ForkedJournal":
        child = cls(path)
        for ev in parent.iter_events():
            if ev.seq > at_seq:
                break
            child.append(ev.model_copy(update={"seq": -1}))
        return child
