"""Assistant data dir helper.

个人助理工具(calendar/tasks/notes/...)的 SQLite 存放目录:
- 默认: agent_project/data/
- 测试/多实例: 设环境变量 LV_AGENT_ASSISTANT_DATA_DIR 重定向
"""

import os
from pathlib import Path


def assistant_data_dir() -> Path:
    override = os.environ.get("LV_AGENT_ASSISTANT_DATA_DIR")
    if override:
        p = Path(override)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return Path(__file__).resolve().parent.parent / "data"


def db_path(filename: str) -> Path:
    return assistant_data_dir() / filename


_ISO_DATETIME_RE = None
def is_isoish(text: str) -> bool:
    """ISO / 'YYYY-MM-DD HH:MM' / 'YYYY-MM-DD' 判定(True→直接入库)。"""
    global _ISO_DATETIME_RE
    import re
    if _ISO_DATETIME_RE is None:
        _ISO_DATETIME_RE = re.compile(
            r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2})?)?$"
        )
    return bool(_ISO_DATETIME_RE.match((text or "").strip()))


def resolve_when(text: str, *, all_day_if_no_time: bool = False):
    """把 '明天下午三点' / '18:00' / '2026-10-12 10:00' 统一解析成 datetime。

    返回 (start, end, confidence); 解析失败返回 (None, None, 0)。
    """
    if not text:
        return None, None, 0
    text = text.strip()
    if is_isoish(text):
        from datetime import datetime as _dt
        raw = text.replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
            try:
                s = _dt.strptime(raw, fmt)
                if fmt == "%Y-%m-%d" and not all_day_if_no_time:
                    return None, None, 0  # 纯日期给 alarms 用是含糊的, 交给解析器兜底
                return s, s, 1.0
            except ValueError:
                continue
        return None, None, 0
    try:
        from .datetime_parser import parse_datetime
        p = parse_datetime(text)
        return p.start, (p.end or p.start), p.confidence
    except Exception:
        return None, None, 0
