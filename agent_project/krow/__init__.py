"""krow 平台接入：原生点名注入 + 收件箱守候循环 + 空闲轮询 + 防重复消费（外部工作台侧）.

LV-Agent 作为 krow 的「外部工作台」接入时，平台不会主动叫醒它（发帖即判「外部工作台，不会被叫醒；TA 下次自己
上线能看到这条」），kit 的三家 CLI 钩子也套不上自研循环。本包补上这一截：

* :mod:`.items`    注入项与送给模型的文案契约（抬头 + 口令）；
* :mod:`.state`    持久化去重账（已送达 / 待送达 / 游标 / 大厅点名闭环账 / 主人回信账）；
* :mod:`.tools`    MCP 工具名解析（agent-hub ``krow_*`` 优先，平台 kit 桥 ``hall_*`` / ``team_*`` 兜底）；
* :mod:`.sources`  取数源：REST ``GET /agent/v1/inject``（读即消费）与 MCP ``hall_read`` 轮询；
* :mod:`.mailloop` 守候循环（v3）：``krow_inbox`` 长轮询 → 分诊 → 大厅点名「真读 → 取原文 → 作答 → 带 reply_to_id 回帖」闭环 /
  主人直聊最高优先级回信 / 其余站内信照旧；取消、退避、去重、崩溃续办；
* :mod:`.bridge`   统一入口 :class:`KrowBridge`：执行中注入、空闲轮询、互斥、装配守候循环；
* :mod:`.wake`     唤醒任务文本与常驻守护 :class:`KrowDaemon`（驱动守候循环）；
* :mod:`.selfcheck` 离线自检（假服务端 + 假 MCP 桥，不碰现网）。

全部只用标准库；``config.krow.enabled`` 为假时本包不会被实例化，对既有行为零影响。
"""
from __future__ import annotations

from .bridge import KrowBridge, KrowConfig, build_bridge, find_krow_server, list_server_tools
from .items import InjectItem, contract_text, render_items
from .mailloop import KrowMailLoop, build_mention_task, build_owner_task
from .state import DedupStore
from .tools import KrowToolError, KrowTools
from .wake import KrowDaemon, build_wake_task

__all__ = [
    "DedupStore",
    "InjectItem",
    "KrowBridge",
    "KrowConfig",
    "KrowDaemon",
    "KrowMailLoop",
    "KrowToolError",
    "KrowTools",
    "build_bridge",
    "build_mention_task",
    "build_owner_task",
    "build_wake_task",
    "contract_text",
    "find_krow_server",
    "list_server_tools",
    "render_items",
]
