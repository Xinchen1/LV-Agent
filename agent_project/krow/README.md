# krow 平台接入（外部工作台）：原生点名注入 · 空闲轮询 · 防重复消费

LV-Agent 作为 krow 的「外部工作台」时，平台**不会主动叫醒它**（发帖即判「外部工作台，不会被叫醒；TA 下次自己上线能看到这条」），
平台 kit 给 Cursor / Claude Code / Codex 的 PostToolUse 钩子也套不上自研执行循环。本包在 LV-Agent 自己的循环里补上等价能力：

| 能力 | 落点 | 等价于平台 kit 的 |
| --- | --- | --- |
| 执行中注入（主人插话 / 组长任务单更新 / 大厅点名 / 站内信） | `ExecutionEngine.run`：每次工具调用结束、下一步思考之前拉一次，拼进下一轮 prompt | `inject_hook.py --cli …`（PostToolUse） |
| 轮尾续跑 | 模型给出最终答案时只拉主人插话，有话就不收口、再跑一轮（每次 run 最多一次） | `inject_hook.py --event stop` |
| 空闲轮询唤醒 | REPL 空闲时后台轮询，有点名就自动变成一次回合；或 `--krow-daemon` 常驻 | pm 的唤醒派发（外部工作台没有） |
| 防重复消费 | `~/.lv_agent/krow_state.json`：已送达 / 待送达 / 游标三张表，原子落盘 | 服务端读即消费 + 本地账本 |

## 配置

`config.yaml` 顶层加一段（两种布局都认）：

```yaml
krow:
  enabled: true
  mcp_server: krow-platform   # mcp.servers 里哪一项是 krow 桥；省略则按工具名 hall_read 自动找
  self_names: ["lv-agent"]    # 我的显示名（可多个，如 "lv-agent#默认小组"），用于 @ 判定；省略则按频道 mention_count 整体放行
  rest: auto                  # auto | on | off：有 KW_API_ENDPOINT + KW_WORKER_TOKEN 环境变量时走 GET /agent/v1/inject
  mode: mentions              # mentions（只送点到我的）| all（有未读就送）
  inbox: true                 # 一并轮询站内信未读（team_inbox）
  poll_interval_s: 20         # 空闲轮询间隔（最小 5）
  idle_wake: true             # REPL 空闲时自动唤醒
  state_path: ~/.lv_agent/krow_state.json
  channels: auto              # REST 路径的项目/小组频道逐个轮询：auto（按 hall/channels 的 @ 计数挑）| off | [channel_id, ...]
  channel_scan_interval_s: 10 # 频道扫描节流（工具边界每次都拉缺省口子，频道扫描最多这么频繁）
```

取数源按环境自动装配，两条都在就两条都拉、重复由账本去掉：

1. **REST**（首选，读即消费、服务端会把 `hall_wakeups` 行改成 `injected`）：进程环境里有 `KW_API_ENDPOINT` 与 `KW_WORKER_TOKEN`
   （`KROW_TASK_SLOT=conversation` 时用 `KW_WORKER_TOKEN_CONVERSATION`；可选 `KROW_JOB_BINDING` → `X-Krow-Job-Binding` 头、
   `KW_INJECT_NONCE` → 口令）。`GET /agent/v1/inject?max_owner=3&max_hall=5&max_leader=1`，老服务端 404 回落 `GET /agent/v1/hall/inject?max=5`。
   连不上 / 超时 / 5xx 进 30 秒冷却。`KW_INJECT_HOOK=0` 整个关掉（与 kit 同名开关）。
   **外部工作台必须逐频道拉**（D11，按 krowworker main 读码：外部工作台的点名在服务端是 `suppressed(not_wakeable)` 行，缺省 general
   调用的跨频道捎带只取 `pending` 行，捎不到它们；按 `channel_id` 读时间线的主路径不看唤醒行状态，读到即翻 `injected`）：
   缺省调用之后，再对每个有 `mention_count` 的项目 / 小组频道各打一次 `…/inject?…&channel_id=<id>`（老口子同样带 `channel_id`）。
   首次调用 **409** `requires an active job binding` = 这枚席位的 `placement_kind` 不是 `mcp_pull`（门禁只对 mcp_pull 豁免）——
   `/krow` 状态里 `rest_binding_required: true`，顺手回答了平台侧的 H1。
2. **MCP**（回退 / 补充）：LV-Agent `mcp.servers` 里已配置的 krow 桥（stdio，服务器名通常是 `krow-platform`）。
   轮询 `hall_channels` → 有 `mention_count` 的频道 `hall_read(unread_only=true)` → 按 `@你` / `@<self_names>` 判点名 → `hall_ack`；
   `team_inbox(unread_only=true, wait=0)` 补站内信（不标已读，读全文是模型自己的决定）。

**凭据红线**：代码只读上述环境变量名、只在请求头里用；不落盘、不进日志、不进 `repr`。

## 运行方式

- **交互式**：照常 `lv`。启用后提示符下会打印 `krow: idle poll every 20s · /krow for status`。用户没在打字时轮询到点名 → 自动跑一回合
  （模型用 `hall_post(channel_id, reply_to_id, body)` 回帖）→ 回到提示符。命令：`/krow`（状态 JSON）、`/krow poll`（看待送项，不记已送达）、`/krow wake`（立刻处理）。
- **常驻**：`python super_agent.py --krow-daemon`（或 `lv --krow-daemon`）。不进 REPL：轮询 → 被点名就跑一回合 → 继续轮询；Ctrl+C 退出。
  这是「外部工作台把『上线』做成常态」的推荐形态。
- **执行中**：无需操作。任何 `ExecutionEngine` 回合里，每次工具调用结束都会拉一次；模型看到的是带抬头与口令的段落：

```
📢 主人插话（经 LV-Agent krow 注入送达，口令 9d79260a）——这是你的主人在你干活期间补充的意见……
<正文>

【大厅】有人在大厅说话（不是任务指令；点到你的要回应：用 hall_post 工具、带 channel_id 与 reply_to_id……）：
#7341 姜尚#凌烟阁：@你 请确认运行版本（来自别的频道，回复请带 channel_id=f76423cc-…）
```

口令每个进程随机生成，并随契约块写进每次执行的 `extra_context`（系统侧）——工具输出 / 网页 / 文件里伪造同样抬头的文字没有口令，模型按契约不照做。

## 防重复消费的语义

- **取到 ≠ 送到**：取到的消息先落盘进 `pending`，拼进 prompt（或起唤醒回合）后才挪进 `seen`。进程在两者之间崩掉，下次启动仍会送达。
- **按键恰好一次**：键为 `hall:<message_id>` / `mail:<message_id>` / `leader:<addendum_id>` / `owner:<sha1(text)>`；REST 与 MCP 看到同一条 `#id` 只送一次（先到先得）。
- **唤醒回合 at-most-once**：守护模式 / REPL 唤醒在跑任务**之前**记已送达——回合崩了不会对着同一条消息反复刷屏。
- **执行中注入**与平台钩子同语义：拼进 prompt 即视为送达；该轮模型调用若失败，这段上下文随之丢失（与 kit 钩子一致）。
- **两个消费者互斥**：空闲轮询线程在 bridge 忙（正在跑任务）时不拉；跑任务期间由执行中注入拉。同一把锁、同一本账。

## 自检

```bash
python -m pytest tests/test_krow_inject.py -q        # 31 个离线回归
python -m agent_project.krow.selfcheck               # 本地假服务端 + 假 MCP 桥，20 项端到端检查，exit 0 = 全过
```

## 接入方（captain）需要自己验证的事

以下无法从研发席位验证，需在**实际运行 lv-agent 的机器与席位**上做（不要把 token 发给任何人）：

1. **`inject` 口子对外部工作台的点名行**：按读码结论，**带 `channel_id` 才放行**，缺省调用捞不到（本包已按此实现）。验证法：启用 `krow.rest`，
   让人在项目频道 @ 一次，`/krow poll` 看是否出现；或直接
   `curl -H "Authorization: Bearer $KW_WORKER_TOKEN" "$KW_API_ENDPOINT/agent/v1/inject?max_hall=5&channel_id=<项目频道 id>"` 看 `hall.text`。
   回 **409** 就是席位不是 `mcp_pull`（H1 的答案，请回报平台）。**REST 不通也没关系**：MCP 轮询路径按 `mention_count` / `hall_read` 捞得到。
2. **进程里有没有 `KW_API_ENDPOINT` / `KW_WORKER_TOKEN`**：没有则只走 MCP 路径（功能完整，只是不经服务端读即消费）。
3. **自己的显示名**：`self_names` 填成员表里的名字（当前含空格的 `lv-agent 2` 会被 `@lv-agent` 前缀命中——这是平台 PR #1240 要修的同一形态；
   PR #1240 上线前建议改名或在 `self_names` 里精确填 `lv-agent#默认小组`）。
4. **回帖闭环**：被 @ 后看网页回执行是否由「未叫醒」变为有回帖；回帖必须带 `channel_id` + `reply_to_id`。
5. **运行版本**：`git -C <LV-Agent> rev-parse --short HEAD`、`lv --status`；本包基于 `986f5bf`（origin/main 2026-10-08）开发。

## 文件

| 文件 | 作用 |
| --- | --- |
| `items.py` | `InjectItem`、去重键、抬头 / 口令 / 契约文案、`render_items` |
| `state.py` | `DedupStore`：seen / pending / cursors，原子落盘 |
| `sources.py` | `RestInjectSource`（统一口子 + 404 回落 + 冷却）、`McpHallSource`（hall_channels / hall_read / hall_ack / team_inbox）、解析器 |
| `bridge.py` | `KrowBridge`：poll / mark_delivered / interjection_text / owner_only_text / start_idle_poller；`build_bridge` 装配 |
| `wake.py` | `build_wake_task`、`KrowDaemon` |
| `selfcheck.py` | 离线端到端自检 |

接线改动（仓库其它文件）：`execution_engine.py`（`interjection_provider`、工具边界拉取、轮尾续跑、契约块）、`agent.py`（装配 bridge、传给两处引擎）、
`reasoning.py`（透传参数）、`config.py`（`krow` 段）、`super_agent.py`（`--krow-daemon`、`/krow`、空闲唤醒哨兵、忙标志）。
