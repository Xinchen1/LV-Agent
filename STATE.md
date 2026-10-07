# STATE.md — 项目状态

## 架构
自进化 LLM Agent 工作台（execution_engine + policies + sqlite_memory + tools 协议 + Web/桌面双端）。

## 当前坐标
Agent 工作流已优化、站点品牌已更新（zh→en），config.yaml 已去除硬编码 API key（安全加固）；最新推送已同步至 origin/main。

## Changelog
- [2026-09-26] README 更新日志新增 2026-09-26 条目，记录工作流优化、站点品牌更新与安全加固两笔提交。
- [2026-09-27] hotswap 接入 LLM 后端（B 方案，可自动回滚退化模型）；模型最终输出配色改为终端原生命令风格；README 更新日志新增 2026-09-27 条目。
- [2026-10-05 20:33] 修复 agent 工作循环 11 项问题（复读熔断/kwargs 过滤/收敛 key/线程防堆积/限流退避/Key 掩码等），新增 11 个回归测试（297 全绿），.gitignore 屏蔽记忆隐私文件。
- [2026-10-05 21:05] 修复 fast_read 首次调用 KeyError 'model'（>5KB read 自动升级即炸），新增回归测试（298 全绿）。
- [2026-10-05 22:40] 精简 file_ops（砍 grep/find/analyze/backup/diff，三工具分工 + 纠偏提示）+ 输出美化（工具参数 JSON 化、错误 ✗ 只出现一次、[TOOL:] 不泄漏、footer 单复数），新增 12 个回归测试（310 全绿）。
- [2026-10-05 23:05] 配置向导 API Key 改为逐字符掩码回显（替换 getpass 完全不回显），支持退格擦除与粘贴转义序列（316 全绿）。
- [2026-10-05 23:35] 安装包去隐私重打包：electron-builder extraResources 移除 config.yaml、过滤 agent_project 下 config/memory/session 等隐私文件；根 config.yaml 真实 key 改 ${ENV:-} 占位；重建 dmg/zip 上传并发布 GitHub v1.0.0 正式版，官网 Download 按钮恢复可下载。
- [2026-10-05 23:55] 修复安装包“已损坏”：app 仅 linker-signed 被 Gatekeeper 判定损坏，手动 codesign --force --deep --sign - 重签，重打 zip(ditto)/dmg(hdiutil) 后重新上传 v1.0.0 并发布；release 说明补充右键打开/xattr -cr 提示。
- [2026-10-06 15:05] 安全扫描与加固：防火墙+隐身已开；SuperIDE 远程调试端口 9223 已从 app.asar 永久移除并重启；.zshrc 明文 API key 迁移到 chmod600 的 ~/.env.keys；.claude/.fcc/.lv_agent 的 .env 权限收紧为 600。
- [2026-10-06 17:30] 把本地 Qwopus3.5 接入各端：opencode/pi 新增 local ollama provider；gemini-cli 经 gateway 的 ollama/ 前缀路由（已重启网关并实测）；FCC .env 加入 ollama/ 模型与 allowlist（fcc watchdog 已清理）；SuperIDE config-store 新增 local-ollama 模型并重打 asar 重启。
- [2026-10-06 19:00] 新增 Qwopus3.5-9B-Coder：HF 拉取主包成功后 CDN 断连，用已缓存 blob 直接 ollama create 注册为 Qwopus3.5-9B-Coder:latest（可跑）；已加入 opencode/pi/FCC 双 .env allowlist/gemini settings/SuperIDE config-store，fcc /v1/models 与 /v1/messages 实测通过。
- [2026-10-06 13:50] 心跳统计改造：_version_beacon 改为每 15 分钟周期心跳并带 ts/uptime/真实包版本；新增 cloudflare-stats-worker（POST /api/version 记录 iid 去重装机量，GET /stats 看 24h/7d 活跃、版本与平台分布）。域名 lv-agent.cleveris.research 当前 NXDOMAIN，需 wrangler login 后创建 KV 并部署。
- [2026-10-06 14:00] GitHub 流量自动存档：scripts/archive_github_stats.sh 每天 09:30 经 launchd 归档 clones/views/stars/forks 到 data/github_traffic.jsonl（绕开 traffic 数据 14 天过期）。
- [2026-10-06 14:35] 系统梳理+站点对齐：清理 1.5G dist/、.codeartsdoer/、frontend-design-proposal/、.DS_Store/.pytest_cache/__pycache__、.bak/PDF/session 残档；/deep_research 补自动补全；pyproject 版本对齐站点 V1.1.0（DMG 仍是 1.0.0，下次打包需出 1.1.0）。
- [2026-10-06 22:40] 实测 lv 四任务后修复：config.yaml NIM key 外置到 .env（${NVAPI_KEY:-}）；删 .env 里失效的 OPENAI_API_KEY；置信度行过滤正则放宽（吞 1.02/85%/整数）；hotswap asyncio.iscoroutinefunction→inspect；telegram 未装时静默、fallback tokenizer 降 debug；新增 30s 慢后端提示与 _last_activity 空闲看门狗。 pytest 316 全绿。loops 偏高实为模型步数而非配置 bug，未改。
- [2026-10-06 22:55] UI 简洁化：工具结果兜底摘要从 800 字符→200（合并换行封顶）；⊙ STOP/⊙ Smart Dedup 的 cached 摘录不再透出 400 字原始结果（改为只报字符数）。pytest 316 全绿。
- [2026-10-06 23:10] file_ops 本地强操作补齐：新增 mkdir/delete/move/copy/stat 五个原生 action（Python 实现、系统路径防护仍生效、Rust 后端不支持的操作在 execute 层拦截处理），tool 描述与 schema 同步，test_file_ops_slim 白名单更新。
- [2026-10-06 23:15] MCP filesystem 映射修 bug：create_directory 原错误映射到 file_ops list（实际执行是列目录而不是建目录），已改指 mkdir；get_file_info 改指 stat；补齐 delete_file/move_file/copy_file→delete/move/copy。
- [2026-10-06 23:30] 本地操作快路：_try_direct_intent_path 让高置信只读意图(file_ops list/read/stat/exists/multi_read/verify、glob、search_files、weather)直接执行不经 LLM；_classify_intent 新增"X 几个文件夹/有哪些文件"规则(桌面/文档/下载/当前目录)。pytest 316 全绿，实测 桌面几个文件夹 毫秒直出。
- [2026-10-06 23:45] 思考过程不再全量刷屏：LiveStreamFilter 在中文输出尚未开始前抑制匹配英文自言自语模式的行；PlainStreamAdapter.emit_reasoning 改为只打一行 "thinking…" 轻提示。
- [2026-10-07 00:05] 修 rust 探测误判：_check_rust_binary 原用 --version 探测，但 one-shot 模式 read_to_string(stdin) 在 TTY 下永远阻塞 → 3s 超时 → 误判"不可用"白掉 Python fallback 还拖慢启动 3s；改用真实 JSON 请求 + input=（写完即关），RUST_AVAILABLE 恢复 True，list 走 rust 进程池。
- [2026-10-07 00:20] 本地操作极致增强：_try_direct_intent_path 提到 _try_fast_routes 最顶（原在 simple query 之后被 LLM 快路截胡）；rust 进程池 init 时后台预热（消除首调 21ms 冷启动）；file_ops 新增 tree（深度受限目录树）/du（目录大小）原生 action；prewarm except 打日志过异常卫生基线。实测：直达路径 11.7ms 全程 0 LLM，list 7ms/stat 0.3ms/tree 2ms/du 2ms（15k 文件目录 1.8s）。
- [2026-10-07 18:50] 恢复 build_mac_app.sh 为 e9192f1 真实构建脚本（HEAD 中曾被系统分析文本覆盖）。
- [2026-10-07 19:05] 增强本地操作: grep/find/analyze/backup/diff 重新集成进 file_ops, 内部委托到 search_files/glob/bash_exec 实现; 描述更新, enum 更新为 21 个 action; test_file_ops_slim 已适配, pytest 316 全绿。
- [2026-10-07 18:35] 修 fts5 + 调研意图判定：MATCH 查询消毒（每词加引号 OR 拼接，失败回退 LIKE 分词）；is_research_report_task 补 搜集/搜罗/收集 动词、pdf/docx/html 显式输出；deep_prefix 长度 off-by-one；extract_research_topic 对 搜集 正确剥离。实测“搜集实在智能的产品技术，生成pdf,打开”→ topic=“实在智能的产品技术”，深度调研AI→True。
