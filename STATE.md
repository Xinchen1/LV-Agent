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
