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
