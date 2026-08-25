<div align="center">

<img src="assets/screenshot.png" width="800" alt="LV Agent GitHub" />

</div>

# LV Agent

> 终端原生智能体框架。Deep thinking, real tools.

---

## 设计灵感：图灵机

> **灵感来源**：图灵机通用计算模型

LV Agent 的架构设计深受**图灵机**启发：
- **磁带内存模型** → 上下文窗口管理
- **状态转移函数** → Agent 状态管理
- **通用计算模型** → 通用工具执行
- **无限磁带** → 理论上的无限上下文窗口

<div align="center">
  <img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0iNDUwIiB2aWV3Qm94PSIwIDAgODAwIDQ1MCI+CiAgPGRlZnM+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InRhcGVHcmFkIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIwJSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0eWxlPSJzdG9wLWNvbG9yOiMxYzE5MTciLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI1MCUiIHN0eWxlPSJzdG9wLWNvbG9yOiMyZDJhMjYiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdHlsZT0ic3RvcC1jb2xvcjojMWMxOTE3Ii8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJoZWFkR3JhZCIgeDE9IjAlIiB5MT0iMCUiIHgyPSIwJSIgeTI9IjEwMCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojZDZhMDhhIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3R5bGU9InN0b3AtY29sb3I6I2NjN2E2MCIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICA8L2RlZnM+CiAgCiAgPCEtLSBCYWNrZ3JvdW5kIC0tPgogIDxyZWN0IHdpZHRoPSI4MDAiIGhlaWdodD0iNDUwIiBmaWxsPSJ1cmwoI3RhcGVHcmFkKSIvPgogIAogIDwhLS0gVGFwZSAtLT4KICA8cmVjdCB4PSI1MCIgeT0iMTgwIiB3aWR0aD0iNzAwIiBoZWlnaHQ9IjkwIiBmaWxsPSIjMWMxOTE3IiByeD0iOCIgc3Ryb2tlPSIjMzMzIiBzdHJva2Utd2lkdGg9IjIiLz4KICA8cmVjdCB4PSI1MCIgeT0iMTgwIiB3aWR0aD0iNzAwIiBoZWlnaHQ9IjkwIiBmaWxsPSJub25lIiBzdHJva2U9IiNkNmEwOGEiIHN0cm9rZS13aWR0aD0iMiIgcng9IjgiIHN0cm9rZS1kYXNoYXJyYXk9IjEwLDUiLz4KICAKICA8IS0tIFRhcGUgY2VsbHMgLS0+CiAgPGcgaWQ9InRhcGVDZWxscyI+CiAgICA8cmVjdCB4PSI3MCIgeT0iMTg1IiB3aWR0aD0iODAiIGhlaWdodD0iODAiIGZpbGw9IiMyYTJhMmEiIHJ4PSI0Ii8+CiAgICA8dGV4dCB4PSIxMTAiIHk9IjIzNSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIyNCIgZmlsbD0iI2Q2YTA4YSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC13ZWlnaHQ9ImJvbGQiPjE8L3RleHQ+CiAgICA8dGV4dCB4PSIxMTAiIHk9IjI2MCIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMCIgZmlsbD0iIzg4OCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+MDwvdGV4dD4KICAgIAogICAgPHJlY3QgeD0iMTYwIiB5PSIxODUiIHdpZHRoPSI4MCIgaGVpZ2h0PSI4MCIgZmlsbD0iIzJhMmEyYSIgcng9IjQiLz4KICAgIDx0ZXh0IHg9IjIwMCIgeT0iMjM1IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjI0IiBmaWxsPSIjZDZhMDhhIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXdlaWdodD0iYm9sZCI+MDwvdGV4dD4KICAgIDx0ZXh0IHg9IjIwMCIgeT0iMjYwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj4xPC90ZXh0PgogICAgCiAgICA8cmVjdCB4PSIyNTAiIHk9IjE4NSIgd2lkdGg9IjgwIiBoZWlnaHQ9IjgwIiBmaWxsPSIjMmEyYTJhIiByeD0iNCI+CiAgICAgIDx0ZXh0IHg9IjI5MCIgeT0iMjM1IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjI0IiBmaWxsPSIjZDZhMDhhIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXdlaWdodD0iYm9sZCI+MTwvdGV4dD4KICAgICAgPHRleHQgeD0iMjkwIiB5PSIyNjAiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTAiIGZpbGw9IiM4ODgiIHRleHQtYW5jaG9yPSJtaWRkbGUiPjE8L3RleHQ+CiAgICAKICAgIDxyZWN0IHg9IjM0MCIgeT0iMTg1IiB3aWR0aD0iODAiIGhlaWdodD0iODAiIGZpbGw9IiMyYTJhMmEiIHJ4PSI0Ij4KICAgICAgPHRleHQgeD0iMzgwIiB5PSIyMzUiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMjQiIGZpbGw9IiNkNmEwOGEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtd2VpZ2h0PSJib2xkIj4wPC90ZXh0PgogICAgICA8dGV4dCB4PSIzODAiIHk9IjI2MCIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMCIgZmlsbD0iIzg4OCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+MDwvdGV4dD4KICAgIAogICAgPHJlY3QgeD0iNDMwIiB5PSIxODUiIHdpZHRoPSI4MCIgaGVpZ2h0PSI4MCIgZmlsbD0iIzJhMmEyYSIgcng9IjQiPgogICAgICA8dGV4dCB4PSI0NzAiIHk9IjIzNSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIyNCIgZmlsbD0iI2Q2YTA4YSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC13ZWlnaHQ9ImJvbGQiPjE8L3RleHQ+CiAgICAgIDx0ZXh0IHg9IjQ3MCIgeT0iMjYwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj4wPC90ZXh0PgogICAgCiAgICA8cmVjdCB4PSI1MjAiIHk9IjE4NSIgd2lkdGg9IjgwIiBoZWlnaHQ9IjgwIiBmaWxsPSIjMmEyYTJhIiByeD0iNCI+CiAgICAgIDx0ZXh0IHg9IjU2MCIgeT0iMjM1IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjI0IiBmaWxsPSIjZDZhMDhhIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXdlaWdodD0iYm9sZCI+MDwvdGV4dD4KICAgICAgPHRleHQgeD0iNTYwIiB5PSIyNjAiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTAiIGZpbGw9IiM4ODgiIHRleHQtYW5jaG9yPSJtaWRkbGUiPjE8L3RleHQ+CiAgICAKICAgIDxyZWN0IHg9IjYxMCIgeT0iMTg1IiB3aWR0aD0iODAiIGhlaWdodD0iODAiIGZpbGw9IiMyYTJhMmEiIHJ4PSI0Ij4KICAgICAgPHRleHQgeD0iNjUwIiB5PSIyMzUiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMjQiIGZpbGw9IiNkNmEwOGEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtd2VpZ2h0PSJib2xkIj4xPC90ZXh0PgogICAgICA8dGV4dCB4PSI2NTAiIHk9IjI2MCIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMCIgZmlsbD0iIzg4OCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+MTwvdGV4dD4KICAgIAogICAgPHJlY3QgeD0iNzAwIiB5PSIxODUiIHdpZHRoPSI4MCIgaGVpZ2h0PSI4MCIgZmlsbD0iIzJhMmEyYSIgcng9IjQiPgogICAgICA8dGV4dCB4PSI3NDAiIHk9IjIzNSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIyNCIgZmlsbD0iI2Q2YTA4YSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC13ZWlnaHQ9ImJvbGQiPjA8L3RleHQ+CiAgICAgIDx0ZXh0IHg9Ijc0MCIgeT0iMjYwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj4xPC90ZXh0PgogIDwvZz4KICAKICA8IS0tIFJlYWQvV3JpdGUgaGVhZCAtLT4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyOTAsIDE0MCkiPgogICAgPHJlY3QgeD0iLTI1IiB5PSItMTAiIHdpZHRoPSI1MCIgaGVpZ2h0PSI1MCIgZmlsbD0idXJsKCNoZWFkR3JhZCkiIHJ4PSI4IiBzdHJva2U9IiNkNmEwOGEiIHN0cm9rZS13aWR0aD0iMiIvPgogICAgPHBhdGggZD0iTSAtNSAxNSBMIDAgNSBMIDUgMTUiIHN0cm9rZT0iI2Q2YTA4YSIgc3Ryb2tlLXdpZHRoPSIzIiBmaWxsPSJub25lIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KICAgIDx0ZXh0IHg9IjAiIHk9IjIwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEyIiBmaWxsPSIjMWMxOTE3IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LXdlaWdodD0iYm9sZCI+Ui9XPC90ZXh0PgogICAgPHRleHQgeD0iMCIgeT0iNDAiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iOSIgZmlsbD0iI2Q2YTA4YSIgdGV4dC1hbmNob3I9Im1pZGRsZSI+SEVBRDwvdGV4dD4KICA8L2c+CiAgCiAgPCEtLSBTdGF0ZSByZWdpc3RlciAtLT4KICA8cmVjdCB4PSI1MCIgeT0iNTAiIHdpZHRoPSIxODAiIGhlaWdodD0iODAiIGZpbGw9IiMyYTJhMmEiIHJ4PSI4IiBzdHJva2U9IiNkNmEwOGEiIHN0cm9rZS13aWR0aD0iMiIvPgogIDx0ZXh0IHg9IjE0MCIgeT0iODAiIGZvbnQtZmFtaWx5PSJtb25vc3BhY2UiIGZvbnQtc2l6ZT0iMTEiIGZpbGw9IiNkNmEwOGEiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtd2VpZ2h0PSJib2xkIj5TVEFURTogceKCgDwvdGV4dD4KICA8dGV4dCB4PSIxNDAiIHk9IjEwNSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSI5IiBmaWxsPSIjODg4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5DVVJSRU5UIFNUQVRFPC90ZXh0PgogIAogIDwhLS0gVHJhbnNpdGlvbiB0YWJsZSAtLT4KICA8cmVjdCB4PSI1ODAiIHk9IjUwIiB3aWR0aD0iMTcwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iIzFjMTkxNyIgcng9IjgiIHN0cm9rZT0iIzMzMyIgc3Ryb2tlLXdpZHRoPSIxIi8+CiAgPHRleHQgeD0iNjY1IiB5PSI3NSIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIxMiIgZmlsbD0iI2Q2YTA4YSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC13ZWlnaHQ9ImJvbGQiPs60IFRBQkxFPC90ZXh0PgogIDx0ZXh0IHg9IjU5MCIgeT0iMTAwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4Ij7OtChx4oKALDApIOKGkiAoceKCgSwxLFIpPC90ZXh0PgogIDx0ZXh0IHg9IjU5MCIgeT0iMTIwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4Ij7OtChx4oKBLDEpIOKGkiAoceKCgiwwLEwpPC90ZXh0PgogIDx0ZXh0IHg9IjU5MCIgeT0iMTQwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4Ij7OtChx4oKCLDApIOKGkiAoceKCgCwxLFIpPC90ZXh0PgogIDx0ZXh0IHg9IjU5MCIgeT0iMTYwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4Ij7OtChx4oKALDEpIOKGkiAoceKCgSwwLEwpPC90ZXh0PgogIDx0ZXh0IHg9IjU5MCIgeT0iMTgwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4Ij7OtChx4oKBLDApIOKGkiAoceKCgywxLFIpPC90ZXh0PgogIDx0ZXh0IHg9IjU5MCIgeT0iMjAwIiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjEwIiBmaWxsPSIjODg4Ij7OtChx4oKDLDEpIOKGkiAoceKCgCwwLGhhbHQpPC90ZXh0PgogIAogIDwhLS0gVGl0bGUgLS0+CiAgPHRleHQgeD0iNDAwIiB5PSIzMCIgZm9udC1mYW1pbHk9Im1vbm9zcGFjZSIgZm9udC1zaXplPSIyMCIgZmlsbD0iI2Q2YTA4YSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZm9udC13ZWlnaHQ9ImJvbGQiPlRVUklORyBNQUNISU5FPC90ZXh0PgogIDx0ZXh0IHg9IjQwMCIgeT0iNTAiIGZvbnQtc2l6ZT0iMTIiIGZpbGw9IiM4ODgiIHRleHQtYW5jaG9yPSJtaWRkbGUiPlVuaXZlcnNhbCBDb21wdXRhdGlvbiBNb2RlbCDihpIgTFYgQWdlbnQgQXJjaGl0ZWN0dXJlPC90ZXh0PgogIAogIDwhLS0gQXJyb3dzIHNob3dpbmcgbWFwcGluZyAtLT4KICA8ZyBzdHJva2U9IiNkNmEwOGEiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlLWRhc2hhcnJheT0iNSw1Ij4KICAgIDxsaW5lIHgxPSIyMzAiIHkxPSIxMzAiIHgyPSIxNDAiIHkyPSIxMzAiIG1hcmtlci1lbmQ9InVybCgjYXJyb3doZWFkKSIvPgogICAgPGxpbmUgeDE9IjQwMCIgeTE9IjEzMCIgeDI9IjQwMCIgeTI9IjE4MCIgbWFya2VyLWVuZD0idXJsKCNhcnJvd2hlYWQpIi8+CiAgICA8bGluZSB4MT0iNzUwIiB5MT0iMTMwIiB4Mj0iNjY1IiB5Mj0iMTUwIiBtYXJrZXItZW5kPSJ1cmwoI2Fycm93aGVhZCkiLz4KICA8L2c+CiAgCiAgPGRlZnM+CiAgICA8bWFya2VyIGlkPSJhcnJvd2hlYWQiIG1hcmtlcldpZHRoPSIxMCIgbWFya2VySGVpZ2h0PSI3IiByZWZYPSI5IiByZWZZPSIzLjUiIG9yaWVudD0iYXV0byI+CiAgICAgIDxwb2x5Z29uIHBvaW50cz0iMCAwLCAxMCAzLjUsIDAgNyIgZmlsbD0iI2Q2YTA4YSIvPgogICAgPC9tYXJrZXI+CiAgPC9kZWZzPgogIAogIDwhLS0gTGFiZWxzIC0tPgogIDx0ZXh0IHg9IjE0MCIgeT0iMzAwIiBmb250LXNpemU9IjExIiBmaWxsPSIjODg4IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5UYXBlIOKGlCBDb250ZXh0IFdpbmRvdzwvdGV4dD4KICA8dGV4dCB4PSIxNDAiIHk9IjMyMCIgZm9udC1zaXplPSIxMSIgZmlsbD0iIzg4OCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+SGVhZCDihpQgVG9vbCBFeGVjdXRpb248L3RleHQ+CiAgPHRleHQgeD0iNjY1IiB5PSIyNjAiIGZvbnQtc2l6ZT0iMTEiIGZpbGw9IiM4ODgiIHRleHQtYW5jaG9yPSJtaWRkbGUiPs60IEZ1bmN0aW9uIOKGkiBUb29sIFJvdXRlcjwvdGV4dD4KICA8dGV4dCB4PSI0MDAiIHk9IjQyMCIgZm9udC1zaXplPSIxMSIgZmlsbD0iIzg4OCIgdGV4dC1hbmNob3I9Im1pZGRsZSI+U3RhdGUgUmVnaXN0ZXIg4oaSIEFnZW50IFN0YXRlPC90ZXh0Pgo8L3N2Zz4KUFlFT0Y=" alt="Turing Machine Inspiration" width="600"/>
  <br>
  <em>图灵机启发的 LV Agent 架构：磁带内存、状态转移、通用计算</em>
</div>

---

## 项目简介

LV Agent 是一个**终端原生的智能体框架**，有基于 mythos 的一些思路，也是我研究机构 **Cleveris Research** 的作品。

它通过**多轮 LLM 调用 + 工具循环 + 自我修正**，实现"逐步深入思考"的过程。目前项目还很早期，有不少不足，期待与大家一起交流进步。

---

## 核心特性

### 推理与规划

| 特性 | 说明 |
|------|------|
| **多策略推理** | CoT / ReAct / Self-Consistency / Verification / MCTS |
| **自适应循环控制** | 简单问题少轮调用，难题多轮调用 |
| **任务规划** | 支持 adaptive / MCTS / graph / key_path 多种策略 |
| **自我修正** | 质量评估 + 自动修正 + 参数自适应调整 |

### 工具链

- **Web 搜索** — 多查询融合
- **文件操作** — 读 / 写 / grep / glob（Rust 加速可选）
- **代码执行** — Python / Bash，timeout 隔离
- **GitHub 搜索、PDF 读取、天气查询、网页抓取**
- **Telegram Bot 集成**（可独立运行）

### 记忆与上下文

- **知识图谱** — 实体-关系结构化长期记忆
- **经历记忆** — 跨会话向量相似度检索
- **记忆技能** — 从对话中提取可复用的策略（`/learn` + `/memskill`）
- **上下文压缩** — 长会话自动归纳，512 token 预算内保留核心信息

### Harness 运行时

- **事件溯源** — 执行过程可追溯、可重放
- **会话持久化** — SQLite 存储，支持历史会话选择（`/sessions`）
- **预算控制** — token 消耗 + 时间双重限制
- **工具确认** — 危险操作前请求用户批准
- **检查点** — 执行中断后可恢复

### 终端体验

- 头像像素画启动画面（Braille 渲染）
- 底部状态栏：token 占用 / 上下文进度
- 输入历史翻页 / Ctrl+S 草稿暂存 / Ctrl+\ Dashboard
- 实时流式输出 + 深色 / 浅色主题

---

## 技术架构

### API 模式（主流使用方式）

```
用户输入
    ↓
Planner 分解任务 → 分配 thinking loops
    ↓
[循环推理引擎]
    ├→ LLM 调用 （策略：CoT / ReAct / MCTS / Self-Consistency）
    ├→ 工具执行 → 观察结果 → 继续推理
    ├→ 自我修正 → 质量不达标则重答
    └→ ACT 自适应停止（达到质量阈值即退出循环）
    ↓
上下文压缩（长会话自动归纳）
    ↓
最终回答
```

### 深度研究模式

```
深度研究请求
    ↓
多角度搜索（多 query 并行）
    ↓
网页正文抓取 + 评分排序
    ↓
信息综合 + 信源评估
    ↓
生成 HTML 报告 → 自动打开浏览器
```

### 本地模型模式（实验性）

```
输入 → [Prelude] → [Recurrent Block 循环 T 次] → [Coda] → 输出

特性：
- 同一组权重循环多次，像人一样"反复思考"
- MoE 稀疏专家、LoRA 深度适配、LTI 稳定注入
- 需要本地 GPU / Metal 加速
```

---

## 快速开始

### 方式一：一键安装（推荐，全局 `lv` 命令）

```bash
git clone https://github.com/Xinchen1/LV-Agent.git
cd LV-Agent
./install.sh
source ~/.zshrc   # 或 source ~/.bashrc
lv                # 任意目录直接启动
```

### 方式二：手动运行（无需安装）

```bash
git clone https://github.com/Xinchen1/LV-Agent.git
cd LV-Agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp config.example.yaml config.yaml
./lv            # 或 python super_agent.py
```

**支持后端：** DeepSeek / OpenAI / Anthropic / OpenRouter / Ollama（本地离线）。

---

## 已实现的开箱功能

### 内置命令

| 命令 | 功能 |
|------|------|
| `/deep_research` | 多角度搜索 + 自动生成 HTML 报告 |
| `!命令` | 直接执行 shell 命令 |
| `@文件` | 把文件内容注入到输入 |
| `/model` | 实时切换模型 |
| `/strategy` | 切换推理策略 |
| `/compress` | 手动压缩上下文 |
| `/learn` | 从当前对话中学习记忆技能 |
| `/memskill` | 管理已学习的策略（list / evolve / snapshot / restore） |
| `/sessions` | 浏览历史会话 |
| `/dashboard` | 打开 Agent 状态面板 |
| `/drafts` | 查看暂存的输入草稿 |

### 快捷键

| 快捷键 | 动作 |
|--------|------|
| `Ctrl+S` | 暂存当前输入草稿 |
| `Ctrl+\` | 打开 Dashboard |
| `ESC` | 中断正在运行的任务（深度研究等长任务） |
| `↑↓` | 翻页输入历史 |

---

## 未来规划

### 仍在探索的方向

- 接入更多搜索源
- 更丰富的工具（数据库 / Git 操作 / 浏览器自动化）
- 更强的推理策略（Best-of-N / 投票机制）

### 中期探索

- 记忆检索改进（混合向量 + 关键词召回）
- 多模态支持（图像理解输入）
- 插件系统（MCP 协议初步支持）

### 长期方向

- 本地循环模型与 API 模型的深度融合
- 更智能的自主规划能力

---

## 链接

- **GitHub：** https://github.com/Xinchen1/LV-Agent
- **推荐后端：** DeepSeek Chat / DeepSeek Reasoner（开箱即用）
- **本地离线：** Ollama + `qwen2.5-coder:7b`（断网可用）

---

> 项目在成长，期待与各位交流。
