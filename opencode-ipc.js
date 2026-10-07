const { ipcMain } = require('electron');
const http = require('http');
const https = require('https');
const net = require('net');
const log = require('electron-log');
const { getOpencodeState, ensureReady, CANCRI_PROVIDER_ID, CANCRI_MODEL_ID, CANCRI_BASE_URL, OPENROUTER_PROVIDER_ID, OPENROUTER_BASE_URL, OPENCODE_GO_PROVIDER_ID, OPENCODE_GO_BASE_URL, OPENCODE_ZEN_PROVIDER_ID, OPENCODE_ZEN_OC_PROVIDER_ID, OPENCODE_ZEN_BASE_URL, DEEPSEEK_PROVIDER_ID, DEEPSEEK_BASE_URL, ensureDeepseekProvider } = require('../services/opencode-server');
const { EXPERT_SYSTEM_PROMPT } = require('../services/ai-client');
const intelligenceContext = require('../services/intelligence-context');

// 端点预检：TCP 直连，6 秒不通则快速失败（不等 30 分钟），错误码 endpoint-down
function endpointReachable(urlStr, timeoutMs = 6000) {
  return new Promise((resolve) => {
    let url;
    try {
      url = new URL(urlStr);
    } catch (e) {
      resolve(false);
      return;
    }
    const port = url.port ? parseInt(url.port, 10) : (url.protocol === 'https:' ? 443 : 80);
    const socket = net.connect({ host: url.hostname, port, timeout: timeoutMs });
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      try { socket.destroy(); } catch (e) {}
      resolve(ok);
    };
    socket.on('connect', () => finish(true));
    socket.on('timeout', () => finish(false));
    socket.on('error', () => finish(false));
  });
}

// 测试时可覆盖目标（单测用），生产走 getOpencodeState()
let testTarget = null;
function __setTestTarget(baseUrl, password) {
  testTarget = baseUrl ? { baseUrl, password } : null;
}

function ocTarget() {
  if (testTarget) return testTarget;
  const st = getOpencodeState();
  if (!st.ready || !st.baseUrl) return null;
  return { baseUrl: st.baseUrl, password: st.password };
}

// 需要服务的操作先过这一关：标志未就绪时做一次懒复检（自愈启动检测失手的情况）
async function requireServer() {
  if (ocTarget()) return null;
  try {
    if (await ensureReady()) return null;
  } catch (e) {}
  return { success: false, error: 'OpenCode 服务未就绪', code: 'failed' };
}

/**
 * OpenCode HTTP 调用。
 *
 * 关键改造（Composer 挂死 30 分钟的根因）：
 *   旧实现只有一个 `timeout: timeoutMs`（Composer 路径默认 1800s = 30 分钟），
 *   于是上游挂死时**要等满 30 分钟**才判超时 —— 日志里同一请求
 *   oc_1790862431208 出现在 21:47 / 22:17 / 22:47 / 23:17，正好每 30 分钟一次。
 *
 *   现在是**两层预算**：
 *     - 静默预算：从"最后一次真实活动"算起，分两段
 *       · idleMs（默认 10 分钟）：有过活动之后才静默这么久才算异常
 *       · ttftMs（默认 5 分钟）：一次活动都没有才算异常（免费模型排队常超 90s）
 *     - 总时长兜底 totalMs：防"一直有微小活动但永不停"。
 *
 * ⚠️ 活动判定必须包含 SSE（这是本实现最关键、也最容易再次踩错的一点）：
 *   `POST /session/:id/message` 是**跑完才回包**的阻塞请求，响应体在整个 Agent
 *   运行期间不产生任何字节；实时活动（文本增量 / 工具调用 / 完成）全部走 SSE `/event`。
 *   曾因此把静默预算设成 90s 且只看 POST 响应体 → 52 次超时**全部**被判成"首字超时"，
 *   0 次"中途卡住"，而成功轮次中位 244s、53% 超过 90s —— 等于把一半以上的正常任务杀掉，
 *   还会因自动续跑在同一 req id 上堆出最多 10 个孤儿运行。
 *   所以调用方必须传 `opts.activity = { lastActivityAt }` 并在 SSE 回调里更新它。
 *
* 超时原因会如实区分：ttft（全程无任何活动）/ idle（有过活动后卡住），
  * 便于上层决定"换模型/换通道"还是"重试"。
 */

// 生产默认值。**不要再调回 90s** —— 实测 19 次成功轮次中位 244s / p75 701s / p90 980s，
// 90s 预算会误杀其中 53%（详见 test/composer-idle-timeout.test.js 的回归断言）。
const DEFAULT_IDLE_MS = 600000;   // 有过活动之后的静默容忍：10 分钟（长工具执行期上游不发字节）
const DEFAULT_TTFT_MS = 300000;   // 全程无活动的容忍：5 分钟（免费模型排队常超过 90s）

/**
 * 活动信标：调用方在 SSE 回调里 touch()，ocFetch 立刻知道"上游还活着"。
 *
 * 为什么是**推送**而不是只读时间戳：若 ocFetch 靠每秒轮询 lastActivityAt，
 * 就会漏掉"活动恰好在两次轮询之间结束"的爆发 —— 表现为把"跑了一阵才卡住"
 * 误报成"首字超时"，也会让长任务在轮询间隙被误杀。
 * onActivity 让每次 touch 同步通知到 ocFetch，事件不丢。
 */
function createActivityBeacon() {
  const listeners = new Set();
  return {
    lastActivityAt: 0,
    touch() {
      this.lastActivityAt = Date.now();
      for (const fn of listeners) {
        try { fn(this.lastActivityAt); } catch (e) { /* 监听器异常不影响主流程 */ }
      }
    },
    /** @returns {() => void} 取消订阅（ocFetch 在 cleanup 里调用，防监听器泄漏） */
    onActivity(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    }
  };
}

function ocFetch(method, apiPath, body, timeoutMs = 30000, opts = {}) {
  const target = ocTarget();
  if (!target) return Promise.reject(new Error('OpenCode 服务未就绪'));
  // 静默预算：从"最后一次真实活动"算起，分两段
  //   idleMs —— 已经有过活动之后的静默容忍度。工具执行期间上游本来就不发字节
  //             （跑一次 npm test 可能几分钟无输出），默认 10 分钟。
  //   ttftMs —— 一次活动都没有时的容忍度。免费模型排队常超过 90s，默认 5 分钟。
  //             调用方只传 idleMs 而不传 ttftMs 时，两段都用 idleMs（保持旧调用语义）。
  const idleExplicit = Number(opts.idleMs) > 0;
  const envIdle = Number(process.env.SUPERIDE_OC_IDLE_TIMEOUT_MS);
  const envTtft = Number(process.env.SUPERIDE_OC_TTFT_TIMEOUT_MS);
  const idleMs = idleExplicit
    ? Number(opts.idleMs)
    : (envIdle > 0 ? envIdle : DEFAULT_IDLE_MS);
  const ttftMs = Number(opts.ttftMs) > 0
    ? Number(opts.ttftMs)
    : (envTtft > 0 ? envTtft : (idleExplicit ? idleMs : DEFAULT_TTFT_MS));
  const totalMs = Number(opts.totalMs) > 0 ? Number(opts.totalMs) : timeoutMs;
  // 活动信标：**关键**。
  // POST /session/:id/message 是"跑完才回包"的阻塞请求，真正的事件流走 SSE /event。
  // 若只看 POST 响应体，任何超过静默预算的任务都会被误判成挂死 ——
  // 实测 52 次全是「首字超时」而 0 次「中途卡住」，且成功轮次中位 244s、53% 超过当时的 90s 预算。
  // 所以必须由调用方把 SSE 活动喂进来。
  const beacon = (opts.activity && typeof opts.activity.lastActivityAt === 'number') ? opts.activity : null;

  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    let firstActivityAt = 0;
    let lastActivityAt = 0;
    let sawExternal = false;      // 曾收到外部(SSE)活动 —— 必须**粘住**：
                                  // 活动停止不代表"从未活动过"，
                                  // 否则会把"跑了一阵才卡住"误报成"首字超时"
    let quietTimer = null;
    let totalTimer = null;
    let unsubscribe = null;
    let settled = false;

    const beaconAt = () => {
      const v = beacon ? beacon.lastActivityAt : 0;
      if (v > 0) sawExternal = true;
      return v;
    };
    const sawActivity = () => firstActivityAt > 0 || sawExternal;
    /** 外部(SSE)活动到达：同步记为活动（不依赖轮询，事件不丢） */
    const onExternalActivity = (at) => {
      if (at) lastActivityAt = at;
      sawExternal = true;
    };

    const cleanup = () => {
      if (quietTimer) clearInterval(quietTimer);
      if (totalTimer) clearTimeout(totalTimer);
      if (unsubscribe) { try { unsubscribe(); } catch (e) {} }
      quietTimer = totalTimer = unsubscribe = null;
    };
    const failWith = (message, code) => {
      if (settled) return;
      settled = true;
      cleanup();
      const e = new Error(message);
      e.code = code;
      e.ttft = sawActivity() ? firstActivityAt - startedAt : null;
      reject(e);
    };
    /** 收到真实活动（POST 字节）→ 更新首/末活动时间 */
    const touch = () => {
      const now = Date.now();
      if (!firstActivityAt) firstActivityAt = now;
      lastActivityAt = now;
    };
    function onQuietExpired() {
      // 一次活动都没有 → ttft（上游不可用/排队过久）；有过活动后静默 → idle
      const saw = sawActivity();
      const budget = saw ? idleMs : (ttftMs > 0 ? ttftMs : idleMs);
      failWith(
        saw
          ? `OpenCode 上游中断：${Math.round(idleMs / 1000)}s 无新数据`
          : `OpenCode 首字超时：${Math.round(budget / 1000)}s 未收到任何响应`,
        saw ? 'idle' : 'ttft'
      );
      try { req.destroy(); } catch (e) { /* 已销毁 */ }
    }

    // 用「每秒核对静默时长」而不是「收到字节就重新 setTimeout」：
    // 活动有两个来源（本响应体 + 外部 SSE 信标），轮询才能把两者统一进 lastActivityAt。
    // 信标若支持 onActivity 则订阅之（事件不丢，见 createActivityBeacon 注释）；
    // 退化的纯 { lastActivityAt } 信标则靠轮询兜底。
    if (beacon && typeof beacon.onActivity === 'function') {
      unsubscribe = beacon.onActivity(onExternalActivity);
    }
    quietTimer = setInterval(() => {
      const last = Math.max(lastActivityAt, beaconAt(), startedAt);
      const budget = sawActivity() ? idleMs : (ttftMs > 0 ? ttftMs : idleMs);
      if (Date.now() - last >= budget) onQuietExpired();
    }, 1000);
    if (quietTimer.unref) quietTimer.unref();

    totalTimer = setTimeout(() => {
      failWith(`OpenCode 总时长超限（${Math.round(totalMs / 1000)}s）`, 'total');
      try { req.destroy(); } catch (e) { /* 已销毁 */ }
    }, totalMs);

    let payload = null;
    const headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Basic ' + Buffer.from(`opencode:${target.password || ''}`).toString('base64')
    };
    if (body !== undefined) {
      payload = JSON.stringify(body);
      headers['Content-Length'] = Buffer.byteLength(payload);
    }
    let url;
    try {
      url = new URL(apiPath, target.baseUrl);
    } catch (e) {
      failWith(e.message, 'invalid_url');
      return;
    }
    const client = url.protocol === 'https:' ? https : http;
    const req = client.request({
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
      method,
      headers
    }, (res) => {
      let data = '';
      res.on('data', (c) => { touch(); data += c; });
      res.on('end', () => {
        if (settled) return;
        settled = true;
        cleanup();
        if (res.statusCode < 200 || res.statusCode >= 300) {
          const e = new Error(`OpenCode HTTP ${res.statusCode}: ${data.slice(0, 300)}`);
          e.code = 'http_' + res.statusCode;
          reject(e);
          return;
        }
        if (!data) { resolve(null); return; }
        try { resolve(JSON.parse(data)); }
        catch { resolve(data); }
      });
    });
    req.on('error', (e) => failWith(e.message, 'network'));
    if (payload) req.write(payload);
    req.end();
  });
}

/**
 * Composer 路径的进度心跳：把 TTFT / 已输出字符 / 已等待秒数推给渲染层。
 * 目的：上游挂死时界面必须**说清在等什么、等了多久**，而不是看起来像卡死。
 */
function startProgressReporter({ requestId, intervalMs = 1000, onTick }) {
  const startedAt = Date.now();
  let firstTokenAt = 0;
  let chars = 0;
  const markFirstToken = (deltaChars = 0) => {
    if (!firstTokenAt) firstTokenAt = Date.now();
    chars += deltaChars;
  };
  const timer = setInterval(() => {
    try {
      onTick({
        waitedSec: Math.round((Date.now() - startedAt) / 1000),
        ttftMs: firstTokenAt ? firstTokenAt - startedAt : null,
        chars
      });
    } catch (e) { /* 心跳异常不影响主流程 */ }
  }, intervalMs);
  if (timer.unref) timer.unref();
  return {
    markFirstToken,
    stop() { clearInterval(timer); }
  };
}

// 订阅全局 SSE /event，回调每个事件对象；返回 { close() }
function ocSubscribe(onEvent) {
  const target = ocTarget();
  if (!target) throw new Error('OpenCode 服务未就绪');
  const url = new URL('/event', target.baseUrl);
  const client = url.protocol === 'https:' ? https : http;
  let closed = false;
  let buffer = '';
  const req = client.request({
    hostname: url.hostname,
    port: url.port,
    path: url.pathname,
    method: 'GET',
    headers: {
      'Accept': 'text/event-stream',
      'Authorization': 'Basic ' + Buffer.from(`opencode:${target.password || ''}`).toString('base64')
    }
  }, (res) => {
    res.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        const t = line.trim();
        if (!t.startsWith('data:')) continue;
        const raw = t.slice(5).trim();
        if (!raw || raw === '[DONE]') continue;
        try {
          onEvent(JSON.parse(raw));
        } catch (e) { /* 忽略心跳/非JSON */ }
      }
    });
    res.on('end', () => { closed = true; });
    res.on('error', () => { closed = true; });
  });
  req.on('error', () => { closed = true; });
  req.end();
  return { close: () => { closed = true; try { req.destroy(); } catch (e) {} } };
}

// ===== 权限策略：只放行项目根目录内的操作 =====
const sessionProjects = new Map(); // sessionId -> projectRoot
let permWatcherStarted = false;

function normalizePath(p) {
  if (!p) return '';
  return String(p).replace(/\\/g, '/').replace(/\/+$/, '');
}

function inProject(absPath, projectRoot) {
  const p = normalizePath(absPath);
  const r = normalizePath(projectRoot);
  if (!p || !r) return false;
  return p === r || p.startsWith(r + '/');
}

const DANGEROUS_CMD = [
  /rm\s+-rf?\s+\/(?!\S)/, /rm\s+-rf?\s+--no-preserve-root/, /\bmkfs\b/,
  /\bdd\s+.*of=\/dev\//, /:\(\)\s*\{[^}]*\}\s*;/, /\b(shutdown|reboot|halt|poweroff)\b/,
  />\s*\/dev\//, /chmod\s+-R\s+777\s+\//
];

function decidePermission(sessionId, perm) {
  const root = sessionProjects.get(sessionId);
  if (!root) return { reply: 'reject', message: '未知会话' };
  const md = perm.metadata || {};
  // 写/读/编辑类：看 filepath
  if (md.filepath) {
    if (inProject(md.filepath, root)) return { reply: 'always' };
    return { reply: 'reject', message: `只允许操作项目内文件（项目根：${root}），请使用绝对路径重试` };
  }
  // 目录类：directories 必须全在项目内
  if (Array.isArray(md.directories) && md.directories.length > 0) {
    const ok = md.directories.every((d) => inProject(d, root));
    return ok
      ? { reply: 'always' }
      : { reply: 'reject', message: `只允许操作项目内目录（项目根：${root}）` };
  }
  // shell 命令：危险直接拒；含项目外绝对路径则拒（要求绝对路径）
  const cmd = md.command || (perm.tool && perm.tool.command) || '';
  if (cmd) {
    if (DANGEROUS_CMD.some((re) => re.test(cmd))) {
      return { reply: 'reject', message: '危险命令被拒绝执行' };
    }
    const absPaths = cmd.match(/(^|[\s"'=])(\/(?:[^\/\s"'|;&]+))+/g) || [];
    const outside = absPaths.map((s) => s.trim().replace(/^["'=]/, '')).filter((p) => p.startsWith('/') && !inProject(p, root));
    if (outside.length > 0) {
      return { reply: 'reject', message: `命令涉及项目外路径 ${outside[0]}，请只操作项目根 ${root} 内的绝对路径` };
    }
    return { reply: 'once' };
  }
  // 其他类型默认拒绝（收紧），并给出原因让 Agent 自纠
  return { reply: 'reject', message: `该操作不在项目 ${root} 内，已拒绝` };
}

function ensurePermWatcher() {
  if (permWatcherStarted) return;
  permWatcherStarted = true;
  try {
    ocSubscribe(async (evt) => {
      try {
        if (!evt || evt.type !== 'permission.asked') return;
        const props = evt.properties || {};
        const sessionId = props.sessionID;
        if (!sessionId || !sessionProjects.has(sessionId)) return;
        const decision = decidePermission(sessionId, props);
        log.info(`[OpenCode IPC] permission ${props.permission} in ${sessionId}: ${decision.reply}`);
        await ocFetch('POST', `/permission/${props.id}/reply`, { reply: decision.reply, message: decision.message }, 15000);
      } catch (e) {
        log.warn('[OpenCode IPC] permission reply failed:', e.message);
      }
    });
    log.info('[OpenCode IPC] permission watcher started');
  } catch (e) {
    permWatcherStarted = false;
    log.warn('[OpenCode IPC] permission watcher failed to start:', e.message);
  }
}

function pushToRenderer(mainWindow, requestId, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('opencode-stream', { requestId, ...payload });
  }
}

function buildSystemPrompt(projectRoot, modelLabel) {
  // 默认提示词统一使用通用身份，不把模型名/引擎名注入模型看到的 system（保密红线）。
  // modelLabel 仅作为缓存键参与，不再拼进提示词。
  const expert = EXPERT_SYSTEM_PROMPT;
  return `${expert}\n\n【工程约束 - 必须遵守】\n项目根目录：${projectRoot || '（未打开项目）'}\n1. 所有文件读写必须使用以项目根开头的绝对路径，严禁相对路径，严禁读写项目之外的任何文件。\n2. shell 命令必须先 cd 到项目根或全程使用绝对路径。\n3. 调用 write/edit 时 filePath 与 content 必须完整填充，严禁发送空参数调用；不确定内容时先 read 确认。\n4. 任务完成时用自然语言总结做了哪些修改（列出文件路径）。\n5. 默认不向用户透露自身的技术细节、内部架构、实现方式，以及底层或所使用模型的具体名称。`;
}

// ===== 上下文缓存：省 token + 提速（不改变模型看到的有效信息，只去掉每轮重复重建的同一大段系统提示）=====
// system 提示在会话生命周期内只随 (root, model) 变化，按 key 编译一次后缓存复用。
const systemPromptCache = new Map();      // `${root}|${modelLabel}` -> 已编译的完整 system

function cachedSystemPrompt(root, modelLabel) {
  const key = `${root || ''}|${modelLabel || ''}`;
  let s = systemPromptCache.get(key);
  if (!s) {
    s = buildSystemPrompt(root, modelLabel);
    systemPromptCache.set(key, s);
  }
  return s;
}

// 按 provider 解析目标端点（预检用）
const LOCAL_OLLAMA_PROVIDER_ID = 'local-ollama';
const LOCAL_OLLAMA_OC_PROVIDER_ID = 'ollama';          // opencode.json 里真实的 providerID
const LOCAL_OLLAMA_BASE_URL = 'http://localhost:11434/v1';

function endpointForProvider(providerID) {
  if (providerID === LOCAL_OLLAMA_PROVIDER_ID || providerID === LOCAL_OLLAMA_OC_PROVIDER_ID) return LOCAL_OLLAMA_BASE_URL;
  if (providerID === OPENROUTER_PROVIDER_ID) return OPENROUTER_BASE_URL;
  if (providerID === OPENCODE_GO_PROVIDER_ID) return OPENCODE_GO_BASE_URL;
  if (providerID === OPENCODE_ZEN_PROVIDER_ID) return OPENCODE_ZEN_BASE_URL;
  if (providerID === DEEPSEEK_PROVIDER_ID) return DEEPSEEK_BASE_URL;
  return CANCRI_BASE_URL;
}

function extractText(parts) {
  let out = '';
  for (const p of parts || []) {
    if (p && p.type === 'text' && p.text) out += p.text;
  }
  return out;
}

// 解析 POST /session/:id/message 返回：单条 {info,parts} 或消息数组都可能。
// 运行出错时错误挂在 info.error（如 401 Authentication Fails）——必须上抛，
// 否则 parts 为空会静默返回 success，前端只看到“未收到响应”。
function parseRunResponse(data) {
  const msgs = Array.isArray(data) ? data : (data ? [data] : []);
  const msg = msgs.find((m) => m && m.info && m.info.role === 'assistant')
    || msgs.find((m) => m && m.info && m.info.error)
    || msgs[msgs.length - 1]
    || {};
  const runError = (msg.info && msg.info.error)
    || msgs.map((m) => m && m.info && m.info.error).find(Boolean)
    || null;
  const parts = (msg.parts && msg.parts.length)
    ? msg.parts
    : (Array.isArray(data) ? [] : ((data && data.parts) || []));
  // finish 字段各版本命名不一（finish / finishReason / stop_reason / reason），
  // 归一后回传，前端才能可靠识别"输出被 max_tokens 截断"并触发自动续跑。
  const info = msg.info || {};
  const finish = String(
    info.finish
    || info.finishReason
    || info.stop_reason
    || info.stopReason
    || info.reason
    || (info.time && info.time.reason)
    || (!Array.isArray(data) && data && data.info && (data.info.finish || data.info.finishReason || data.info.reason))
    || ''
  );
  return { parts, finish, runError };
}

/**
 * 从 opencode 响应里提取 token 用量。
 * 背景：Composer 的主路径走独立 opencode 进程，usage 原本被整个丢掉，
 * 导致 Composer 模式既不显示也不累计 token。
 * opencode 各版本字段名不一，这里做兼容提取，取不到就返回 null（由前端按文本估算兜底）。
 */
function extractUsage(data) {
  const msgs = Array.isArray(data) ? data : (data ? [data] : []);
  let input = 0;
  let output = 0;
  let cost = 0;
  let found = false;

  const num = (v) => (typeof v === 'number' && isFinite(v) ? v : 0);

  for (const m of msgs) {
    const info = (m && m.info) || m || {};
    // 形状 A：info.tokens = { input, output, reasoning, cache }
    const t = info.tokens;
    if (t && typeof t === 'object') {
      const i = num(t.input ?? t.prompt ?? t.input_tokens);
      const o = num(t.output ?? t.completion ?? t.output_tokens);
      const r = num(t.reasoning);
      if (i || o || r) found = true;
      input += i;
      output += o + r;   // 推理 token 同样是输出成本
    }
    // 形状 B：info.usage = { prompt_tokens, completion_tokens, ... }
    const u = info.usage;
    if (u && typeof u === 'object') {
      const i = num(u.prompt_tokens ?? u.input_tokens);
      const o = num(u.completion_tokens ?? u.output_tokens);
      if (i || o) found = true;
      input += i;
      output += o;
    }
    // 形状 C：顶层 total
    if (!found && typeof info.total_tokens === 'number') {
      const i = num(info.prompt_tokens ?? info.input_tokens);
      output += Math.max(0, num(info.total_tokens) - i);
      found = true;
    }
    if (typeof info.cost === 'number' && isFinite(info.cost)) cost += info.cost;
  }

  if (!found) return null;
  return {
    prompt_tokens: Math.round(input),
    completion_tokens: Math.round(output),
    total_tokens: Math.round(input + output),
    cost_usd: Math.round(cost * 1e6) / 1e6,
  };
}

function runErrorMessage(runError) {
  if (!runError) return '';
  if (typeof runError === 'string') return runError;
  return (runError.data && runError.data.message)
    || runError.message
    || JSON.stringify(runError).slice(0, 300);
}

function extractToolFiles(parts) {
  const files = [];
  for (const p of parts || []) {
    if (p && p.type === 'tool' && (p.tool === 'write' || p.tool === 'edit')) {
      const fp = p.state && p.state.input && (p.state.input.filePath || p.state.input.path);
      if (fp && !files.includes(fp)) files.push(fp);
    }
  }
  return files;
}

// ===== 阶段4：结果回灌 + 跨会话进化 =====
// 会话结束后（opencode:chat 完成/失败/超时），把 (消息, 工具序列, 成败) 写回
// 洞察/技能/记忆，形成自进化闭环。全部失败仅 warn 不阻断主流程。
function finalizeSessionEvolution({ sessionId, projectRoot, userText, parts, finish, result }) {
  try {
    const completed = !!finish && finish !== 'length';
    const msgs = (Array.isArray(parts) && parts.length > 0)
      ? parts.map(p => ({ role: p.type === 'text' ? 'assistant' : 'agent', content: p.text || '' }))
      : [];
    // 用户消息作为首条上下文，供 extractPatterns 的任务分类
    if (userText) msgs.unshift({ role: 'user', content: userText });
    const toolCalls = (Array.isArray(parts) ? parts : []).map(p => ({ name: p.tool || 'unknown', status: (p.state && p.state.status) || '' }));

    // 4.1 会话洞察：记录完整会话（消息/工具/token/成败）
    // 注意：这些服务都在 services/ 下（本文件在 ipc/ 下），路径写错会让整个 finalize 静默失败
    const insights = require('../services/session-insights');
    if (insights && typeof insights.endSession === 'function') {
      insights.endSession({
        id: sessionId,
        title: (userText || '').slice(0, 100),
        model: '',
        messages: msgs,
        toolCalls,
        promptTokens: 0,
        completionTokens: 0,
        projectPath: projectRoot || '',
        startTime: Date.now() - 60000,
        endTime: Date.now(),
      });
    }

    // 4.2 技能进化：成功会话提取模式沉淀为 active 技能，失败会话记录 usage(success:false)
    const skillEvo = require('../services/skill-evolution-service');
    if (skillEvo) {
      if (completed && typeof skillEvo.extractPatterns === 'function') {
        // toolCalls 必须传：工具序列的真源是会话工具调用（消息文本里没有工具标记）
        const patterns = skillEvo.extractPatterns(msgs, { completed: true, toolCalls });
        for (const pt of patterns || []) {
          try {
            const skill = skillEvo.create(pt.name, pt.description, {
              category: pt.category,
              content: pt.content,
              triggerPatterns: pt.triggerPatterns,
              tags: pt.tags,
              source: pt.source,
              pinned: false,
            });
            if (skill && skill.id && typeof skillEvo.recordUsage === 'function') {
              skillEvo.recordUsage(skill.id, { trigger: userText || '', projectPath: projectRoot || '', success: true });
            }
          } catch (e) { log.warn('[OpenCode IPC] extract skill failed:', e.message); }
        }
      } else if (typeof skillEvo.recordUsage === 'function') {
        // 失败会话：把命中的技能记一次失败，供 getBehaviorHints 规避
        const matched = skillEvo.matchTrigger && skillEvo.matchTrigger(userText || '');
        if (matched && matched.id) {
          skillEvo.recordUsage(matched.id, { trigger: userText || '', projectPath: projectRoot || '', success: false });
        }
      }
    }

    // 4.3 验证通过的关键决策写回增强记忆（project-scoped），供阶段2注入
    if (completed && result && result.toolFiles && result.toolFiles.length > 0) {
      const enh = require('../services/enhanced-memory-service');
      if (enh && typeof enh.add === 'function') {
        const decisionText = `任务完成（finish=${finish}）：${userText.slice(0, 120)}。涉及文件：${result.toolFiles.join(', ')}`;
        enh.add(decisionText, { scope: 'project', projectPath: projectRoot || '', type: 'decision', importance: 0.7 });
      }
    }
  } catch (e) {
    log.warn('[OpenCode IPC] finalizeSessionEvolution failed:', e.message);
  }
}

function registerOpencodeIPC(mainWindow) {
  // 创建会话
  ipcMain.handle('opencode:create-session', async (_event, payload = {}) => {
    try {
      {
        const notReady = await requireServer();
        if (notReady) return notReady;
      }
      const { title, projectRoot } = payload;
      const data = await ocFetch('POST', '/session', { title: title || 'SuperIDE Composer', agent: 'build' }, 30000);
      if (projectRoot && data && data.id) sessionProjects.set(data.id, projectRoot);
      ensurePermWatcher();
      return { success: true, id: data && data.id };
    } catch (e) {
      log.error('[OpenCode IPC] create-session failed:', e.message);
      return { success: false, error: e.message };
    }
  });

  // 发送消息（阻塞等待完成，同时经 SSE 转发增量事件）
  ipcMain.handle('opencode:chat', async (_event, payload = {}) => {
    const { sessionId, text, projectRoot, requestId, model } = payload;
    // 默认 30 分钟：长任务（多轮工具）经常超过 10 分钟，过短超时会把进行中的任务掐死
    const timeoutMs = (payload.timeoutMs && payload.timeoutMs > 0) ? payload.timeoutMs : 1800000;
    if (!sessionId || !text) return { success: false, error: '缺少 sessionId 或 text', code: 'failed' };

    // 检查 OpenCode 服务器是否可用
    const notReady = await requireServer();

    // OpenCode 不可用时，降级到本地 AgentCore
    if (notReady) {
      log.info('[OpenCode IPC] Server not ready, falling back to AgentCore local');
      try {
        const agentCoreLocal = require('../services/agent-core-local');
        const config = require('../services/config-store').loadConfig();
        const result = await agentCoreLocal.agentChat(text, {
          sessionId,
          model: model?.modelID || config?.model?.name,
          provider: model?.providerID || config?.model?.provider,
          endpoint: config?.model?.endpoint,
          apiKey: config?.model?.apiKey,
          fileContext: payload.fileContext,
        });
        // 通知渲染进程
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.webContents.send('opencode:chat-event', {
            sessionID: sessionId,
            kind: 'message',
            properties: { role: 'assistant', content: result.content },
          });
        }
        // 兜底路径同样回报 usage（若有），保证 Composer 不论走哪条链路都有 token 统计
        return { success: true, result: result.content, code: 'completed', usage: result.usage || null };
      } catch (e) {
        log.error('[OpenCode IPC] AgentCore fallback failed:', e.message);
        return { success: false, error: e.message, code: 'failed' };
      }
    }

    if (projectRoot) sessionProjects.set(sessionId, projectRoot);
    ensurePermWatcher();

    const root = projectRoot || sessionProjects.get(sessionId) || '';
    const modelCfg = model && model.providerID && model.modelID
      ? { providerID: model.providerID, modelID: model.modelID }
      : { providerID: CANCRI_PROVIDER_ID, modelID: CANCRI_MODEL_ID };
    // 前端显示名（如 cancri-fast-v1）与 opencode.json 注册的模型 key（DeepSeek-V4.1-Flash）不一致，
    // 这里按 provider 分别映射，确保传给 opencode 的 modelID 与注册的一致。
    let ocProviderID = modelCfg.providerID;
    let ocModelID = modelCfg.modelID;
    if (modelCfg.providerID === CANCRI_PROVIDER_ID) {
      ocModelID = CANCRI_MODEL_ID;                           // cancri-fast-v1 → DeepSeek-V4.1-Flash
    } else if (modelCfg.providerID === OPENROUTER_PROVIDER_ID) {
      // openrouter 前端 name 与 opencode.json modelID 一致，无需额外映射
      ocModelID = modelCfg.modelID;
    } else if (modelCfg.providerID === OPENCODE_ZEN_PROVIDER_ID) {
      // 前端用 opencode-zen（对齐 auth.json key），opencode 内部真实 providerID 是 'opencode'
      ocProviderID = OPENCODE_ZEN_OC_PROVIDER_ID;
    } else if (modelCfg.providerID === LOCAL_OLLAMA_PROVIDER_ID || modelCfg.providerID === 'ollama') {
      // 本地 Ollama: 前端/config 用 local-ollama, opencode.json 里注册的 providerID 是 'ollama';
      // 不映射会传不存在的 provider -> opencode HTTP 500 "Unexpected server error"
      ocProviderID = LOCAL_OLLAMA_OC_PROVIDER_ID;
    }
    const ocModelObj = { providerID: ocProviderID, modelID: ocModelID };
    const modelLabel = (payload && payload.displayName) || modelCfg.modelID;
    log.info(`[OpenCode IPC] chat model: ${JSON.stringify(ocModelObj)}`);

    // DeepSeek：发消息前确保 opencode.json 有官方 provider + 最新 key（界面保存过 key 才写；
    // opencode 进程不热加载配置，中途换的 key 下次启动 opencode 生效，本次失败会降级本地工具循环）
    if (ocProviderID === DEEPSEEK_PROVIDER_ID) {
      try { ensureDeepseekProvider(); } catch (e) { log.warn('[OpenCode IPC] ensureDeepseekProvider failed:', e.message); }
    }

    // 端点预检：不通则秒级失败，不占 30 分钟超时
    {
      const target = endpointForProvider(modelCfg.providerID);
      const ok = await endpointReachable(target, 6000);
      if (!ok) {
        const msg = `模型服务端点不可达（${target} TCP 连接失败，请检查网络或稍后重试）`;
        log.error('[OpenCode IPC]', msg);
        try { require('../services/health-store').recordEvent('opencode', 'fail', '模型端点预检不通：' + target); } catch (e) {}
        pushToRenderer(mainWindow, requestId, { kind: 'error', error: msg });
        return { success: false, error: msg, code: 'endpoint-down' };
      }
    }

    let sub = null;
    let progress = null;
    let parts = [], finish = '', toolFiles = [];
    // 活动信标：SSE 每次事件都 touch()，POST 的静默预算据此判断"上游还活着"。
    // 少了它，正在跑长工具的任务会被误判成挂死（见 ocFetch 顶部注释里的实测数据）。
    const activity = createActivityBeacon();
    log.info(`[OpenCode IPC] chat start: session=${sessionId} req=${requestId || '-'}`);
    try {
      // 进度心跳：TTFT / 已输出字符 / 已等待秒数。
      // 上游挂死时界面必须说清"在等什么、等了多久"，否则看起来就是卡死
      // （实测旧实现要等满 30 分钟总超时才报错）。
      progress = startProgressReporter({
        requestId,
        onTick: (p) => pushToRenderer(mainWindow, requestId, {
          kind: 'progress',
          waitedSec: p.waitedSec,
          ttftMs: p.ttftMs,
          chars: p.chars
        })
      });
      // 先订阅事件流，再发消息，避免漏掉早期事件
      try {
        sub = ocSubscribe((evt) => {
          try {
            const props = evt && evt.properties;
            if (!props || props.sessionID !== sessionId) return;
            // 任何本会话的事件都算"上游有活动"（含工具调用/完成，不只是文本增量）
            activity.touch();
            if (evt.type === 'message.part.delta' && props.field === 'text' && props.delta) {
              if (progress) progress.markFirstToken(String(props.delta).length);
              pushToRenderer(mainWindow, requestId, { kind: 'text-delta', delta: props.delta });
            } else if (evt.type === 'message.part.updated' && props.part && props.part.type === 'tool') {
              const st = props.part.state || {};
              pushToRenderer(mainWindow, requestId, {
                kind: 'tool',
                tool: props.part.tool,
                // partId：同一工具部件 pending/running/completed 跨事件的稳定标识，前端据此原地更新
                partId: props.part.id || '',
                status: st.status || '',
                // 输入放宽到 150k：write 的 content 需要完整透传给前端做“正在写入”动态展示
                input: st.input ? JSON.stringify(st.input).slice(0, 150000) : '',
                output: (st.output || (st.metadata && st.metadata.output) || '').toString().slice(0, 2000),
                title: props.part.title || st.title || ''
              });
            } else if (evt.type === 'message.updated' && props.info && props.info.time && props.info.time.completed) {
              pushToRenderer(mainWindow, requestId, { kind: 'done', finish: props.info.finish || '' });
            }
          } catch (e) { /* 转发失败不影响主流程 */ }
        });
      } catch (e) {
        log.warn('[OpenCode IPC] SSE subscribe failed, continuing without live events:', e.message);
      }

      // 上下文缓存：system 按 (root, model) 编译一次，避免每轮重建同一大段专家提示（省 token + 提速）。
      // 静态项目上下文（架构/文件清单/Moonwalker）由前端拼进首轮 text（会话历史持久保留），后续轮次省略——此处不重复注入。
      const sys = cachedSystemPrompt(root, modelLabel);

      // 动态智能上下文：技能进化 + 会话洞察 + 记忆，经 intelligence-context 聚合注入到 text 首部。
      // 注意：这部分是动态的（随用户消息/项目变化），故不放进 system（system 被缓存），拼到 text 层。
      const intelCtx = (() => {
        try {
          const ctx = intelligenceContext.buildContext(text, root);
          if (!ctx) return '';
          return `[INTELLIGENCE-CONTEXT]\n${ctx}\n[/INTELLIGENCE-CONTEXT]\n\n`;
        } catch (e) {
          log.warn('[OpenCode IPC] buildContext failed, skipping:', e.message);
          return '';
        }
      })();

      const effectiveText = intelCtx + text;
      // 两层预算：静默（默认 10 分钟，工具执行期间上游不发字节）+ 总时长兜底（沿用 timeoutMs，默认 30min）
      const data = await ocFetch('POST', `/session/${sessionId}/message`, {
        agent: 'build',
        model: ocModelObj,
        system: sys,
        parts: [{ type: 'text', text: effectiveText }]
      }, timeoutMs, {
        idleMs: Number(payload.idleTimeoutMs) > 0 ? Number(payload.idleTimeoutMs) : undefined,
        ttftMs: Number(payload.ttftTimeoutMs) > 0 ? Number(payload.ttftTimeoutMs) : undefined,
        totalMs: timeoutMs,
        activity
      });

      const parsed = parseRunResponse(data);
      parts = parsed.parts;
      finish = parsed.finish;
      toolFiles = extractToolFiles(parts);
      const runText = extractText(parts);
      if (parsed.runError) {
        // 模型侧运行错误（401/429/模型不存在…）：如实上抛，让前端降级本地工具循环而不是“未收到响应”
        const em = runErrorMessage(parsed.runError);
        log.error('[OpenCode IPC] model run failed:', em);
        pushToRenderer(mainWindow, requestId, { kind: 'error', error: em });
        return { success: false, error: em, code: 'failed' };
      }
      if (!runText.trim() && toolFiles.length === 0) {
        const em = `Agent 未返回任何内容（finish=${finish || '-'}）`;
        log.warn('[OpenCode IPC]', em);
        pushToRenderer(mainWindow, requestId, { kind: 'error', error: em });
        return { success: false, error: em, code: 'failed' };
      }
      // Token 用量：此前这条主路径把 usage 整个丢掉，Composer 模式因此既不显示也不累计
      const usage = extractUsage(data);
      if (usage) {
        pushToRenderer(mainWindow, requestId, { kind: 'usage', usage });
        log.info(`[OpenCode IPC] usage: in=${usage.prompt_tokens} out=${usage.completion_tokens}`);
      }
      pushToRenderer(mainWindow, requestId, { kind: 'done', finish });
      return {
        success: true,
        text: runText,
        toolFiles,
        finish,
        usage
      };
    } catch (e) {
      const isTimeout = /超时|timeout|中断/i.test(e.message || '');
      // 归因：ttft=首字都没来（上游不可用）· idle=已开始输出后卡住 · total=超总时长
      const reason = e.code || (/首字/.test(e.message || '') ? 'ttft' : (/中断/.test(e.message || '') ? 'idle' : (isTimeout ? 'total' : 'error')));
      log.error(`[OpenCode IPC] chat failed (${reason}):`, e.message);
      pushToRenderer(mainWindow, requestId, { kind: isTimeout ? 'timeout' : 'error', error: e.message, reason });
      // 超时：不 abort——服务端可能仍在执行，掐断会毁掉进行中的任务；由前端自动续跑/等待
      // 非超时错误：中断会话，避免悬挂
      if (!isTimeout) {
        try {
          ocFetch('POST', `/session/${sessionId}/abort`, undefined, 10000).catch(() => {});
        } catch (_) {}
      }
      return { success: false, error: e.message, code: isTimeout ? 'timeout' : 'failed', reason };
    } finally {
      if (progress) { try { progress.stop(); } catch (e) {} }
      if (sub) { try { sub.close(); } catch (e) {} }
      // 阶段4：会话结束回灌（完成/失败/超时都沉淀，供洞察/技能/记忆自进化）
      try {
        finalizeSessionEvolution({
          sessionId,
          projectRoot: root || projectRoot || sessionProjects.get(sessionId) || '',
          userText: text,
          parts,
          finish,
          result: { toolFiles },
        });
      } catch (e) {
        log.warn('[OpenCode IPC] session finalize failed:', e.message);
      }
    }
  });

  // 中断
  ipcMain.handle('opencode:abort', async (_event, sessionId) => {
    try {
      {
        const notReady = await requireServer();
        if (notReady) return notReady;
      }
      await ocFetch('POST', `/session/${sessionId}/abort`, undefined, 15000);
      return { success: true };
    } catch (e) {
      return { success: false, error: e.message };
    }
  });

  // 删除会话
  ipcMain.handle('opencode:delete-session', async (_event, sessionId) => {
    try {
      {
        const notReady = await requireServer();
        if (notReady) return notReady;
      }
      await ocFetch('DELETE', `/session/${sessionId}`, undefined, 15000);
      sessionProjects.delete(sessionId);
      return { success: true };
    } catch (e) {
      return { success: false, error: e.message };
    }
  });

  // 查询会话
  ipcMain.handle('opencode:get-session', async (_event, sessionId) => {
    try {
      {
        const notReady = await requireServer();
        if (notReady) return notReady;
      }
      const data = await ocFetch('GET', `/session/${sessionId}`, undefined, 15000);
      return { success: true, session: data };
    } catch (e) {
      return { success: false, error: e.message };
    }
  });

  // 取待回答问题（Agent question 工具阻塞时）
  ipcMain.handle('opencode:get-questions', async (_event, sessionId) => {
    try {
      {
        const notReady = await requireServer();
        if (notReady) return notReady;
      }
      const data = await ocFetch('GET', '/question', undefined, 15000);
      const list = Array.isArray(data) ? data : [];
      const mine = sessionId ? list.filter((q) => q.sessionID === sessionId) : list;
      return { success: true, questions: mine };
    } catch (e) {
      return { success: false, error: e.message };
    }
  });

  // 回答问题（结构：answers 为每题所选 label 数组的数组）
  ipcMain.handle('opencode:answer-question', async (_event, payload = {}) => {
    try {
      {
        const notReady = await requireServer();
        if (notReady) return notReady;
      }
      const { requestId, answers } = payload;
      if (!requestId || !answers) return { success: false, error: '缺少 requestId 或 answers' };
      await ocFetch('POST', `/question/${requestId}/reply`, { answers }, 15000);
      return { success: true };
    } catch (e) {
      log.error('[OpenCode IPC] answer-question failed:', e.message);
      return { success: false, error: e.message };
    }
  });

  // 任务验证闭环：对会话最近一次生成结果做"确定性校验(build/test/lint/code-intel)
  // + LLM 自评 + 有界自修"，输出结构化 PASS/REPAIR/FAIL 报告。
  // 复用 ai-engine 的 TaskVerificationLoop（此前只存在源码、未接入运行时，现经
  // @super-ide/ai-engine 导出并在此接入）。失败不抛，降级为"未验证"。
  ipcMain.handle('opencode:verify', async (_event, payload = {}) => {
    const { sessionId, projectRoot, criteria = '', maxRepairs = 1, stepTimeoutMs = 120000 } = payload;
    try {
      const notReady = await requireServer();
      if (notReady) return notReady;
      const sessionData = await ocFetch('GET', `/session/${sessionId}`, undefined, 15000);
      const msgs = (sessionData && sessionData.messages) || [];
      const lastAssistant = [...msgs].reverse().find(m => m.role === 'assistant' || m.role === 'agent');
      const task = (lastAssistant && (lastAssistant.content || lastAssistant.text)) || '';
      if (!task) return { success: false, error: '会话无可验证的助手输出', code: 'no-output' };

      const { TaskVerificationLoop } = require('@super-ide/ai-engine');
      const agentCore = require('../services/agent-core-local');
      const loop = new TaskVerificationLoop({
        projectRoot,
        aiGenerate: async (opts) => {
          const r = await agentCore.callLLM(opts.userPrompt || '', {
            system: opts.systemPrompt || '',
            model: opts.model,
            provider: opts.provider,
            temperature: opts.temperature,
            maxTokens: opts.maxTokens,
          });
          return (r && (r.content || r.text)) || '';
        },
        fetchDiagnostics: async (relPath) => {
          try {
            const resp = await fetch(`http://127.0.0.1:9099/code-intel?op=diagnostics&path=${encodeURIComponent(relPath)}`, { timeout: 8000 });
            const json = await resp.json();
            if (json && json.output && typeof json.output === 'string') {
              // code-intel 返回 "未发现诊断问题" 时视为无诊断
              if (/未发现诊断问题/.test(json.output)) return [];
              return [{ source: 'codeintel', severity: 'error', message: json.output }];
            }
            return [];
          } catch (e) { return []; }
        },
        maxRepairs,
        stepTimeoutMs,
        log: (m) => log.info('[VerifyLoop]', m),
      });

      const report = await loop.verify({ task, projectRoot, criteria, runDiagnostics: true });
      log.info(`[OpenCode IPC] verify done: verdict=${report.verdict} in ${report.durationMs}ms`);
      return { success: true, report };
    } catch (e) {
      log.error('[OpenCode IPC] verify failed:', e.message);
      return { success: false, error: e.message, code: 'failed' };
    }
  });

  // 阶段5：智能进化汇总统计（供面板轮询技能/会话/记忆/成功率）
  ipcMain.handle('opencode:get-stats', async (_event, payload = {}) => {
    try {
      const { projectRoot } = payload || {};
      const stats = intelligenceContext.getStats(projectRoot || '');
      return { success: true, stats };
    } catch (e) {
      log.error('[OpenCode IPC] get-stats failed:', e.message);
      return { success: false, error: e.message };
    }
  });
}
module.exports = {
  extractUsage, registerOpencodeIPC, createActivityBeacon,
  __setTestTarget, __internals: { ocFetch, DEFAULT_IDLE_MS, DEFAULT_TTFT_MS }
};
