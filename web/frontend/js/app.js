// === Web bridge: mock electronAPI over WebSocket ===
(function(){
  const wsUrl = (location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/ws';
  let ws = null, fullText = '';
  let outputCb = () => {}, chunkCb = () => {}, exitCb = () => {}, stderrCb = () => {};
  const DEFAULT_CFG = { backend:'openai', model:'DeepSeek-V4-Flash', base_url:'https://developer.amd.com.cn/radeon/api/v1', api_key:'', temperature:0.7, max_tokens:4096 };
  function connect(){
    ws = new WebSocket(wsUrl);
    ws.onclose = () => setTimeout(connect, 2000);
    ws.onmessage = (ev) => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'ready') { const b = document.getElementById('modelBadge'); if (b) b.textContent = m.config?.model || ''; }
      else if (m.type === 'stream') { fullText += m.token; outputCb(fullText); }
      else if (m.type === 'done') { outputCb(m.final_answer || fullText); exitCb(0); fullText = ''; }
      else if (m.type === 'error') { stderrCb((m.message || 'error').slice(0, 500)); }
      else if (m.type === 'tool_call') { handleToolCall(m); }
      else if (m.type === 'fs_status') { const b = document.getElementById('modelBadge'); if (b && m.enabled) b.textContent = (b.textContent || '') + ' · 📁' + m.folder; }
    };
  }
  connect();
  function send(o){ if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }

  // === File System Access API: 本地文件操作 ===
  let dirHandle = null;
  async function selectFolder() {
    if (!window.showDirectoryPicker) { alert('浏览器不支持 File System Access API，请用 Chrome/Edge 86+'); return; }
    try {
      dirHandle = await window.showDirectoryPicker({mode: 'readwrite'});
      const perm = await dirHandle.requestPermission({mode: 'readwrite'});
      if (perm !== 'granted') { alert('需要读写权限才能操作文件'); dirHandle = null; return; }
      send({type: 'fs_ready', folder: dirHandle.name});
      const badge = document.getElementById('modelBadge');
      if (badge) { badge.textContent = (badge.textContent || '').split(' · 📁')[0] + ' · 📁' + dirHandle.name; }
    } catch (e) { if (e.name !== 'AbortError') console.error('selectFolder:', e); }
  }
  async function handleToolCall(msg) {
    const {call_id, tool, args} = msg;
    if (tool !== 'file_ops' || !dirHandle) { send({type: 'tool_result', call_id, result: {success: false, error: '未选择本地文件夹'}}); return; }
    try { const r = await execFileOps(args); send({type: 'tool_result', call_id, result: r}); }
    catch (e) { send({type: 'tool_result', call_id, result: {success: false, error: String(e)}}); }
  }
  async function resolvePath(path) {
    if (!path || path === '.' || path === './') return dirHandle;
    const parts = path.replace(/^\.\//, '').split('/').filter(p => p);
    let cur = dirHandle;
    for (const part of parts) {
      try { cur = await cur.getDirectoryHandle(part); }
      catch { try { cur = await cur.getFileHandle(part); } catch { return null; } }
    }
    return cur;
  }
  async function execFileOps(args) {
    const {action, path = '', content, offset, limit} = args;
    if (action === 'list') return await listDir(path);
    if (action === 'read' || action === 'fast_read') return await readFile(path, offset, limit);
    if (action === 'write') return await writeFile(path, content);
    if (action === 'exists') return await existsPath(path);
    if (action === 'delete') return await deletePath(path);
    if (action === 'multi_read') {
      const paths = args.paths || [path];
      const results = [];
      for (const p of paths) { const r = await readFile(p); results.push(`--- ${p} ---\n${r.output || r.error}`); }
      return {success: true, output: results.join('\n\n')};
    }
    return {success: false, error: `action "${action}" 暂不支持本地执行`};
  }
  async function listDir(path) {
    const h = await resolvePath(path);
    if (!h) return {success: false, error: `路径不存在: ${path}`};
    const entries = [];
    for await (const [name, handle] of h.entries()) {
      let size = 0;
      if (handle.kind === 'file') { try { size = (await handle.getFile()).size; } catch {} }
      entries.push(`${handle.kind === 'directory' ? '📁' : '📄'} ${name}${size ? `  (${size} bytes)` : ''}`);
    }
    entries.sort();
    return {success: true, output: entries.join('\n') || '(空目录)', metadata: {count: entries.length}};
  }
  async function readFile(path, offset, limit) {
    const h = await resolvePath(path);
    if (!h || h.kind !== 'file') return {success: false, error: `文件不存在: ${path}`};
    const file = await h.getFile();
    const text = await file.text();
    const lines = text.split('\n');
    const start = offset || 0;
    const end = limit ? start + limit : lines.length;
    const sliced = lines.slice(start, end);
    const output = sliced.map((line, i) => `${start + i + 1}: ${line}`).join('\n');
    return {success: true, output, metadata: {total_lines: lines.length, shown: sliced.length}};
  }
  async function writeFile(path, content) {
    const parts = path.replace(/^\.\//, '').split('/').filter(p => p);
    const fileName = parts.pop();
    let dir = dirHandle;
    for (const part of parts) { dir = await dir.getDirectoryHandle(part, {create: true}); }
    const fh = await dir.getFileHandle(fileName, {create: true});
    const writable = await fh.createWritable();
    await writable.write(content || '');
    await writable.close();
    return {success: true, output: `已写入 ${path} (${(content || '').length} bytes)`};
  }
  async function existsPath(path) {
    const h = await resolvePath(path);
    return {success: true, output: h ? 'exists' : 'not found', metadata: {exists: !!h, kind: h?.kind}};
  }
  async function deletePath(path) {
    const parts = path.replace(/^\.\//, '').split('/').filter(p => p);
    const name = parts.pop();
    let dir = dirHandle;
    for (const part of parts) { dir = await dir.getDirectoryHandle(part); }
    await dir.removeEntry(name, {recursive: true});
    return {success: true, output: `已删除 ${path}`};
  }
  window.lvSelectFolder = selectFolder;
  // === end File System Access API ===

  window.electronAPI = {
    startAgent: async () => ({ success: true, pid: 1 }),
    stopAgent: async () => ({ success: true }),
    sendAgentMessage: async (text) => { send({ type: 'message', task: text }); return { success: true }; },
    onAgentOutput: (cb) => { outputCb = cb; },
    onAgentChunk: (cb) => { chunkCb = cb; },
    onAgentStderr: (cb) => { stderrCb = cb; },
    onAgentExit: (cb) => { exitCb = cb; },
    getModelConfig: async () => DEFAULT_CFG,
    setModelConfig: async (cfg) => { send({ type: 'config', config: cfg }); return { success: true }; },
    getAgentStatus: async () => ({ running: false }),
    listArtifacts: async () => ({ artifacts: [] }), openArtifact: async () => {}, revealArtifact: async () => {},
    listJobs: async () => ({ jobs: [] }), addJob: async () => ({ success: true }), toggleJob: async () => ({ success: true }), removeJob: async () => ({ success: true }),
    startTelegram: async () => ({ success: true }), stopTelegram: async () => ({ success: true }), getTelegramStatus: async () => ({}), getTelegramConfig: async () => ({}),
    onTelegramLog: () => {}, onTelegramExit: () => {},
    checkForUpdates: async () => ({}), getUpdateSettings: async () => ({}), setUpdateUrl: async () => ({ success: true }), skipUpdateVersion: async () => ({ success: true }),
    onMenuCheckUpdates: () => {},
  };
})();
// === end web bridge ===
const chatMessages = document.getElementById('chatMessages');
const chatArea = document.getElementById('chatArea');
const userInput = document.getElementById('userInput');
const btnSend = document.getElementById('btnSend');
const btnNew = document.getElementById('btnNew');
const btnSettings = document.getElementById('btnSettings');
const btnCloseSettings = document.getElementById('btnCloseSettings');
const settingsPanel = document.getElementById('settingsPanel');
const statusDot = document.getElementById('statusDot');
const btnSaveConfig = document.getElementById('btnSaveConfig');
const cfgStatus = document.getElementById('cfgStatus');

let agentRunning = false;
let fullOutput = '';
let thinkingEl = null;
let currentAgentEl = null;
let thinkingStart = 0;
let thinkingTimer = null;
let operationsEl = null;
let operationsLines = [];
let thinkingTimerEl = null;
let thinkingStepEl = null;

// --- ANSI / HTML helpers ---

function stripAnsi(str) {
  return str.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '').replace(/\x1b\]8;;.*?\x1b\\/g, '');
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function formatAgentContent(text) {
  let clean = stripAnsi(text);
  // Strip think tags that leak from model
  clean = clean.replace(/<\/?think[^>]*>/gi, '');
  clean = clean.trim();
  if (!clean) return '';
  // Extract code blocks first to avoid double-escaping
  const codeBlocks = [];
  clean = clean.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const idx = codeBlocks.length;
    codeBlocks.push({ lang: lang || '', code: code.trim() });
    return `\u0000CODEBLOCK_${idx}\u0000`;
  });
  // Escape remaining
  clean = escapeHtml(clean);
  // Restore code blocks with styling
  codeBlocks.forEach((b, i) => {
    const escapedCode = escapeHtml(b.code);
    const label = b.lang ? `<span class="code-lang">${escapeHtml(b.lang)}</span>` : '';
    clean = clean.replace(`\u0000CODEBLOCK_${i}\u0000`, `${label}<pre><code>${escapedCode}</code></pre>`);
  });
  // Inline code (after block restore)
  clean = clean.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold
  clean = clean.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  // Links
  clean = clean.replace(/(https?:\/\/[^\s<]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  // Stats line: ok · 10 steps · budget ... -> faint small
  clean = clean.replace(/(ok\s*·\s*\d+\s*steps[^<\n]*)/gi, '<span class="msg-stats">$1</span>');
  // Horizontal rule ---
  clean = clean.replace(/(\n|^)([-─]{3,})(\n|$)/g, '<span class="msg-stats">$2</span>');
  // Preserve newlines as <br> inside non-pre sections (white-space: pre-wrap handles it, so keep raw)
  return clean;
}

// --- Status / Operation extraction ---
function isStatusLine(line) {
  const t = line.trim();
  if (!t) return false;
  if (/<\/?think/i.test(t)) return true;
  if (/\[(STATUS|INFO|CONTENT|TOOL_|WARN|REFLECTION|CALL|RESULT|ERROR)\]/i.test(t)) return true;
  if (t.includes('->') && /web_search|bash_exec|grep|glob|file_ops|api_call/i.test(t)) return true;
  if (t.includes('<-') && /\[reflection\]|\[TOOL/i.test(t)) return true;
  if (t.match(/thinking\s*\(step\s*\d+\/\d+\)/i)) return true;
  if (t.match(/^\d{4}-\d{2}-\d{2}T.*\[/)) return true;
  if (t.match(/^memory\s+\d+/i)) return true;
  if (t.match(/^loops?\s+\d+/i)) return true;
  if (t.match(/^task:\s*/i)) return true;
  if (/^\s*(strategy|reflection|planning|reasoning|memory|context|self_correction|memskill|file_memory|sqlite_memory|pdf_generation|experience|harness|evolution|wiki_memory|skills|agents loaded|backend OpenAI)/i.test(t)) return true;
  if (t.match(/Lv agent|Lux Vita|Captain OS|Open-source AI|config loaded|agent ready|module.*status/i)) return true;
  if (t.match(/^\s*ok\s*·\s*\d+\s*steps/i)) return true;
  if (t.match(/^[-─]{3,}$/)) return true;
  return false;
}

function extractOperationsAndContent(text) {
  const rawLines = text.split('\n');
  const ops = [];
  const content = [];
  // Regex for inline timestamped bracket fragments: e.g. " 2026-09-02T11:58:28Z [TOOL_RESULT] ..."
  const inlineRe = /\d{4}-\d{2}-\d{2}T[^\n]*?\[(STATUS|INFO|CONTENT|TOOL_|WARN|REFLECTION|CALL|RESULT|ERROR)[^\]]*\][^\n]*/i;
  for (const line of rawLines) {
    const m = line.match(inlineRe);
    if (m && m.index !== undefined) {
      const idx = m.index;
      if (idx > 0) {
        const before = line.slice(0, idx).trim();
        const after = line.slice(idx).trim();
        if (before) content.push(before);
        if (after) ops.push(after);
        continue;
      }
      // Pure status line starting with timestamp
      ops.push(line);
      continue;
    }
    if (isStatusLine(line)) {
      ops.push(line);
    } else {
      content.push(line);
    }
  }
  return { ops, contentText: content.join('\n') };
}

// --- Noise filter for final answer (keep ops separate) ---
function filterNoise(text) {
  let lines = text.split('\n');
  lines = lines.filter(line => {
    const t = line.trim();
    if (!t) return true;
    if (/\[(STATUS|INFO|CONTENT|TOOL_|WARN|REFLECTION|CALL|RESULT|ERROR)\]/i.test(t)) return false;
    if (t.match(/^[·•]\s*[-─]+\s*(ok|done|finished|loop|step|token)/i)) return false;
    if (t.match(/^[·•]\s*\d+\s*(loops?|steps?|tokens?)/i)) return false;
    if (t.match(/^─+\s*(ok|done|finished)/i)) return false;
    if (t.match(/^\.+\s*ok\s*·/i)) return false;
    if (t.startsWith('Agent ready') || t.startsWith('Type your')) return false;
    if (t.match(/^Using\s+(backend|model)/i)) return false;
    if (t.match(/^(Model|Backend|Token|Config)\s*:/i)) return false;
    if (t.startsWith('>>>')) return false;
    if (t.startsWith('...')) return false;
    if (t.startsWith('Thought:') || t.startsWith('Action:')) return false;
    if (/^\[TOOL:|^Final Answer:\s*/i.test(t)) return false;
    if (/^\s*(strategy|reflection|planning|reasoning|memory|context|self_correction|memskill|file_memory|sqlite_memory|pdf_generation|experience|harness|evolution|wiki_memory|skills|agents loaded|backend OpenAI)/i.test(t)) return false;
    if (t.match(/Lv agent|Lux Vita|Captain OS|Open-source AI|config loaded/i)) return false;
    if (t.match(/^\s*ok\s*·\s*\d+\s*steps/i)) return false;
    if (t.match(/module\s+status/i)) return false;
    if (t.match(/^\s*-\s*strategy|reflection/i)) return false;
    return true;
  });
  let out = lines.join('\n');
  out = out.replace(/\n{3,}/g, '\n\n');
  return out;
}

// --- Chat rendering ---

function clearChat() {
  chatMessages.innerHTML = '';
  fullOutput = '';
  currentAgentEl = null;
}

function addUserMessage(text) {
  const welcome = chatMessages.querySelector('.welcome-msg');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = `
    <div class="msg-avatar">You</div>
    <div class="msg-body"><div class="msg-text">${escapeHtml(text)}</div></div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function createAgentMessage() {
  const welcome = chatMessages.querySelector('.welcome-msg');
  if (welcome) welcome.remove();

  const div = document.createElement('div');
  div.className = 'msg agent';
  div.innerHTML = `
    <div class="msg-avatar">LV</div>
    <div class="msg-body"><div class="msg-text"></div></div>
  `;
  chatMessages.appendChild(div);
  scrollToBottom();
  currentAgentEl = div.querySelector('.msg-text');
  return currentAgentEl;
}

function ensureCopyButton(el, rawText) {
  const body = el.closest('.msg-body');
  if (!body) return;
  let actions = body.querySelector('.msg-actions');
  if (!actions) {
    actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = `<button class="copy-btn" title="copy"><span class="copy-icon">⧉</span> <span class="copy-label">copy</span></button>`;
    body.appendChild(actions);
    const btn = actions.querySelector('.copy-btn');
    btn.addEventListener('click', async () => {
      const textToCopy = el.dataset.rawText || el.textContent || rawText || '';
      try {
        await navigator.clipboard.writeText(textToCopy);
        const label = btn.querySelector('.copy-label');
        const orig = label.textContent;
        label.textContent = 'copied';
        btn.classList.add('copied');
        setTimeout(() => { label.textContent = orig; btn.classList.remove('copied'); }, 1500);
      } catch (e) {
        // fallback: select
        const range = document.createRange();
        range.selectNodeContents(el);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand('copy');
        sel.removeAllRanges();
      }
    });
  }
  // Update raw text for copy
  el.dataset.rawText = rawText || el.textContent;
  // Ensure actions stays at bottom
  body.appendChild(actions);
}

function deduplicateParagraphs(text) {
  const paras = text.split(/\n{2,}/);
  const seen = new Set();
  const out = [];
  for (const p of paras) {
    const key = p.trim().slice(0, 120);
    if (key.length < 10) { out.push(p); continue; }
    if (seen.has(key)) continue;
    // Also check for large duplicate headings like "日常办公" appearing twice verbatim
    if (p.includes('日常办公') && out.some(q => q.includes('日常办公') && q.includes('技术开发'))) continue;
    seen.add(key);
    out.push(p);
  }
  return out.join('\n\n');
}

function updateAgentMessage(el, text) {
  let filtered = filterNoise(text);
  filtered = deduplicateParagraphs(filtered);
  if (filtered.trim()) {
    const prev = el.dataset.filteredHash || '';
    const curHash = filtered.slice(0, 200) + '|' + filtered.length;
    if (prev && el.innerHTML && filtered.includes(prev.slice(0, 80)) && Math.abs(filtered.length - (parseInt(prev.split('|')[1])||0)) < 50) {
      // Likely same content replay, skip
    }
    el.innerHTML = formatAgentContent(filtered);
    el.dataset.filteredHash = curHash;
    // Force scroll to newest position (always, not only if wasAtBottom)
    scrollToBottom(true);
    // Double-rAF to ensure layout is flushed before scrolling long content
    requestAnimationFrame(() => requestAnimationFrame(() => scrollToBottom(true)));
    setTimeout(() => scrollToBottom(true), 50);
    el.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', (e) => {
        e.preventDefault();
        window.open(a.href, '_blank');
      });
    });
    ensureCopyButton(el, filtered);
  } else if (!filtered.trim() && text.trim()) {
    el.textContent = text.trim().slice(0, 8000);
    scrollToBottom(true);
    setTimeout(() => scrollToBottom(true), 50);
    ensureCopyButton(el, text);
  }
}

function isAtBottom() {
  return chatArea.scrollTop + chatArea.clientHeight >= chatArea.scrollHeight - 32;
}

function scrollToBottom(force = false) {
  // Always push latest to visible bottom — newest at latest position
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
    // Also scroll operations log to bottom if exists
    if (operationsEl) operationsEl.scrollTop = operationsEl.scrollHeight;
  });
}

function appendOperations(newOps) {
  // Append only to the current query's thinking block (created at sendMessage). Do not create new after Done.
  if (!thinkingEl || !operationsEl) {
    // Try to find last thinking block for current query (if still at bottom)
    const lastBlock = chatMessages.querySelector('.thinking-block:last-of-type');
    if (lastBlock && lastBlock === chatMessages.lastElementChild) {
      thinkingEl = lastBlock;
      operationsEl = lastBlock.querySelector('.operations-log');
      thinkingTimerEl = lastBlock.querySelector('.thinking-timer');
      thinkingStepEl = lastBlock.querySelector('.thinking-step');
    } else {
      return;
    }
  }
  for (const line of newOps) {
    const t = line.trim();
    if (!t) continue;
    // Deduplicate consecutive same lines
    if (operationsLines.length && operationsLines[operationsLines.length - 1] === t) continue;
    operationsLines.push(t);
    // Keep last 80 lines
    if (operationsLines.length > 80) operationsLines.shift();
    const div = document.createElement('div');
    div.className = 'op-line';
    // Color by type
    if (t.includes('thinking')) div.classList.add('op-thinking');
    else if (t.includes('memory')) div.classList.add('op-memory');
    else if (t.includes('loops')) div.classList.add('op-loops');
    else if (t.includes('task:')) div.classList.add('op-task');
    div.textContent = t;
    operationsEl.appendChild(div);
  }
  // Keep only last 80 DOM nodes
  while (operationsEl.children.length > 80) operationsEl.removeChild(operationsEl.firstChild);
  // Continuous scroll
  operationsEl.scrollTop = operationsEl.scrollHeight;
  scrollToBottom();
}

function showThinking() {
  if (thinkingEl) return;
  operationsLines = [];
  const div = document.createElement('div');
  div.className = 'msg agent thinking-block';
  div.innerHTML = `
    <div class="msg-avatar">LV</div>
    <div class="msg-body" style="flex:1; min-width:0;">
      <div class="thinking-indicator">
        <div class="thinking-spinner"></div>
        <span class="thinking-label">Thinking</span>
        <span class="thinking-timer">0.0s</span>
        <span class="thinking-step"></span>
      </div>
      <div class="operations-log"></div>
    </div>
  `;
  chatMessages.appendChild(div);
  thinkingEl = div;
  operationsEl = div.querySelector('.operations-log');
  thinkingTimerEl = div.querySelector('.thinking-timer');
  thinkingStepEl = div.querySelector('.thinking-step');
  thinkingStart = Date.now();
  // Countdown timer: update every 100ms — scoped to current block only
  if (thinkingTimer) clearInterval(thinkingTimer);
  thinkingTimer = setInterval(() => {
    if (!thinkingTimerEl || !thinkingEl) return;
    const secs = ((Date.now() - thinkingStart) / 1000).toFixed(1);
    thinkingTimerEl.textContent = `${secs}s`;
    const last = operationsLines[operationsLines.length - 1] || '';
    const m = last.match(/step\s*(\d+)\s*\/\s*(\d+)/i);
    if (m && thinkingStepEl) {
      thinkingStepEl.textContent = `· step ${m[1]}/${m[2]}`;
      thinkingStepEl.style.display = '';
    }
  }, 100);
  scrollToBottom(true);
}

function hideThinking() {
  if (thinkingTimer) { clearInterval(thinkingTimer); thinkingTimer = null; }
  if (thinkingEl) {
    const spinner = thinkingEl.querySelector('.thinking-spinner');
    if (spinner) { spinner.style.animation = 'none'; spinner.style.opacity = '0.5'; }
    const label = thinkingEl.querySelector('.thinking-label');
    if (label) label.textContent = 'Done';
    if (thinkingTimerEl) {
      const secs = ((Date.now() - thinkingStart) / 1000).toFixed(1);
      thinkingTimerEl.textContent = `${secs}s`;
    }
    thinkingEl.classList.add('done');
    // Freeze current block, clear scoped refs so next query creates fresh block at newest position
    thinkingTimerEl = null;
    thinkingStepEl = null;
    thinkingEl = null;
    operationsEl = null;
  }
}

function _legacyScrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

// --- Status ---

function setStatus(running, pid) {
  agentRunning = running;
  statusDot.className = running ? 'status-dot online' : 'status-dot offline';
  statusDot.title = running ? `Online${pid ? ' (pid ' + pid + ')' : ''}` : 'Offline';
  // Always allow typing — only Send is gated by running state
  userInput.disabled = false;
  userInput.readOnly = false;
  btnSend.disabled = !running || !userInput.value.trim();
  btnNew.textContent = running ? '↻' : '▶';
  btnNew.title = running ? 'Restart Agent' : 'Start Agent';
  if (!running) userInput.placeholder = 'Agent offline — click ▶ to start, then press Enter to send';
  else userInput.placeholder = 'Type a message... (Enter to send, Shift+Enter newline)';
  // Ensure focus so user can type immediately
  if (!running) setTimeout(() => userInput.focus(), 100);
}

function updateSendEnabled() {
  btnSend.disabled = !agentRunning || !userInput.value.trim();
}

// --- Settings ---

function maskApiKey(key) {
  if (!key || key.length < 8) return key || '';
  if (key.startsWith('nvapi-') || key.startsWith('sk-')) return key.slice(0, 7) + '••••••••' + key.slice(-4);
  return '••••••••';
}

async function loadSettings() {
  const cfg = await window.electronAPI.getModelConfig();
  document.getElementById('cfgBackend').value = cfg.backend || 'openai';
  document.getElementById('cfgModel').value = cfg.model || '';
  document.getElementById('cfgBaseUrl').value = cfg.baseUrl || '';
  // Show masked key by default, real key stored in dataset
  const keyInput = document.getElementById('cfgApiKey');
  keyInput.dataset.realKey = cfg.apiKey || '';
  keyInput.value = cfg.apiKey ? maskApiKey(cfg.apiKey) : '';
  keyInput.placeholder = 'sk-... / nvapi-...';
  document.getElementById('cfgTemp').value = cfg.temperature ?? 0.7;
  document.getElementById('cfgMaxTokens').value = cfg.maxTokens ?? 4096;
  // Show friendly model name in topbar (DeepSeek-V4-Flash -> cancri fast)
  try {
    const badge = document.getElementById('modelBadge');
    if (badge) {
      const name = cfg.displayName || cfg.model || '';
      badge.textContent = name ? `${name} · Harness · ReAct` : 'Harness · ReAct';
      badge.title = cfg.model ? `model: ${cfg.model}` : '';
    }
  } catch {}
  // Show config path hint
  if (cfg.configPath) {
    cfgStatus.textContent = `Config: ${cfg.configPath}`;
    setTimeout(() => { if (cfgStatus.textContent.startsWith('Config:')) cfgStatus.textContent = ''; }, 4000);
  }
}

async function saveSettings() {
  const keyInput = document.getElementById('cfgApiKey');
  let apiKey = keyInput.value;
  // If user left masked placeholder, keep real key
  if (apiKey.includes('•') || apiKey === maskApiKey(keyInput.dataset.realKey)) {
    apiKey = keyInput.dataset.realKey;
  }
  const cfg = {
    backend: document.getElementById('cfgBackend').value,
    model: document.getElementById('cfgModel').value,
    baseUrl: document.getElementById('cfgBaseUrl').value,
    apiKey: apiKey,
    temperature: parseFloat(document.getElementById('cfgTemp').value) || 0.7,
    maxTokens: parseInt(document.getElementById('cfgMaxTokens').value) || 4096,
  };
  btnSaveConfig.disabled = true;
  btnSaveConfig.textContent = 'Saving...';
  const result = await window.electronAPI.setModelConfig(cfg);
  btnSaveConfig.disabled = false;
  btnSaveConfig.textContent = 'Save';
  if (result.success) {
    cfgStatus.textContent = 'Saved to ' + (result.path || 'config.yaml') + ' — restart agent to apply';
    cfgStatus.style.color = '#4ade80';
    keyInput.dataset.realKey = apiKey;
    keyInput.value = apiKey ? maskApiKey(apiKey) : '';
  } else {
    cfgStatus.textContent = 'Error: ' + result.error;
    cfgStatus.style.color = '#ff4d4d';
  }
  setTimeout(() => { cfgStatus.textContent = ''; }, 3500);
}

// --- Event handlers ---

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;
  if (!agentRunning) {
    const toast = document.createElement('div');
    toast.style.cssText = 'position:fixed;bottom:72px;left:50%;transform:translateX(-50%);background:#2a2a1a;border:1px solid #5a3a00;color:#ffcc66;padding:8px 14px;border-radius:8px;font-size:12px;z-index:9999;';
    toast.textContent = 'Agent offline — click ▶ to start, then retry.';
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2500);
    // Try auto-start
    try { const r = await window.electronAPI.startAgent(); if (r.success) setStatus(true, r.pid); } catch {}
    return;
  }

  addUserMessage(text);
  userInput.value = '';
  userInput.style.height = 'auto';
  updateSendEnabled();

  showThinking();
  fullOutput = '';
  currentAgentEl = null;
  seenOps.clear();
  operationsLines = [];
  const res = await window.electronAPI.sendAgentMessage(text);
  if (res && res.error) {
    hideThinking();
    const el = createAgentMessage();
    el.innerHTML = `<span style="color:#ff4d4d">${escapeHtml(res.error)}</span>`;
  }
}

btnSend.addEventListener('click', sendMessage);

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 120) + 'px';
  updateSendEnabled();
});

btnNew.addEventListener('click', async () => {
  btnNew.disabled = true;
  if (agentRunning) {
    await window.electronAPI.stopAgent();
    setStatus(false);
    await new Promise(r => setTimeout(r, 400));
  }
  // Don't clear chat on restart — keep history, just add divider
  if (agentRunning) {
    const divider = document.createElement('div');
    divider.style.cssText = 'text-align:center;color:#555;font-size:11px;padding:4px 0;border-top:1px dashed #2a2a2a;margin:8px 0;';
    divider.textContent = '— Session restarted —';
    chatMessages.appendChild(divider);
  } else {
    // First start: clear welcome
    const welcome = chatMessages.querySelector('.welcome-msg');
    if (welcome && chatMessages.children.length === 1) {
      // Keep welcome until first message
    }
  }
  cfgStatus.textContent = 'Starting agent...';
  const result = await window.electronAPI.startAgent();
  btnNew.disabled = false;
  if (result.success || result.alreadyRunning) {
    setStatus(true, result.pid);
    cfgStatus.textContent = '';
    if (!result.alreadyRunning) {
      // Show ready hint
      if (!fullOutput) {
        // Wait for output
      }
    }
  } else {
    cfgStatus.textContent = 'Failed: ' + (result.error || 'unknown');
    cfgStatus.style.color = '#ff4d4d';
    setStatus(false);
    setTimeout(() => cfgStatus.textContent = '', 4000);
  }
});

btnSettings.addEventListener('click', () => {
  settingsPanel.classList.toggle('hidden');
  if (!settingsPanel.classList.contains('hidden')) { loadSettings(); loadUpdateSettings(); }
});

document.getElementById('btnFolder')?.addEventListener('click', () => {
  if (window.lvSelectFolder) window.lvSelectFolder();
});

btnCloseSettings.addEventListener('click', () => {
  settingsPanel.classList.add('hidden');
});

btnSaveConfig.addEventListener('click', saveSettings);

// Focus API key on click to reveal
document.getElementById('cfgApiKey').addEventListener('focus', function() {
  if (this.value.includes('•') && this.dataset.realKey) {
    this.value = this.dataset.realKey;
    this.select();
  }
});
document.getElementById('cfgApiKey').addEventListener('blur', function() {
  if (this.value && !this.value.includes('•') && this.value.length > 8) {
    this.dataset.realKey = this.value;
    this.value = maskApiKey(this.value);
  }
});

// --- IPC listeners (full + chunk + stderr) ---
// Deduplicate ops across full-buffer replays
const seenOps = new Set();

window.electronAPI.onAgentOutput((data) => {
  fullOutput = data;
  const { ops, contentText } = extractOperationsAndContent(data);
  // 1) Operations: append only unseen lines to current query's thinking block
  if (ops.length) {
    const newOps = ops.filter(line => {
      const t = line.trim();
      if (!t || seenOps.has(t)) return false;
      seenOps.add(t);
      return true;
    });
    if (newOps.length) {
      // If final answer already started for this query, ignore late ops (they belong to previous thinking)
      if (currentAgentEl && currentAgentEl.textContent.trim()) {
        // Check if currentAgentEl is the last message and has content -> final answer in progress, ignore ops
        const lastIsCurrent = chatMessages.lastElementChild?.contains(currentAgentEl);
        if (lastIsCurrent) {
          // ignore ops that arrive after content
        } else {
          appendOperations(newOps);
        }
      } else {
        appendOperations(newOps);
      }
    }
  }
  // 2) Final content: show at newest position, freeze thinking timer
  const filtered = filterNoise(contentText);
  if (filtered.trim()) {
    // First content chunk ends thinking countdown
    if (thinkingEl) hideThinking();
    let target = currentAgentEl;
    // Always create new bubble at bottom for this query's answer (newest position)
    if (!target || target.textContent.trim() === '' || target.parentElement !== chatMessages.lastElementChild?.querySelector('.msg-text')?.parentElement) {
      // If last element is not our currentAgentEl, create new
      const lastAgentText = chatMessages.querySelector('.msg.agent:last-child .msg-text');
      const isLastOurTarget = lastAgentText === target;
      if (!target || !isLastOurTarget) {
        target = createAgentMessage();
        currentAgentEl = target;
      }
    }
    if (!target) {
      target = createAgentMessage();
      currentAgentEl = target;
    }
    updateAgentMessage(target, contentText);
  } else if (ops.length && !filtered.trim()) {
    // Only ops, no content yet — keep thinking visible and scroll to newest
    scrollToBottom(true);
  }
});

// Incremental chunk (optional, for smoother typewriter if renderer supports)
window.electronAPI.onAgentChunk((_chunk) => {
  // We use fullOutput path above; chunk is for future optimization
});

window.electronAPI.onAgentStderr((msg) => {
  // Show stderr as toast
  const toast = document.createElement('div');
  toast.style.cssText = 'position:fixed;bottom:72px;right:16px;background:#2a1a1a;border:1px solid #5a2a2a;color:#ff9999;padding:8px 12px;border-radius:8px;font-size:12px;max-width:360px;z-index:9999;';
  toast.textContent = msg.slice(0, 500);
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
});

window.electronAPI.onAgentExit((code) => {
  setStatus(false);
  hideThinking();
  if (fullOutput) {
    let lastAgent = currentAgentEl || chatMessages.querySelector('.msg.agent:last-child .msg-text');
    if (lastAgent) updateAgentMessage(lastAgent, fullOutput);
  }
  // Auto-toast exit
  if (code !== 0 && code !== null) {
    const el = createAgentMessage();
    el.innerHTML = `<span style="color:#ff9a9a">Agent exited (code ${code}). Click ▶ to restart.</span>`;
  }
});

// --- Splash hide ---
function hideSplash() {
  const s = document.getElementById('splash');
  if (s) { s.classList.add('hide'); setTimeout(() => s.remove(), 800); }
}
setTimeout(hideSplash, 1600);
document.getElementById('splash')?.addEventListener('click', hideSplash);

// --- Theme (default light) ---
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('lv-theme', theme);
  const btn = document.getElementById('btnTheme');
  if (btn) btn.textContent = theme === 'light' ? '☾' : '◐';
  if (btn) btn.title = theme === 'light' ? '切换到深色' : '切换到浅色 (当前浅色)';
}
const savedTheme = localStorage.getItem('lv-theme') || 'light';
applyTheme(savedTheme);
document.getElementById('btnTheme')?.addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme') || 'light';
  applyTheme(cur === 'light' ? 'dark' : 'light');
});

// Sidebar interactions
document.getElementById('btnNewSession')?.addEventListener('click', () => {
  // Return to chat view + clear chat → new session
  if (typeof showView === 'function') showView('chat');
  const wasActive = document.querySelector('.session-item.active');
  if (wasActive) wasActive.classList.remove('active');
  chatMessages.innerHTML = '<div class="welcome-msg"><div class="welcome-icon">LV</div><p class="welcome-title">LV Agent</p><p class="welcome-sub">Deep thinking, real tools. 开始新对话。</p></div>';
  fullOutput = ''; currentAgentEl = null; operationsLines = []; seenOps.clear();
  document.getElementById('btnNew')?.click();
});
// Nav items -> real view router
const VIEWS = ['capabilities', 'messaging', 'artifacts', 'scheduled'];
let activeView = 'chat';

function showView(view) {
  if (!VIEWS.includes(view)) view = 'chat';
  // Hide all view panels + chat view
  VIEWS.forEach(v => {
    const el = document.getElementById('view' + v.charAt(0).toUpperCase() + v.slice(1));
    if (el) el.classList.add('hidden');
  });
  const isChat = view === 'chat';
  document.querySelectorAll('.chat-view').forEach(el => el.classList.toggle('hidden', !isChat));
  if (!isChat) {
    const panel = document.getElementById('view' + view.charAt(0).toUpperCase() + view.slice(1));
    if (panel) panel.classList.remove('hidden');
  }
  activeView = view;
  // Sync sidebar active state
  document.querySelectorAll('.sidebar-nav .nav-item[data-view]').forEach(n => {
    n.classList.toggle('active', n.dataset.view === view);
  });
  // Render on show
  if (view === 'capabilities') renderCapabilities();
  else if (view === 'messaging') renderMessaging();
  else if (view === 'artifacts') renderArtifacts();
  else if (view === 'scheduled') renderScheduled();
}

document.querySelectorAll('.sidebar-nav .nav-item[data-view]').forEach(btn => {
  btn.addEventListener('click', () => {
    const view = btn.dataset.view;
    // Toggle: clicking active nav item returns to chat
    if (activeView === view) { showView('chat'); }
    else { showView(view); }
  });
});
// Back buttons
document.querySelectorAll('.view-back').forEach(btn => {
  btn.addEventListener('click', () => showView(btn.dataset.back || 'chat'));
});

// --- Capabilities view (static-but-accurate manifest) ---
function renderCapabilities() {
  const body = document.getElementById('capabilitiesBody');
  if (!body) return;
  if (body.dataset.rendered) return; // static; render once
  body.dataset.rendered = '1';

  const groups = [
    {
      title: 'Reasoning',
      items: [
        { icon: '◈', name: 'Chain-of-Thought', tag: 'CoT', tagClass: 'purple', desc: 'Linear step-by-step decomposition with self-check.' },
        { icon: '◈', name: 'ReAct', tag: 'core', tagClass: 'green', desc: 'Reason + Act loop: interleaved thought, action, observation.' },
        { icon: '◈', name: 'Verification', tag: 'SC', tagClass: 'amber', desc: 'Self-Consistency voting; MCTS planner in progress.' },
      ],
    },
    {
      title: 'Planning',
      items: [
        { icon: '⇄', name: 'Sequential', desc: 'Strict ordered execution of subtasks.' },
        { icon: '⇄', name: 'Parallel', desc: 'Independent subtasks run concurrently.' },
        { icon: '⇄', name: 'Hierarchical', desc: 'Nested subagents with parent oversight.' },
        { icon: '⇄', name: 'Adaptive', desc: 'Strategy chosen per-task by intent classifier.' },
      ],
    },
    {
      title: 'Harness runtime',
      items: [
        { icon: '◐', name: 'Event sourcing', desc: 'Execution trace is append-only; replayable.' },
        { icon: '◐', name: 'Session persistence', desc: 'SQLite store; restore via /sessions.' },
        { icon: '◐', name: 'Budget control', desc: 'Token + wall-clock double limit.' },
        { icon: '◐', name: 'Tool confirmation', desc: 'Dangerous ops require user approval.' },
        { icon: '◐', name: 'Checkpoints', desc: 'Resume interrupted runs.' },
        { icon: '◐', name: 'Hot-plug', desc: 'Live module swap with rollback.' },
      ],
    },
    {
      title: 'Memory',
      items: [
        { icon: '◉', name: 'Knowledge graph', desc: 'Entity-relation long-term store.' },
        { icon: '◉', name: 'Experience memory', desc: 'Cross-session vector similarity recall.' },
        { icon: '◉', name: 'Memory skills', desc: '/learn + /memskill extract reusable tactics.' },
        { icon: '◉', name: 'Context compression', desc: 'Auto-summarize to 512-token budget.' },
      ],
    },
  ];
  const tools = [
    'web_search', 'web_fetcher', 'file_ops', 'grep', 'glob', 'bash_exec', 'python_exec',
    'calculator', 'github_search', 'git_ops', 'pdf_tool', 'weather', 'api_call', 'database',
    'telegram_bot', 'playwright_browser', 'turing_machine', 'process_manager', 'project_context',
    'discovery', 'mcp_client',
  ];

  let html = '';
  for (const g of groups) {
    html += `<div class="view-section"><div class="view-section-title">${g.title}</div><div class="cap-grid">`;
    for (const it of g.items) {
      const tag = it.tag ? `<span class="cap-tag ${it.tagClass || ''}">${it.tag}</span>` : '';
      html += `<div class="cap-card"><div class="cap-card-head"><div class="cap-icon">${it.icon}</div><div class="cap-name">${it.name}</div>${tag}</div><div class="cap-desc">${it.desc}</div></div>`;
    }
    html += `</div></div>`;
  }
  html += `<div class="view-section"><div class="view-section-title">Tools (${tools.length})</div><div class="cap-list">`;
  for (const t of tools) html += `<span class="cap-chip">${t}</span>`;
  html += `</div></div>`;
  body.innerHTML = html;
}

// --- Messaging view (Telegram) ---
let telegramLogLines = [];
function maskToken(t) {
  if (!t || t.length < 12) return t ? '••••••' : '';
  return t.slice(0, 8) + '••••••••' + t.slice(-4);
}
async function renderMessaging() {
  const body = document.getElementById('messagingBody');
  if (!body) return;
  let cfg = { enabled: false, hasToken: false, botToken: '', polling: true, allowedUserIds: [] };
  let status = { running: false };
  try { cfg = await window.electronAPI.getTelegramConfig(); } catch {}
  try { status = await window.electronAPI.getTelegramStatus(); } catch {}
  const running = !!status.running;
  const dotClass = running ? 'on' : (cfg.hasToken ? 'off' : '');
  const statusText = running ? 'Running' : (cfg.hasToken ? 'Stopped' : 'Not configured');
  body.innerHTML = `
    <div class="msg-status-card">
      <div class="msg-status-row"><span class="msg-dot ${dotClass}"></span><span class="value">${statusText}${status.pid ? ' (pid ' + status.pid + ')' : ''}</span></div>
      <div class="msg-status-row"><span class="label">Bot token</span><span class="value">${cfg.hasToken ? maskToken(cfg.botToken) : '— not set —'}</span></div>
      <div class="msg-status-row"><span class="label">Enabled in config</span><span class="value">${cfg.enabled ? 'yes' : 'no'}</span></div>
      <div class="msg-status-row"><span class="label">Polling</span><span class="value">${cfg.polling ? 'yes' : 'no'}</span></div>
      <div class="msg-status-row"><span class="label">Allowed users</span><span class="value">${(cfg.allowedUserIds || []).length ? cfg.allowedUserIds.join(', ') : 'any'}</span></div>
      <div class="msg-actions-row">
        <button id="btnTgStart" class="msg-btn" ${running || !cfg.hasToken ? 'disabled' : ''}>Start bot</button>
        <button id="btnTgStop" class="msg-btn secondary" ${running ? '' : 'disabled'}>Stop bot</button>
      </div>
    </div>
    <div class="msg-hint">
      Set the token via <code>TELEGRAM_BOT_TOKEN</code> env var or <code>telegram.bot_token</code> in config (Settings → Save).
      The bot bridges agent replies to your Telegram chat.
    </div>
    <div class="msg-log" id="telegramLog"></div>
  `;
  const logEl = document.getElementById('telegramLog');
  logEl.textContent = telegramLogLines.join('\n');
  logEl.scrollTop = logEl.scrollHeight;
  document.getElementById('btnTgStart')?.addEventListener('click', async () => {
    const r = await window.electronAPI.startTelegram();
    if (r.success || r.alreadyRunning) { renderMessaging(); }
    else { telegramLogLines.push('[error] ' + (r.error || 'failed to start')); renderMessaging(); }
  });
  document.getElementById('btnTgStop')?.addEventListener('click', async () => {
    await window.electronAPI.stopTelegram();
    renderMessaging();
  });
}
window.electronAPI.onTelegramLog((data) => {
  const prefix = data.stream === 'stderr' ? '[err] ' : '';
  telegramLogLines.push(prefix + (data.text || '').trimEnd());
  if (telegramLogLines.length > 200) telegramLogLines.shift();
  const logEl = document.getElementById('telegramLog');
  if (logEl) {
    logEl.textContent = telegramLogLines.join('\n');
    logEl.scrollTop = logEl.scrollHeight;
  }
});
window.electronAPI.onTelegramExit((_code) => {
  if (activeView === 'messaging') renderMessaging();
});

// --- Artifacts view ---
let artifactsCache = [];
function fmtSize(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1024 / 1024).toFixed(1) + ' MB';
}
function fmtTime(ms) {
  const d = new Date(ms);
  const now = Date.now();
  const diff = now - ms;
  if (diff < 60000) return 'just now';
  if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
  if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
  if (diff < 604800000) return Math.floor(diff / 86400000) + 'd ago';
  return d.toLocaleDateString();
}
async function renderArtifacts() {
  const body = document.getElementById('artifactsBody');
  if (!body) return;
  body.innerHTML = `<div class="art-toolbar"><input class="art-search" id="artSearch" placeholder="Filter artifacts by name…" /><span class="art-count" id="artCount">loading…</span></div><div class="art-list" id="artList"></div>`;
  const listEl = document.getElementById('artList');
  const countEl = document.getElementById('artCount');
  let result = { success: false, items: [] };
  try { result = await window.electronAPI.listArtifacts(); } catch (e) { result.error = e.message; }
  if (!result.success) {
    listEl.innerHTML = `<div class="empty-state"><div class="empty-icon">⚠</div><div class="empty-title">Couldn't scan artifacts</div><div class="empty-sub">${escapeHtml(result.error || 'unknown error')}</div></div>`;
    countEl.textContent = '';
    return;
  }
  artifactsCache = result.items || [];
  const draw = (filter) => {
    const items = artifactsCache.filter(a => !filter || a.name.toLowerCase().includes(filter.toLowerCase()));
    countEl.textContent = `${items.length} of ${artifactsCache.length}`;
    if (!items.length) {
      listEl.innerHTML = `<div class="empty-state"><div class="empty-icon">▭</div><div class="empty-title">${artifactsCache.length ? 'No matches' : 'No artifacts yet'}</div><div class="empty-sub">${artifactsCache.length ? 'Try a different filter.' : 'Generated reports, PDFs, and saved files will appear here.'}</div></div>`;
      return;
    }
    listEl.innerHTML = items.map(a => `
      <div class="art-item" data-path="${escapeHtml(a.path)}">
        <div class="art-ext ${a.ext}">${escapeHtml(a.ext)}</div>
        <div class="art-info">
          <div class="art-name">${escapeHtml(a.name)}</div>
          <div class="art-meta">${escapeHtml(a.dir)} · ${fmtSize(a.size)} · ${fmtTime(a.mtime)}</div>
        </div>
        <button class="art-open" title="Reveal in Finder" data-reveal="${escapeHtml(a.path)}">⌕</button>
        <button class="art-open" title="Open" data-open="${escapeHtml(a.path)}">↗</button>
      </div>
    `).join('');
  };
  draw('');
  document.getElementById('artSearch').addEventListener('input', (e) => draw(e.target.value));
  listEl.addEventListener('click', async (e) => {
    const openPath = e.target.dataset.open;
    const revealPath = e.target.dataset.reveal;
    const itemPath = e.target.closest('.art-item')?.dataset.path;
    if (openPath) { e.stopPropagation(); await window.electronAPI.openArtifact(openPath); }
    else if (revealPath) { e.stopPropagation(); await window.electronAPI.revealArtifact(revealPath); }
    else if (itemPath) { await window.electronAPI.openArtifact(itemPath); }
  });
}
document.getElementById('btnRefreshArtifacts')?.addEventListener('click', async (e) => {
  const btn = e.currentTarget;
  btn.classList.add('spinning');
  await renderArtifacts();
  setTimeout(() => btn.classList.remove('spinning'), 400);
});

// --- Scheduled jobs view ---
async function renderScheduled() {
  const body = document.getElementById('scheduledBody');
  if (!body) return;
  let jobs = [];
  try { jobs = await window.electronAPI.listJobs(); } catch {}
  body.innerHTML = `
    <div class="view-section">
      <div class="view-section-title">New job</div>
      <div class="job-form">
        <div class="form-row"><input id="jobName" type="text" placeholder="Job name" /></div>
        <div class="form-row"><input id="jobSchedule" type="text" placeholder="Cron: min hour day month weekday  (e.g. 0 9 * * * = daily 9am)" /></div>
        <div class="form-row"><textarea id="jobPrompt" placeholder="Prompt to run…"></textarea></div>
        <div class="msg-actions-row"><button id="btnJobAdd" class="msg-btn">Add job</button></div>
      </div>
    </div>
    <div class="view-section">
      <div class="view-section-title">Existing jobs (${jobs.length})</div>
      <div class="job-list" id="jobList"></div>
    </div>
  `;
  const drawJobs = (list) => {
    const listEl = document.getElementById('jobList');
    if (!list.length) {
      listEl.innerHTML = `<div class="empty-state"><div class="empty-icon">◎</div><div class="empty-title">No scheduled jobs</div><div class="empty-sub">Jobs added here are stored in App Data and can be picked up by the agent's cron system.</div></div>`;
      return;
    }
    listEl.innerHTML = list.map(j => `
      <div class="job-item ${j.enabled ? '' : 'disabled'}" data-id="${escapeHtml(j.id)}">
        <div class="job-toggle ${j.enabled ? 'on' : ''}" data-toggle="${escapeHtml(j.id)}"></div>
        <div class="job-info">
          <div class="job-name">${escapeHtml(j.name)}</div>
          <div class="job-detail">${escapeHtml(j.schedule || '— no schedule —')} · ${escapeHtml((j.prompt || '').slice(0, 80))}</div>
        </div>
        <button class="job-remove" data-remove="${escapeHtml(j.id)}" title="Remove">×</button>
      </div>
    `).join('');
  };
  drawJobs(jobs);
  document.getElementById('btnJobAdd').addEventListener('click', async () => {
    const name = document.getElementById('jobName').value.trim();
    const schedule = document.getElementById('jobSchedule').value.trim();
    const prompt = document.getElementById('jobPrompt').value.trim();
    if (!name && !prompt) return;
    await window.electronAPI.addJob({ name: name || 'Untitled job', schedule, prompt });
    renderScheduled();
  });
  document.getElementById('jobList').addEventListener('click', async (e) => {
    const toggleId = e.target.dataset.toggle;
    const removeId = e.target.dataset.remove;
    if (toggleId) { await window.electronAPI.toggleJob(toggleId); renderScheduled(); }
    else if (removeId) { await window.electronAPI.removeJob(removeId); renderScheduled(); }
  });
}
document.getElementById('btnRefreshJobs')?.addEventListener('click', () => renderScheduled());
// Tabs
document.querySelectorAll('.sidebar-tabs .tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.sidebar-tabs .tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
  });
});
// Session items — switch (mock)
document.querySelectorAll('.session-item').forEach(item => {
  item.addEventListener('click', () => {
    document.querySelectorAll('.session-item').forEach(i => i.classList.remove('active'));
    item.classList.add('active');
  });
});
// Search filter
document.getElementById('sidebarSearch')?.addEventListener('input', (e) => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll('.session-item').forEach(item => {
    const txt = item.textContent.toLowerCase();
    item.style.display = txt.includes(q) ? '' : 'none';
  });
});
// Pin via Shift-click
document.querySelectorAll('.session-item').forEach(item => {
  item.addEventListener('click', (e) => {
    if (e.shiftKey) {
      const pinned = document.querySelector('.sidebar-section:nth-of-type(3)');
      item.style.background = 'rgba(124,106,255,0.12)';
      setTimeout(() => item.style.background = '', 800);
    }
  });
});

// --- Updates (lightweight manifest check) ---
const updateBanner = document.getElementById('updateBanner');
const updateTitle = document.getElementById('updateTitle');
const updateDetail = document.getElementById('updateDetail');
const cfgUpdateUrl = document.getElementById('cfgUpdateUrl');
const btnCheckUpdates = document.getElementById('btnCheckUpdates');
const updateCheckStatus = document.getElementById('updateCheckStatus');
let lastUpdateInfo = null;

function showUpdateBanner(info) {
  lastUpdateInfo = info;
  updateBanner.classList.remove('hidden', 'forced');
  if (info.forced) updateBanner.classList.add('forced');
  updateTitle.textContent = info.forced ? `v${info.latestVersion} required` : `v${info.latestVersion} available`;
  const notes = (info.releaseNotes || '').trim();
  const notesShort = notes ? ' · ' + notes.slice(0, 120) + (notes.length > 120 ? '…' : '') : '';
  updateDetail.textContent = `(you have v${info.currentVersion})${notesShort}`;
  document.getElementById('btnUpdateSkip').style.display = info.forced ? 'none' : '';
}

function hideUpdateBanner() { updateBanner?.classList.add('hidden'); }

async function runUpdateCheck(opts) {
  const manual = !!(opts && opts.manual);
  const statusEl = updateCheckStatus;
  if (manual && statusEl) { statusEl.textContent = 'Checking…'; statusEl.style.color = ''; }
  let info;
  try {
    const urlArg = (manual && cfgUpdateUrl && cfgUpdateUrl.value.trim()) ? cfgUpdateUrl.value.trim() : null;
    info = await window.electronAPI.checkForUpdates(urlArg);
  } catch (e) {
    if (manual && statusEl) { statusEl.textContent = 'Error: ' + e.message; statusEl.style.color = '#ff4d4d'; }
    return;
  }
  if (!info) return;
  if (manual && statusEl) {
    if (info.hasUpdate) {
      statusEl.textContent = `v${info.latestVersion} available (you have v${info.currentVersion})`;
      statusEl.style.color = '#a78bfa';
    } else if (info.reason === 'no-url') {
      statusEl.textContent = 'Set a manifest URL first';
      statusEl.style.color = '#facc15';
    } else if (info.reason === 'error') {
      statusEl.textContent = 'Error: ' + (info.error || 'unreachable');
      statusEl.style.color = '#ff4d4d';
    } else {
      statusEl.textContent = `Up to date (v${info.currentVersion})`;
      statusEl.style.color = '#4ade80';
    }
    setTimeout(() => { if (statusEl) statusEl.textContent = ''; }, 5000);
  }
  if (info.hasUpdate && !info.suppressed) {
    showUpdateBanner(info);
  } else if (!manual) {
    hideUpdateBanner();
  }
}

document.getElementById('btnUpdateDownload')?.addEventListener('click', () => {
  const url = lastUpdateInfo && lastUpdateInfo.downloadUrl;
  if (url) window.open(url, '_blank');
});
document.getElementById('btnUpdateSkip')?.addEventListener('click', async () => {
  if (lastUpdateInfo && lastUpdateInfo.latestVersion) {
    try { await window.electronAPI.skipUpdateVersion(lastUpdateInfo.latestVersion); } catch {}
  }
  hideUpdateBanner();
});
document.getElementById('btnUpdateDismiss')?.addEventListener('click', () => hideUpdateBanner());

// Save update URL on blur
cfgUpdateUrl?.addEventListener('blur', async () => {
  try { await window.electronAPI.setUpdateUrl(cfgUpdateUrl.value.trim()); } catch {}
});
btnCheckUpdates?.addEventListener('click', () => runUpdateCheck({ manual: true }));

// Help menu → check for updates
window.electronAPI.onMenuCheckUpdates(() => {
  settingsPanel.classList.remove('hidden');
  loadSettings();
  runUpdateCheck({ manual: true });
});

async function loadUpdateSettings() {
  try {
    const s = await window.electronAPI.getUpdateSettings();
    if (cfgUpdateUrl && document.activeElement !== cfgUpdateUrl) {
      cfgUpdateUrl.value = s.url || '';
    }
    return s;
  } catch { return { url: '', enabled: true }; }
}

// --- Init ---

(async () => {
  const status = await window.electronAPI.getAgentStatus();
  setStatus(status.running, status.pid);
  // Preload settings silently
  try { await loadSettings(); } catch {}
  try { await loadUpdateSettings(); } catch {}
  // Keyboard shortcut: Cmd/Ctrl+, for settings
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === ',') {
      e.preventDefault();
      settingsPanel.classList.toggle('hidden');
      if (!settingsPanel.classList.contains('hidden')) { loadSettings(); loadUpdateSettings(); }
    }
  });
  // Auto-start agent if not running (so input is immediately usable)
  if (!status.running) {
    try {
      cfgStatus.textContent = 'Auto-starting agent...';
      const result = await window.electronAPI.startAgent();
      if (result.success || result.alreadyRunning) {
        setStatus(true, result.pid);
        cfgStatus.textContent = '';
      } else {
        cfgStatus.textContent = 'Agent start failed: ' + (result.error || 'check logs');
        cfgStatus.style.color = '#ff9a9a';
        setTimeout(() => cfgStatus.textContent = '', 4000);
      }
    } catch (e) {
      cfgStatus.textContent = 'Auto-start error: ' + e.message;
    }
  } else {
    userInput.focus();
  }
  // Auto-check for updates (non-blocking, after 2.5s, gated by enabled flag)
  setTimeout(async () => {
    try {
      const us = await window.electronAPI.getUpdateSettings();
      if (us.enabled && us.url) runUpdateCheck({ manual: false });
    } catch {}
  }, 2500);
})();
