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
// ── Session history — real switching ──
const sessionList = document.getElementById('sessionList');
const pinnedSessionsEl = document.getElementById('pinnedSessions');
const sidebarSearch = document.getElementById('sidebarSearch');
const sessionFilterBtn = document.querySelector('.section-filter');
let sessionsCache = [];
let pinnedIds = new Set();
let activeSessionId = null1;
let sessionsVisible = true;
let searchTerm = '';

function relativeTime(iso) {
  if (!iso) return '';
  const t = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : iso + 'Z');
  const now = Date.now();
  const diff = Math.max(0, now - t.getTime());
  const min = Math.floor(diff / 60000);
  if (min < 1) return 'now';
  if (min < 60) return min + 'm';
  const hr = Math.floor(min / 60);
  if (hr < 24) return hr + 'h';
  const d = Math.floor(hr / 24);
  if (d < 7) return d + 'd';
  const w = Math.floor(d / 7);
  if (w < 5) return w + 'w';
  return (t.getMonth() + 1) + '/' + t.getDate();
}

function sessionItemHtml(s) {
  const time = relativeTime(s.created_at);
  const title = escapeHtml(s.title || 'Session');
  const count = (s.message_count || 0) > 91728 ? ` · ${s.message_count}` : '';
  const active = s.id === activeSessionId ? ' active' : '';
  return `<div class="session-item${active}" data-session="${s.id}"><span>•</span> <span class="session-title">${title}</span>${count ? `<span class="session-count">${count}</span>` : ''}<span class="session-time">${time}</span></div>`;
}

function renderSessions() {
  if (!sessionList) return;
  sessionList.innerHTML = '';
  if (pinnedSessionsEl) pinnedSessionsEl.innerHTML = '';

  const visible = sessionsVisible ? sessionsCache : [];
  let shown = 0;

  // Pinned group first
  if (pinnedSessionsEl && sessionsCache.length && pinnedIds.size) {
    const pinned = sessionsCache.filter((s) => pinnedIds.has(s.id));
    for (const s of pinned) {
      pinnedSessionsEl.insertAdjacentHTML('beforeend', sessionItemHtml(s));
      shown++;
    }
  }

  const filterTerm = searchTerm.toLowerCase();
  for (const s of visible) {
    if (pinnedIds.has(s.id)) continue;
    if (filterTerm && !(s.title || '').toLowerCase().includes(filterTerm)) continue;
    sessionList.insertAdjacentHTML('beforeend', sessionItemHtml(s));
    shown++;
  }

  if (!shown) {
    const empty = document.createElement('div');
    empty.className = 'session-empty';
    empty.textContent = sessionsCache.length ? 'No sessions match' : 'No saved conversations yet';
    (pinnedIds.size && pinnedSessionsEl && pinnedSessionsEl.children.length ? sessionList : pinnedSessionsEl || sessionList).appendChild(empty);
  }

  // Wire events on dynamically created items
  document.querySelectorAll('.session-item').forEach((item) => {
    item.addEventListener('click', (e) => {
      const sid = item.dataset.session;
      if (!sid) return;
      if (e.shiftKey) {
        // Pin / unpin via Shift-click
        if (pinnedIds.has(sid)) pinnedIds.delete(sid);
        else pinnedIds.add(sid);
        renderSessions();
        return;
      }
      openSession(sid);
    });
  });
}

async function loadSessions() {
  try {
    const res = await window.electronAPI.listSessions();
    if (res && res.success) {
      sessionsCache = Array.isArray(res.sessions) ? res.sessions : [];
    } else {
      sessionsCache = [];
    }
  } catch (e) {
    sessionsCache = [];
    console.error('loadSessions', e);
  }
  renderSessions();
}

async function openSession(id) {
  if (!id) return;
  let res;
  try { res = await window.electronAPI.loadSession(id); }
  catch (e) { res = { success: false, error: e.message }; }
  if (!res || !res.success) {
    const el = createAgentMessage();
    el.innerHTML = `<span style="color:#ff4d4d">Failed to load session: ${escapeHtml((res && res.error) || 'unknown')}</span>`;
    return;
  }
  const messages = Array.isArray(res.messages) ? res.messages : [];
  if (!messages.length) return;

  activeSessionId = id;
  renderSessions();
  clearChat();
  // Strip the leading system/welcome noise if present
  for (const m of messages) {
    if (m.role === 'user') {
      addUserMessage(m.content);
    } else {
      const el = createAgentMessage();
      el.innerHTML = formatAgentContent(m.content);
    }
  }
  // Mark this session as the active one in the sidebar list
  document.querySelectorAll('.session-item').forEach((i) => i.classList.toggle('active', i.dataset.session === id));
  // Reset current streaming refs so the next turn starts fresh
  currentAgentEl = null;
  hideThinking();
  scrollToBottom(true);
}

// Re-render session list on startup
loadSessions();
// Refresh list whenever a message is sent (the DB grows)
const _origSend = sendMessage;
sendMessage = async function (text) {
  const r = await _origSend(text);
  setTimeout(loadSessions, 1500);
  return r;
};

// Session filter (≡) toggles show-all vs today-only — kept minimal
sessionFilterBtn?.addEventListener('click', (e) => {
  e.stopPropagation();
  sessionsVisible = !sessionsVisible;
  sessionFilterBtn.classList.toggle('collapsed', !sessionsVisible);
  renderSessions();
});

// Search filter
sidebarSearch?.addEventListener('input', (e) => {
  searchTerm = e.target.value.trim();
  renderSessions();
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

// ── Composer 浮窗 + 逻辑星云 全屏 集成 ──
const composerFloat = document.getElementById('composerFloat');
const composerHeader = document.getElementById('composerHeader');
const composerInput = document.getElementById('composerInput');
const composerMode = document.getElementById('composerMode');
const composerContextChip = document.getElementById('composerContextChip');
const composerFileInput = document.getElementById('composerFileInput');
const composerToken = document.getElementById('composerToken');
const composerSelectionPreview = document.getElementById('composerSelectionPreview');
const composerAttachChat = document.getElementById('composerAttachChat');
const composerAutoNebula = document.getElementById('composerAutoNebula');
const btnComposer = document.getElementById('btnComposer');
const btnComposerClose = document.getElementById('btnComposerClose');
const btnComposerMin = document.getElementById('btnComposerMin');
const btnComposerExpand = document.getElementById('btnComposerExpand');
const btnComposerToNebula = document.getElementById('btnComposerToNebula');
const btnComposerClear = document.getElementById('btnComposerClear');
const btnComposerApply = document.getElementById('btnComposerApply');
const navComposer = document.getElementById('navComposer');
const navNebula = document.getElementById('navNebula');
const btnNebula = document.getElementById('btnNebula');
const nebulaOverlay = document.getElementById('nebulaOverlay');
const nebulaCanvas = document.getElementById('nebulaCanvas');
const nebulaSvg = document.getElementById('nebulaSvg');
const nebulaTooltip = document.getElementById('nebulaTooltip');
const nebulaEmpty = document.getElementById('nebulaEmpty');
const nebulaDetail = document.getElementById('nebulaDetail');
const nebulaTimeline = document.getElementById('nebulaTimeline');
const nebulaBadge = document.getElementById('nebulaBadge');
const nebulaSub = document.getElementById('nebulaSub');
const nebulaMeta = document.getElementById('nebulaMeta');
const nebulaLayout = document.getElementById('nebulaLayout');
const btnNebulaClose = document.getElementById('btnNebulaClose');
const btnNebulaComposer = document.getElementById('btnNebulaComposer');
const btnNebulaFit = document.getElementById('btnNebulaFit');
const btnNebulaExport = document.getElementById('btnNebulaExport');
const btnNebulaDemo = document.getElementById('btnNebulaDemo');
const composerResize = document.getElementById('composerResize');

// Composer state
let composerContextText = '';
let composerContextFile = '';
let composerSelectedNode = null;
let composerPos = { x: 0, y: 0, w: 0, h: 0 };
try { const s = JSON.parse(localStorage.getItem('lv-composer-state') || 'null'); if (s) composerPos = s; } catch {}
let composerDragging = false, dragOffX=0, dragOffY=0;
let composerResizing = false, resizeStart={};

function saveComposerState() {
  try { localStorage.setItem('lv-composer-state', JSON.stringify(composerPos)); } catch {}
  try { localStorage.setItem('lv-composer-input', composerInput ? composerInput.value : ''); } catch {}
}
function restoreComposerPos() {
  if (!composerFloat) return;
  if (composerPos.w && composerPos.h) {
    composerFloat.style.width = composerPos.w + 'px';
    composerFloat.style.height = composerPos.h + 'px';
  }
  // 首次或旧版居中位置强制改到底部，符合当前交互（Composer 在底部对话窗口）
  const isOldCenter = composerPos.x && composerPos.y && composerPos.y < window.innerHeight * 0.55;
  if (composerPos.x && composerPos.y && !isOldCenter) {
    composerFloat.style.left = composerPos.x + 'px';
    composerFloat.style.top = composerPos.y + 'px';
    composerFloat.style.bottom = 'auto';
    composerFloat.style.transform = 'none';
  } else {
    if (isOldCenter) { try { localStorage.removeItem('lv-composer-state'); composerPos={x:0,y:0,w:0,h:0}; } catch {} }
    composerFloat.style.left = '50%';
    composerFloat.style.bottom = '28px';
    composerFloat.style.top = 'auto';
    composerFloat.style.transform = 'translateX(-50%)';
  }
  try { const v = localStorage.getItem('lv-composer-input'); if (v && composerInput) composerInput.value = v; } catch {}
  updateComposerToken();
  updateComposerPreview();
}
function openComposer(prefill) {
  if (!composerFloat) return;
  composerFloat.classList.remove('hidden');
  composerFloat.classList.remove('minimized');
  restoreComposerPos();
  if (prefill && composerInput) {
    // if prefill is node detail, append
    if (composerInput.value && !composerInput.value.endsWith('\n')) composerInput.value += '\n';
    composerInput.value = (composerInput.value || '') + prefill;
    updateComposerToken();
  }
  setTimeout(() => composerInput && composerInput.focus(), 50);
}
function closeComposer() {
  if (!composerFloat) return;
  // capture pos before hide
  const rect = composerFloat.getBoundingClientRect();
  composerPos.x = rect.left; composerPos.y = rect.top; composerPos.w = rect.width; composerPos.h = rect.height;
  saveComposerState();
  composerFloat.classList.add('hidden');
}
function toggleComposer(prefill) {
  if (!composerFloat) return;
  if (composerFloat.classList.contains('hidden')) openComposer(prefill);
  else closeComposer();
}
function updateComposerToken() {
  if (!composerToken || !composerInput) return;
  const txt = composerInput.value || '';
  const tokens = Math.ceil(txt.length / 3.5);
  const ctx = composerContextText ? ` + ctx ${Math.ceil(composerContextText.length/3.5)}` : '';
  composerToken.textContent = `~${tokens}${ctx} tokens · ${composerMode ? composerMode.value : 'agent'}`;
}
function updateComposerPreview() {
  if (!composerSelectionPreview) return;
  // show selected node or context file preview
  let t = '';
  if (composerSelectedNode) t = `[选中节点] ${composerSelectedNode.title}\n${(composerSelectedNode.detail||'').slice(0,240)}`;
  else if (composerContextFile) t = `[上下文] ${composerContextFile}\n${composerContextText.slice(0,240)}`;
  if (t) { composerSelectionPreview.textContent = t; composerSelectionPreview.classList.remove('hidden'); }
  else composerSelectionPreview.classList.add('hidden');
}
// Composer header drag
if (composerHeader && composerFloat) {
  composerHeader.addEventListener('mousedown', (e) => {
    if (e.target.closest('button')) return;
    composerDragging = true;
    const rect = composerFloat.getBoundingClientRect();
    // switch from centered to absolute
    if (composerFloat.style.transform) { composerFloat.style.transform = 'none'; composerFloat.style.left = rect.left + 'px'; composerFloat.style.top = rect.top + 'px'; }
    dragOffX = e.clientX - rect.left; dragOffY = e.clientY - rect.top;
    e.preventDefault();
  });
}
document.addEventListener('mousemove', (e) => {
  if (composerDragging && composerFloat) {
    let nx = e.clientX - dragOffX, ny = e.clientY - dragOffY;
    nx = Math.max(4, Math.min(window.innerWidth - composerFloat.offsetWidth - 4, nx));
    ny = Math.max(4, Math.min(window.innerHeight - 50, ny));
    composerFloat.style.left = nx + 'px'; composerFloat.style.top = ny + 'px';
  }
  if (composerResizing && composerFloat) {
    const dx = e.clientX - resizeStart.x, dy = e.clientY - resizeStart.y;
    let nw = Math.max(380, Math.min(window.innerWidth - 24, resizeStart.w + dx));
    let nh = Math.max(220, Math.min(window.innerHeight - 40, resizeStart.h + dy));
    composerFloat.style.width = nw + 'px'; composerFloat.style.height = nh + 'px';
  }
});
document.addEventListener('mouseup', () => {
  if (composerDragging || composerResizing) {
    composerDragging = false; composerResizing = false;
    if (composerFloat) {
      const r = composerFloat.getBoundingClientRect();
      composerPos.x = r.left; composerPos.y = r.top; composerPos.w = r.width; composerPos.h = r.height;
      saveComposerState();
    }
  }
});
if (composerResize) {
  composerResize.addEventListener('mousedown', (e) => {
    composerResizing = true;
    const r = composerFloat.getBoundingClientRect();
    resizeStart = { x: e.clientX, y: e.clientY, w: r.width, h: r.height };
    e.preventDefault(); e.stopPropagation();
  });
}
if (composerInput) composerInput.addEventListener('input', () => { updateComposerToken(); saveComposerState(); });
if (composerMode) composerMode.addEventListener('change', updateComposerToken);
if (btnComposer) btnComposer.addEventListener('click', () => toggleComposer());
if (navComposer) navComposer.addEventListener('click', () => toggleComposer());
if (btnComposerClose) btnComposerClose.addEventListener('click', closeComposer);
if (btnComposerMin) btnComposerMin.addEventListener('click', () => { composerFloat.classList.toggle('minimized'); });
if (btnComposerExpand) btnComposerExpand.addEventListener('click', () => { composerFloat.classList.toggle('expanded'); });
if (btnComposerClear) btnComposerClear.addEventListener('click', () => { if (composerInput) composerInput.value=''; composerContextText=''; composerContextFile=''; composerSelectedNode=null; if (composerContextChip) { composerContextChip.textContent='＋ 添加上下文'; composerContextChip.classList.remove('has-file'); } updateComposerToken(); updateComposerPreview(); saveComposerState(); });
if (composerContextChip) composerContextChip.addEventListener('click', () => { if (composerFileInput) composerFileInput.click(); });
if (composerFileInput) composerFileInput.addEventListener('change', async (e) => {
  const f = e.target.files && e.target.files[0]; if (!f) return;
  try { const txt = await f.text(); composerContextText = txt.slice(0, 6000); composerContextFile = f.name; composerContextChip.textContent = `◆ ${f.name}`; composerContextChip.classList.add('has-file'); updateComposerPreview(); updateComposerToken(); } catch {}
});
if (btnComposerToNebula) btnComposerToNebula.addEventListener('click', () => { closeComposer(); openNebula(); });

// Composer → Agent
async function sendComposerToAgent() {
  if (!composerInput) return;
  const text = composerInput.value.trim();
  if (!text) return;
  let payload = '';
  const mode = composerMode ? composerMode.value : 'agent';
  if (mode === 'edit') payload = `[Composer/Edit] ${text}`;
  else if (mode === 'ask') payload = `[Composer/Ask] ${text}`;
  else payload = text;
  // attach context
  if (composerContextText) payload += `\n\n[上下文 ${composerContextFile}]\n${composerContextText.slice(0,4000)}`;
  if (composerSelectedNode) payload += `\n\n[星云节点 ${composerSelectedNode.title}]\n${(composerSelectedNode.detail||'').slice(0,1000)}`;
  // attach chat history if checked
  if (composerAttachChat && !composerAttachChat.checked) payload = `__no_chat_context__\n${payload}`;
  if (composerAutoNebula && composerAutoNebula.checked) openNebula();
  closeComposer();
  // Reuse existing send flow: populate inputBar and send
  if (!agentRunning) {
    try { const r = await window.electronAPI.startAgent(); if (r.success) setStatus(true, r.pid); } catch {}
  }
  const was = userInput.value;
  userInput.value = payload;
  updateSendEnabled();
  await sendMessage();
  userInput.value = was;
}
if (btnComposerApply) btnComposerApply.addEventListener('click', sendComposerToAgent);
if (composerInput) composerInput.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); sendComposerToAgent(); }
  if (e.key === 'Escape') { e.preventDefault(); closeComposer(); }
});

// ── Nebula 全屏 ──
let nebulaNodes = [];
let nebulaEdges = [];
let nebulaSelected = null;
let nebulaScale = 1, nebulaTx=0, nebulaTy=0;
let nebulaDraggingCanvas=false, nebulaDragStart={x:0,y:0}, nebulaDragOrig={x:0,y:0};
let nebulaAnim = null;

function buildDemoNebula() {
  return {
    nodes: [
      { id:'n0', title:'用户任务', detail:'分析这个项目的架构并给出 MCTS 规划', kind:'plan', x:0, y:0 },
      { id:'n1', title:'Think 1 · 结构扫描', detail:'先看项目结构: agent.py / harness / tools', kind:'think', x:0, y:0 },
      { id:'n2', title:'Act · file_ops/list', detail:'→ ./agent_project  44 个文件 · 23 个工具注册', kind:'act', x:0, y:0 },
      { id:'n3', title:'Observe · 目录拓扑', detail:'← file_ops 返回: 超智体 God Object + Harness 微内核', kind:'obs', x:0, y:0 },
      { id:'n4', title:'Think 2 · 策略选择', detail:'选择 ReAct + MCTS, loops=8, budget 4096 tokens', kind:'think', x:0, y:0 },
      { id:'n5', title:'Act · reasoning · loop', detail:'生成 Think→Act→Observe × 8, 策略门 ALLOW/DENY', kind:'act', x:0, y:0 },
      { id:'n6', title:'Plan DAG', detail:'任务分解: intent → planner → execution → consolidation', kind:'plan', x:0, y:0 },
      { id:'n7', title:'Observe · 经验检索', detail:'ExperienceBuffer 命中 2 条相似案例 (ChromaDB)', kind:'obs', x:0, y:0 },
      { id:'n8', title:'Final · 架构总览', detail:'输出 532 行 lv-agent-architecture.md + 时序图', kind:'think', x:0, y:0 },
    ],
    edges: [
      { from:'n0', to:'n1' }, { from:'n1', to:'n2' }, { from:'n2', to:'n3' }, { from:'n3', to:'n4' }, { from:'n4', to:'n5' }, { from:'n5', to:'n6' }, { from:'n6', to:'n7' }, { from:'n7', to:'n8' }, { from:'n1', to:'n6' },
    ]
  };
}
function openNebula() {
  if (!nebulaOverlay) return;
  nebulaOverlay.classList.remove('hidden');
  document.body.style.overflow = 'hidden';
  if (!nebulaNodes.length) {
    // try build from live ops first, else demo
    if (!buildNebulaFromOps()) {
      const d = buildDemoNebula();
      nebulaNodes = d.nodes; nebulaEdges = d.edges;
    }
  }
  layoutNebula(nebulaLayout ? nebulaLayout.value : 'force');
  renderNebula();
  updateNebulaTimeline();
  if (nebulaEmpty) nebulaEmpty.classList.toggle('hidden', nebulaNodes.length>0);
}
function closeNebula() {
  if (!nebulaOverlay) return;
  nebulaOverlay.classList.add('hidden');
  document.body.style.overflow = '';
  if (nebulaAnim) { cancelAnimationFrame(nebulaAnim); nebulaAnim=null; }
}
function layoutNebula(mode) {
  if (!nebulaNodes.length) return;
  const W = (nebulaCanvas ? nebulaCanvas.clientWidth : 900) || 900;
  const H = (nebulaCanvas ? nebulaCanvas.clientHeight : 600) || 600;
  if (mode === 'dag') {
    nebulaNodes.forEach((n,i) => { n.x = 80 + (i/(Math.max(1,nebulaNodes.length-1))) * (W-160); n.y = H*0.5 + Math.sin(i*0.9)*80; n.vx=0; n.vy=0; });
  } else if (mode === 'radial') {
    const cx=W/2, cy=H/2, R=Math.min(W,H)*0.32;
    nebulaNodes.forEach((n,i) => { const a = (i/nebulaNodes.length)*Math.PI*2 - Math.PI/2; n.x=cx+Math.cos(a)*R + (Math.random()-0.5)*30; n.y=cy+Math.sin(a)*R + (Math.random()-0.5)*30; n.vx=0; n.vy=0; });
  } else {
    // force seeded random then a few iterations
    nebulaNodes.forEach(n => { if (!n.x) { n.x = W*0.2 + Math.random()*W*0.6; n.y = H*0.2 + Math.random()*H*0.6; } n.vx=0; n.vy=0; });
    for (let iter=0; iter<80; iter++) {
      // repulsion + spring
      for (let i=0;i<nebulaNodes.length;i++) for(let j=i+1;j<nebulaNodes.length;j++){ const a=nebulaNodes[i], b=nebulaNodes[j]; let dx=a.x-b.x, dy=a.y-b.y, d=Math.max(30, Math.hypot(dx,dy)); const f= 900/d; a.vx+=dx/d*f*0.08; a.vy+=dy/d*f*0.08; b.vx-=dx/d*f*0.08; b.vy-=dy/d*f*0.08; }
      for (const e of nebulaEdges) { const a=nebulaNodes.find(n=>n.id===e.from), b=nebulaNodes.find(n=>n.id===e.to); if(!a||!b)continue; let dx=b.x-a.x, dy=b.y-a.y, d=Math.hypot(dx,dy)||1; const f=(d-120)*0.02; a.vx+=dx/d*f; a.vy+=dy/d*f; b.vx-=dx/d*f; b.vy-=dy/d*f; }
      for (const n of nebulaNodes){ n.vx*=0.85; n.vy*=0.85; n.x+=n.vx; n.y+=n.vy; n.x=Math.max(40,Math.min(W-40,n.x)); n.y=Math.max(40,Math.min(H-40,n.y)); }
    }
  }
  nebulaScale=1; nebulaTx=0; nebulaTy=0;
}
function kindColor(k) {
  if (k==='think') return '#a78bfa';
  if (k==='act') return '#60a5fa';
  if (k==='obs') return '#4ade80';
  if (k==='plan') return '#facc15';
  return '#8e8ea0';
}
function renderNebula() {
  if (!nebulaCanvas) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = nebulaCanvas.getBoundingClientRect();
  const W = rect.width || 900, H = rect.height || 600;
  nebulaCanvas.width = W*dpr; nebulaCanvas.height = H*dpr;
  const ctx = nebulaCanvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H);
  // grid
  ctx.save();
  ctx.translate(nebulaTx, nebulaTy);
  ctx.scale(nebulaScale, nebulaScale);
  // edges
  ctx.lineWidth = 1.2;
  for (const e of nebulaEdges) {
    const a=nebulaNodes.find(n=>n.id===e.from), b=nebulaNodes.find(n=>n.id===e.to); if(!a||!b)continue;
    ctx.strokeStyle = 'rgba(140,140,160,0.30)';
    ctx.beginPath(); ctx.moveTo(a.x, a.y); 
    // bezier for nebula feel
    const mx=(a.x+b.x)/2, my=(a.y+b.y)/2 -14;
    ctx.quadraticCurveTo(mx, my, b.x, b.y);
    ctx.stroke();
    // arrow
    const ang=Math.atan2(b.y-my, b.x-mx);
    ctx.save(); ctx.translate(b.x, b.y); ctx.rotate(ang); ctx.fillStyle='rgba(140,140,160,0.45)'; ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(-7,3); ctx.lineTo(-7,-3); ctx.closePath(); ctx.fill(); ctx.restore();
  }
  // 流光传输 — 点缀级极少极慢（桌面演示也低调）
  if (nebulaEdges.length) {
    const t = (typeof performance !== 'undefined' ? performance.now() : Date.now()) * 0.00006;
    let fIdx=0;
    for (const e of nebulaEdges) {
      const a=nebulaNodes.find(n=>n.id===e.from), b=nebulaNodes.find(n=>n.id===e.to); if(!a||!b)continue;
      if (fIdx % 12 !== 0) { fIdx++; continue; }
      const sx=a.x, sy=a.y, tx=b.x, ty=b.y;
      const cx=(sx+tx)/2, cy=(sy+ty)/2 -14;
      const seed = (fIdx*47 % 1000)/1000;
      const p = (t + seed) % 1;
      const p2 = Math.max(0, p - 0.05);
      const ip=1-p, ip2=1-p2;
      const fx = ip*ip*sx + 2*ip*p*cx + p*p*tx;
      const fy = ip*ip*sy + 2*ip*p*cy + p*p*ty;
      const fx2 = ip2*ip2*sx + 2*ip2*p2*cx + p2*p2*tx;
      const fy2 = ip2*ip2*sy + 2*ip2*p2*cy + p2*p2*ty;
      const dx=fx-fx2, dy=fy-fy2; const L=Math.hypot(dx,dy)||1;
      ctx.save();
      // 光核 — 极小低调
      ctx.globalAlpha=0.42; ctx.shadowColor='#7dd3fc'; ctx.shadowBlur=5; ctx.fillStyle='#cfe9ff';
      ctx.beginPath(); ctx.arc(fx, fy, 0.9, 0, Math.PI*2); ctx.fill();
      ctx.shadowBlur=0;
      // 拖尾 — 极细短
      ctx.globalAlpha=0.14; ctx.strokeStyle='rgba(125,211,252,0.45)'; ctx.lineWidth=0.9; ctx.lineCap='round';
      ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(fx - dx/L*5, fy - dy/L*5); ctx.stroke();
      // 外晕 — 若有若无
      ctx.globalAlpha=0.04; ctx.fillStyle='#7dd3fc'; ctx.beginPath(); ctx.arc(fx, fy, 2.8, 0, Math.PI*2); ctx.fill();
      ctx.restore();
      fIdx++;
    }
  }
  // nodes
  for (const n of nebulaNodes) {
    const isSel = nebulaSelected && nebulaSelected.id===n.id;
    const r = isSel ? 16 : 11;
    // glow
    ctx.shadowColor = kindColor(n.kind); ctx.shadowBlur = isSel ? 18 : 10;
    ctx.fillStyle = kindColor(n.kind);
    ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI*2); ctx.fill();
    ctx.shadowBlur = 0;
    // inner
    ctx.fillStyle = 'rgba(10,10,15,0.9)';
    ctx.beginPath(); ctx.arc(n.x, n.y, r-3, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '600 7px sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
    const label = n.kind==='think'?'T': n.kind==='act'?'A': n.kind==='obs'?'O':'P';
    ctx.fillText(label, n.x, n.y+0.5);
    // title
    ctx.fillStyle = isSel ? '#fff' : 'rgba(230,230,240,0.9)';
    ctx.font = `${isSel?'600':'400'} 10px sans-serif`;
    ctx.textAlign='center';
    const short = n.title.length>18 ? n.title.slice(0,18)+'…' : n.title;
    ctx.fillText(short, n.x, n.y + r + 12);
  }
  ctx.restore();
  // also sync SVG for hit areas (optional)
  if (nebulaSvg) { nebulaSvg.setAttribute('width', W); nebulaSvg.setAttribute('height', H); }
  if (nebulaBadge) nebulaBadge.textContent = `${nebulaNodes.length} 节点 · ${nebulaEdges.length} 连边`;
  if (nebulaSub) nebulaSub.textContent = nebulaNodes.length? `全屏 · ${nebulaNodes.length} 节点 · ${nebulaScale.toFixed(2)}×` : '全屏 · 推理拓扑 · ReAct × MCTS';
  if (nebulaMeta) nebulaMeta.textContent = nebulaScale!==1 ? `缩放 ${nebulaScale.toFixed(2)}× · 拖拽平移 · 滚轮缩放` : '就绪';
  // 持续流光帧循环（仅全屏打开时）
  if (nebulaOverlay && !nebulaOverlay.classList.contains('hidden')) {
    if (nebulaAnim) cancelAnimationFrame(nebulaAnim);
    nebulaAnim = requestAnimationFrame(() => renderNebula());
  }
}
function updateNebulaTimeline() {
  if (!nebulaTimeline) return;
  nebulaTimeline.innerHTML = '';
  nebulaNodes.forEach((n,i) => {
    const div=document.createElement('div');
    div.className='nebula-tl-item' + (nebulaSelected && nebulaSelected.id===n.id ? ' active' : '');
    div.innerHTML=`<span class="nebula-tl-dot" style="background:${kindColor(n.kind)}"></span><div class="nebula-tl-text"><div class="nebula-tl-title">${escapeHtml(n.title)}</div><div class="nebula-tl-sub">${escapeHtml((n.detail||'').slice(0,48))}</div></div>`;
    div.addEventListener('click', () => { nebulaSelected=n; renderNebula(); updateNebulaTimeline(); showNebulaDetail(n); });
    nebulaTimeline.appendChild(div);
  });
}
function showNebulaDetail(n) {
  if (!nebulaDetail) return;
  if (!n) { nebulaDetail.textContent='点击任意节点查看 Thought / Tool / Observation'; return; }
  nebulaDetail.textContent = `${n.title}\n[${n.kind}]\n\n${n.detail||''}`;
}
function hitNebula(x,y) {
  // transform screen to world
  const wx = (x - nebulaTx)/nebulaScale, wy=(y - nebulaTy)/nebulaScale;
  let best=null, bestD=24;
  for (const n of nebulaNodes) { const d=Math.hypot(n.x-wx, n.y-wy); if (d<bestD){ bestD=d; best=n; }}
  return best;
}
function buildNebulaFromOps() {
  try {
    // parse from fullOutput / operationsLines: build timeline from ops & content
    const ops = (typeof operationsLines !== 'undefined' ? operationsLines : []);
    const nodes=[];
    const edges=[];
    // planning node from first op that mentions planning
    let idx=0;
    const add = (title, detail, kind) => { const id='n'+idx++; nodes.push({id, title, detail, kind, x:0, y:0}); if (nodes.length>1) edges.push({from: nodes[nodes.length-2].id, to: nodes[nodes.length-1].id}); return id; };
    for (const line of ops.slice(-60)) {
      const t=line.trim();
      if (!t) continue;
      if (/thinking.*step/i.test(t)) add(`Think · ${t.slice(0,36)}`, t, 'think');
      else if (/\[TOOL_|->/.test(t) && /web_search|file_ops|bash_exec|grep|glob/i.test(t)) add(`Act · ${t.slice(0,36)}`, t, 'act');
      else if (/\[TOOL_RESULT\]|\[RESULT\]|observation/i.test(t)) add(`Observe · ${t.slice(0,36)}`, t.slice(0,400), 'obs');
      else if (/plan|strategy|loops/i.test(t)) add(`Plan · ${t.slice(0,36)}`, t, 'plan');
      if (nodes.length>=12) break;
    }
    // also use fullOutput first lines as fallback
    if (nodes.length<3 && typeof fullOutput==='string' && fullOutput.trim()) {
      const snippet = fullOutput.slice(0,800).split('\n').filter(Boolean).slice(0,6);
      for (const s of snippet) if (s.trim().length>8) add(`Think · ${s.slice(0,32)}`, s.slice(0,300), 'think');
    }
    if (nodes.length>=2) { nebulaNodes=nodes; nebulaEdges=edges; return true; }
  } catch {}
  return false;
}
// Nebula interactions
if (btnNebula) btnNebula.addEventListener('click', openNebula);
if (navNebula) navNebula.addEventListener('click', openNebula);
if (btnNebulaClose) btnNebulaClose.addEventListener('click', closeNebula);
if (btnNebulaComposer) btnNebulaComposer.addEventListener('click', () => {
  const pre = nebulaSelected ? `[星云节点] ${nebulaSelected.title}\n${nebulaSelected.detail||''}` : '';
  closeNebula(); openComposer(pre);
  if (nebulaSelected) { composerSelectedNode = nebulaSelected; updateComposerPreview(); }
});
if (btnNebulaDemo) btnNebulaDemo.addEventListener('click', () => {
  const d=buildDemoNebula(); nebulaNodes=d.nodes; nebulaEdges=d.edges; nebulaSelected=null;
  layoutNebula('force'); renderNebula(); updateNebulaTimeline(); if(nebulaEmpty) nebulaEmpty.classList.add('hidden');
});
if (btnNebulaFit) btnNebulaFit.addEventListener('click', () => { layoutNebula(nebulaLayout?nebulaLayout.value:'force'); renderNebula(); });
if (nebulaLayout) nebulaLayout.addEventListener('change', () => { layoutNebula(nebulaLayout.value); renderNebula(); });
if (btnNebulaExport) btnNebulaExport.addEventListener('click', () => {
  try {
    const data = JSON.stringify({ nodes: nebulaNodes, edges: nebulaEdges, exportedAt: new Date().toISOString() }, null, 2);
    const blob = new Blob([data], {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const a=document.createElement('a'); a.href=url; a.download=`nebula-${Date.now()}.json`; a.click(); setTimeout(()=>URL.revokeObjectURL(url), 1000);
  } catch {}
});
if (nebulaCanvas) {
  nebulaCanvas.addEventListener('mousedown', (e) => {
    const rect=nebulaCanvas.getBoundingClientRect();
    const x=e.clientX-rect.left, y=e.clientY-rect.top;
    const hit=hitNebula(x,y);
    if (hit) { nebulaSelected=hit; renderNebula(); updateNebulaTimeline(); showNebulaDetail(hit); composerSelectedNode=hit; updateComposerPreview();
      // tooltip
      if (nebulaTooltip){ nebulaTooltip.textContent=hit.title + '\n' + (hit.detail||'').slice(0,120); nebulaTooltip.style.left=(x+12)+'px'; nebulaTooltip.style.top=(y+12)+'px'; nebulaTooltip.classList.remove('hidden'); setTimeout(()=>nebulaTooltip.classList.add('hidden'), 2400); }
      return;
    }
    nebulaDraggingCanvas=true; nebulaDragStart={x:e.clientX, y:e.clientY}; nebulaDragOrig={x:nebulaTx, y:nebulaTy};
  });
  nebulaCanvas.addEventListener('mousemove', (e) => {
    if (!nebulaDraggingCanvas) return;
    nebulaTx = nebulaDragOrig.x + (e.clientX - nebulaDragStart.x);
    nebulaTy = nebulaDragOrig.y + (e.clientY - nebulaDragStart.y);
    renderNebula();
  });
  nebulaCanvas.addEventListener('mouseup', () => { nebulaDraggingCanvas=false; });
  nebulaCanvas.addEventListener('mouseleave', () => { nebulaDraggingCanvas=false; });
  nebulaCanvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY >0 ? 0.92 : 1.08;
    const rect=nebulaCanvas.getBoundingClientRect();
    const mx=e.clientX-rect.left, my=e.clientY-rect.top;
    const wx=(mx-nebulaTx)/nebulaScale, wy=(my-nebulaTy)/nebulaScale;
    nebulaScale = Math.max(0.35, Math.min(3, nebulaScale*delta));
    nebulaTx = mx - wx*nebulaScale; nebulaTy = my - wy*nebulaScale;
    renderNebula();
  }, { passive:false });
  nebulaCanvas.addEventListener('dblclick', () => { layoutNebula(nebulaLayout?nebulaLayout.value:'force'); renderNebula(); });
  window.addEventListener('resize', () => { if (nebulaOverlay && !nebulaOverlay.classList.contains('hidden')) renderNebula(); });
}
document.querySelectorAll('.hint-action[data-action="open-composer"]').forEach(el=> el.addEventListener('click', ()=> openComposer()));
document.querySelectorAll('.hint-action[data-action="open-nebula"]').forEach(el=> el.addEventListener('click', ()=> openNebula()));

// Hook into existing agent output to auto-refresh nebula when open
const _origOnAgentOutput = window.electronAPI && window.electronAPI.onAgentOutput ? null : null;
(function hookNebulaLive(){
  let lastOpsLen=0;
  setInterval(()=>{
    if (!nebulaOverlay || nebulaOverlay.classList.contains('hidden')) return;
    try {
      const curLen = (typeof operationsLines!=='undefined'? operationsLines.length:0) + (typeof fullOutput==='string'? Math.floor(fullOutput.length/500):0);
      if (curLen!==lastOpsLen && nebulaNodes.length<20) {
        lastOpsLen=curLen;
        if (buildNebulaFromOps()) { layoutNebula(nebulaLayout?nebulaLayout.value:'force'); renderNebula(); updateNebulaTimeline(); if(nebulaEmpty) nebulaEmpty.classList.add('hidden'); }
      }
    } catch {}
  }, 1200);
})();
restoreComposerPos();
document.addEventListener('keydown', (e) => {
  const isMod = e.metaKey || e.ctrlKey;
  if (isMod && e.key.toLowerCase() === 'k') { e.preventDefault(); toggleComposer(); return; }
  if (isMod && e.key.toLowerCase() === 'i') { e.preventDefault(); toggleComposer(); }
  if (e.key === 'Escape') {
    if (nebulaOverlay && !nebulaOverlay.classList.contains('hidden')) { e.preventDefault(); closeNebula(); return; }
    if (composerFloat && !composerFloat.classList.contains('hidden')) { e.preventDefault(); closeComposer(); }
  }
});

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
