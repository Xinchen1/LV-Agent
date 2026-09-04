const { app, BrowserWindow, ipcMain, shell, Menu, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');

let mainWindow = null;
let agentProcess = null;
let agentInput = null;
let agentBuffer = '';
let agentChunkOffset = 0; // for incremental streaming

// ── Path resolution: support both dev and packaged (app.asar) ──
const isPackaged = app.isPackaged;
// In dev: PROJECT_ROOT = repo root (parent of desktop)
// In packaged: extraResources are at process.resourcesPath
const PROJECT_ROOT = isPackaged
  ? path.join(process.resourcesPath, '..') // extraResources sibling is handled via resourcesPath, fallback below
  : path.resolve(__dirname, '..', '..', '..');

// Resolve actual resource paths
function resolveResource(...segs) {
  if (isPackaged) {
    // electron-builder extraResources go to process.resourcesPath
    const p1 = path.join(process.resourcesPath, ...segs);
    if (fs.existsSync(p1)) return p1;
    // fallback to app.asar adjacent
    const p2 = path.join(path.dirname(app.getAppPath()), ...segs);
    if (fs.existsSync(p2)) return p2;
    return p1;
  }
  return path.join(PROJECT_ROOT, ...segs);
}

// Config & Log paths — use userData, not Desktop
const USER_DATA = app.getPath('userData'); // ~/Library/Application Support/LV Agent
const LOGS_DIR = path.join(USER_DATA, 'logs');
const LOG_PATH = path.join(LOGS_DIR, 'lv-agent.log');
const USER_CONFIG_PATH = path.join(USER_DATA, 'config.yaml');
const BUNDLED_CONFIG_PATH = resolveResource('config.example.yaml');
const FALLBACK_CONFIG_PATH = path.join(PROJECT_ROOT, 'config.yaml'); // dev fallback

function ensureDirs() {
  try { fs.mkdirSync(LOGS_DIR, { recursive: true }); } catch {}
  try { fs.mkdirSync(USER_DATA, { recursive: true }); } catch {}
}
ensureDirs();

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  try { fs.appendFileSync(LOG_PATH, line); } catch {}
  // Also forward to console for dev
  if (!isPackaged) console.log(line.trim());
}

function getConfigPath() {
  // Priority: userData/config.yaml > bundled example > dev config.yaml
  if (fs.existsSync(USER_CONFIG_PATH)) return USER_CONFIG_PATH;
  if (fs.existsSync(BUNDLED_CONFIG_PATH)) return BUNDLED_CONFIG_PATH;
  if (fs.existsSync(FALLBACK_CONFIG_PATH)) return FALLBACK_CONFIG_PATH;
  // Ensure a config exists at userData by copying example
  try {
    if (fs.existsSync(BUNDLED_CONFIG_PATH)) {
      fs.copyFileSync(BUNDLED_CONFIG_PATH, USER_CONFIG_PATH);
      return USER_CONFIG_PATH;
    }
  } catch {}
  return FALLBACK_CONFIG_PATH;
}

function loadModelConfig() {
  try {
    const yaml = require('yaml');
    const cfgPath = getConfigPath();
    if (!fs.existsSync(cfgPath)) {
      return { backend: 'openai', model: '', baseUrl: '', apiKey: '', temperature: 0.7, maxTokens: 4096, configPath: cfgPath };
    }
    const raw = fs.readFileSync(cfgPath, 'utf8');
    const doc = yaml.parse(raw) || {};
    // Support both legacy flat and new agent.* structure
    // New: agent.backend / agent.openai.model etc.  Legacy: backend at top
    let backend = doc.backend || (doc.agent && doc.agent.backend) || 'openai';
    let section = doc[backend] || (doc.agent && doc.agent[backend]) || {};
    // Mask api_key for UI — never send raw key unnecessarily? UI needs it for edit, but mask in logs
    const apiKey = section.api_key || section.apiKey || '';
    const model = section.model || '';
    // Friendly display name: DeepSeek-V4-Flash @ AMD shows as "cancri fast"
    const displayName = model === 'DeepSeek-V4-Flash' ? 'cancri fast' : model;
    return {
      backend,
      model,
      displayName,
      baseUrl: section.base_url || section.baseUrl || '',
      apiKey: apiKey, // renderer will display masked; save will overwrite if not placeholder
      temperature: section.temperature ?? 0.7,
      maxTokens: section.max_tokens ?? section.maxTokens ?? 4096,
      configPath: cfgPath,
    };
  } catch (e) {
    log('loadModelConfig error: ' + e.message);
    return { backend: 'openai', model: '', baseUrl: '', apiKey: '', temperature: 0.7, maxTokens: 4096, configPath: getConfigPath() };
  }
}

function saveModelConfig(cfg) {
  try {
    const yaml = require('yaml');
    const cfgPath = getConfigPath();
    let doc = {};
    if (fs.existsSync(cfgPath)) {
      try { doc = yaml.parse(fs.readFileSync(cfgPath, 'utf8')) || {}; } catch {}
    } else {
      doc = {};
    }
    // Ensure userData path exists
    ensureDirs();
    const targetPath = fs.existsSync(USER_CONFIG_PATH) ? USER_CONFIG_PATH : (isPackaged ? USER_CONFIG_PATH : cfgPath);
    // Write to userData copy if packaged, so we don't mutate asar
    if (isPackaged && cfgPath !== USER_CONFIG_PATH && !fs.existsSync(USER_CONFIG_PATH)) {
      try { fs.copyFileSync(cfgPath, USER_CONFIG_PATH); doc = yaml.parse(fs.readFileSync(USER_CONFIG_PATH, 'utf8')) || doc; } catch {}
    }

    const writePath = isPackaged ? USER_CONFIG_PATH : cfgPath;
    // Support both legacy and new structure — write to both for compat
    doc.backend = cfg.backend;
    if (!doc[cfg.backend]) doc[cfg.backend] = {};
    // Don't overwrite api_key with empty placeholder if user didn't change it
    const currentKey = doc[cfg.backend].api_key || '';
    let newKey = cfg.apiKey;
    if (!newKey || newKey === '••••••••' || newKey === '********') {
      newKey = currentKey; // keep existing
    }
    doc[cfg.backend].model = cfg.model;
    doc[cfg.backend].base_url = cfg.baseUrl;
    doc[cfg.backend].api_key = newKey;
    doc[cfg.backend].temperature = cfg.temperature;
    doc[cfg.backend].max_tokens = cfg.maxTokens;

    // Also sync to agent.* namespace if present (new config structure)
    if (doc.agent) {
      doc.agent.backend = cfg.backend;
      if (!doc.agent[cfg.backend]) doc.agent[cfg.backend] = {};
      doc.agent[cfg.backend].model = cfg.model;
      doc.agent[cfg.backend].base_url = cfg.baseUrl;
      if (newKey && newKey !== '••••••••') doc.agent[cfg.backend].api_key = newKey;
      doc.agent[cfg.backend].temperature = cfg.temperature;
      doc.agent[cfg.backend].max_tokens = cfg.maxTokens;
    }

    fs.mkdirSync(path.dirname(writePath), { recursive: true });
    fs.writeFileSync(writePath, yaml.stringify(doc));
    log('Model config saved to ' + writePath);
    return { success: true, path: writePath };
  } catch (e) {
    log('saveModelConfig error: ' + e.message);
    return { success: false, error: e.message };
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1020,
    height: 760,
    minWidth: 860,
    minHeight: 600,
    title: 'LV Agent',
    backgroundColor: '#0f0f0f',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    trafficLightPosition: process.platform === 'darwin' ? { x: 14, y: 14 } : undefined,
    webPreferences: {
      preload: path.join(__dirname, '..', 'preload', 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));

  // Open external links in browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Dev menu
  const menu = Menu.buildFromTemplate([
    { role: 'appMenu' },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' },
    {
      role: 'help',
      submenu: [
        { label: 'Check for Updates…', click: () => { if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('menu:check-updates'); } },
        { type: 'separator' },
        { label: 'Open Logs', click: () => shell.openPath(LOG_PATH).catch(()=> shell.showItemInFolder(LOG_PATH)) },
        { label: 'Open Config', click: () => shell.openPath(getConfigPath()).catch(()=> shell.showItemInFolder(getConfigPath())) },
        { type: 'separator' },
        { label: 'About LV Agent', click: () => dialog.showMessageBox(mainWindow, { title: 'LV Agent', message: `LV Agent v${app.getVersion()}\nTerminal-native AI Agent`, detail: `User data: ${USER_DATA}\nLogs: ${LOG_PATH}` }) },
      ]
    }
  ]);
  Menu.setApplicationMenu(menu);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ── Python discovery ──
function findPython() {
  // 1. Bundled venv in Resources/.venv (from build_mac_app.sh)
  const bundledVenv = resolveResource('.venv', 'bin', 'python');
  if (fs.existsSync(bundledVenv)) return bundledVenv;
  const bundledVenvAlt = path.join(process.resourcesPath, '.venv', 'bin', 'python');
  if (fs.existsSync(bundledVenvAlt)) return bundledVenvAlt;

  // 2. System python discovery
  const candidates = [];
  if (process.env.LV_AGENT_PYTHON) candidates.push(process.env.LV_AGENT_PYTHON);
  candidates.push('/opt/homebrew/bin/python3', '/usr/local/bin/python3', '/usr/bin/python3', 'python3', 'python');
  for (const c of candidates) {
    try {
      const which = c.includes('/') ? c : require('child_process').execSync(`which ${c}`, { encoding: 'utf8' }).trim();
      if (which && fs.existsSync(which)) return which;
    } catch {}
    // Try direct
    try {
      if (c === 'python3' || c === 'python') {
        const w = require('child_process').execSync(`which ${c}`, { encoding: 'utf8' }).trim();
        if (w) return w;
      }
    } catch {}
  }
  return 'python3';
}

function findSuperAgent() {
  // Bundled extraResources
  const candidates = [
    resolveResource('super_agent.py'),
    path.join(process.resourcesPath, 'super_agent.py'),
    path.join(PROJECT_ROOT, 'super_agent.py'),
    path.resolve(__dirname, '..', '..', '..', 'super_agent.py'),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  log('super_agent.py not found, candidates: ' + candidates.join(', '));
  return candidates[candidates.length - 1];
}

// ── Agent subprocess management (streaming, bounded buffer) ──

function startAgent() {
  if (agentProcess) return { alreadyRunning: true };

  const python = findPython();
  const superAgent = findSuperAgent();

  if (!fs.existsSync(superAgent)) {
    const msg = `super_agent.py not found at ${superAgent}`;
    log(msg);
    dialog.showErrorBox('LV Agent', msg);
    return { success: false, error: msg };
  }

  // Working dir: project root in dev, user home in packaged
  const cwd = isPackaged ? os.homedir() : PROJECT_ROOT;

  // Ensure userData config exists — copy from bundled example or project config if missing
  try {
    ensureDirs();
    const targetCfg = path.join(USER_DATA, 'config.yaml');
    if (!fs.existsSync(targetCfg)) {
      let srcCfg = null;
      const candidates = [
        path.join(PROJECT_ROOT, 'config.yaml'),
        resolveResource('config.yaml'),
        resolveResource('config.example.yaml'),
        BUNDLED_CONFIG_PATH,
      ];
      for (const c of candidates) { if (c && fs.existsSync(c)) { srcCfg = c; break; } }
      if (srcCfg) {
        try { fs.copyFileSync(srcCfg, targetCfg); log('Copied config ' + srcCfg + ' -> ' + targetCfg); } catch (e) { log('copy config failed: ' + e.message); }
      }
    }
  } catch (e) { log('ensure config error: ' + e.message); }

  // Env — merge, sanitize proxy, include userData
  const env = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    TRANSFORMERS_OFFLINE: '1',
    HF_HUB_OFFLINE: '1',
    CHROMADB_TELEMETRY_DISABLED: '1',
    LV_AGENT_USER_DATA: USER_DATA,
    LV_AGENT_CONFIG: path.join(USER_DATA, 'config.yaml'),
  };
  delete env.PYTHONHOME;
  delete env.PYTHONPATH;
  // Load .env from userData
  const userEnv = path.join(USER_DATA, '.env');
  if (fs.existsSync(userEnv)) {
    try {
      const lines = fs.readFileSync(userEnv, 'utf8').split('\n');
      for (const line of lines) {
        const t = line.trim();
        if (!t || t.startsWith('#')) continue;
        const idx = t.indexOf('=');
        if (idx === -1) continue;
        const k = t.slice(0, idx).trim();
        let v = t.slice(idx + 1).trim().replace(/^["']|["']$/g, '');
        if (['NIM_API_KEY','OPENAI_API_KEY','ANTHROPIC_API_KEY','OPENROUTER_API_KEY','DEEPSEEK_API_KEY','SERPAPI_KEY','TELEGRAM_BOT_TOKEN'].includes(k)) {
          env[k] = v;
        }
      }
    } catch {}
  }

  log(`Starting agent: ${python} ${superAgent} (cwd=${cwd}, packaged=${isPackaged})`);

  try {
    agentProcess = spawn(python, ['-u', superAgent], {
      cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
      env,
    });
  } catch (e) {
    log('spawn failed: ' + e.message);
    return { success: false, error: e.message };
  }

  agentInput = agentProcess.stdin;
  agentBuffer = '';
  agentChunkOffset = 0;

  // Chunked streaming — cap buffer to avoid memory bloat (keep last 200k)
  const MAX_BUFFER = 200 * 1024;
  agentProcess.stdout.on('data', (chunk) => {
    const text = chunk.toString();
    agentBuffer += text;
    if (agentBuffer.length > MAX_BUFFER) {
      // Keep tail, adjust offset
      const excess = agentBuffer.length - MAX_BUFFER;
      agentBuffer = agentBuffer.slice(excess);
      agentChunkOffset = Math.max(0, agentChunkOffset - excess);
    }
    if (mainWindow && !mainWindow.isDestroyed()) {
      // Send incremental if possible, else full
      const newChunk = agentBuffer.slice(agentChunkOffset);
      agentChunkOffset = agentBuffer.length;
      // Also send full for renderer that re-renders filtered view
      mainWindow.webContents.send('agent:output', agentBuffer);
      mainWindow.webContents.send('agent:chunk', newChunk);
    }
  });

  agentProcess.stderr.on('data', (chunk) => {
    const msg = chunk.toString();
    log('agent stderr: ' + msg.slice(0, 2000));
    // Forward non-noise stderr as toast
    if (mainWindow && !mainWindow.isDestroyed() && msg.trim()) {
      // Only forward warnings/errors, not progress spam
      if (/error|warning|failed|exception/i.test(msg) && !msg.includes('[STATUS]')) {
        mainWindow.webContents.send('agent:stderr', msg.slice(0, 4000));
      }
    }
  });

  agentProcess.on('close', (code, signal) => {
    log(`agent exited code=${code} signal=${signal}`);
    agentProcess = null;
    agentInput = null;
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('agent:exit', code);
    }
  });

  agentProcess.on('error', (err) => {
    log('agent spawn error: ' + err.message);
    dialog.showErrorBox('Agent Error', err.message + '\nCheck logs: ' + LOG_PATH);
    agentProcess = null;
    agentInput = null;
  });

  log('Agent process started pid=' + (agentProcess.pid || 'unknown'));
  return { success: true, pid: agentProcess.pid };
}

function stopAgent() {
  if (agentProcess) {
    try {
      agentProcess.kill('SIGTERM');
      // Fallback SIGKILL after 2s
      const proc = agentProcess;
      setTimeout(() => { try { if (proc && !proc.killed) proc.kill('SIGKILL'); } catch {} }, 2000);
    } catch (e) { log('kill error: ' + e.message); }
    agentProcess = null;
    agentInput = null;
    log('Agent process stopped');
  }
  return { success: true };
}

function sendAgentMessage(msg) {
  if (!agentInput || !agentProcess) return { error: 'Agent not running — click + to start' };
  try {
    // Reset buffer for new turn (keep history in renderer)
    agentBuffer = '';
    agentChunkOffset = 0;
    agentInput.write(msg + '\n');
    return { success: true };
  } catch (e) {
    log('send error: ' + e.message);
    return { error: e.message };
  }
}

// ── IPC Handlers ──

ipcMain.handle('agent:start', () => startAgent());
ipcMain.handle('agent:stop', () => stopAgent());
ipcMain.handle('agent:send', (_e, msg) => sendAgentMessage(msg));
ipcMain.handle('agent:status', () => ({ running: !!agentProcess, pid: agentProcess ? agentProcess.pid : null, packaged: isPackaged }));

ipcMain.handle('app:get-model-config', () => loadModelConfig());
ipcMain.handle('app:set-model-config', (_e, cfg) => saveModelConfig(cfg));
ipcMain.handle('app:open-log', () => {
  ensureDirs();
  // Ensure log exists
  try { if (!fs.existsSync(LOG_PATH)) fs.writeFileSync(LOG_PATH, ''); } catch {}
  return shell.openPath(LOG_PATH).then(err => {
    if (err) shell.showItemInFolder(LOG_PATH);
    return { success: !err, error: err };
  });
});
ipcMain.handle('app:get-paths', () => ({ userData: USER_DATA, logPath: LOG_PATH, configPath: getConfigPath(), projectRoot: PROJECT_ROOT, resourcesPath: isPackaged ? process.resourcesPath : PROJECT_ROOT }));
ipcMain.handle('app:open-config', () => shell.openPath(getConfigPath()).then(err => { if (err) shell.showItemInFolder(getConfigPath()); return { success: !err }; }));

// ── Artifacts: scan project root + userData for generated reports/files ──
const ARTIFACT_EXTS = ['.pdf', '.md', '.html', '.csv', '.json', '.txt', '.png', '.jpg', '.jpeg', '.svg'];
const ARTIFACT_MAX_PER_DIR = 200;

function scanArtifacts() {
  const out = [];
  const seen = new Set();
  const scanDir = (dir, label, depth = 0) => {
    if (depth > 2) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    let count = 0;
    for (const ent of entries) {
      if (count >= ARTIFACT_MAX_PER_DIR) break;
      const full = path.join(dir, ent.name);
      // Skip noise dirs
      if (ent.isDirectory()) {
        if (/node_modules|__pycache__|\.git|\.venv|dist|build|\.pytest_cache|logs/i.test(ent.name)) continue;
        scanDir(full, label, depth + 1);
        continue;
      }
      const ext = path.extname(ent.name).toLowerCase();
      if (!ARTIFACT_EXTS.includes(ext)) continue;
      // Skip session/config noise
      if (/^session_|^\.|config\.ya?ml|package\.json|package-lock/i.test(ent.name)) continue;
      if (seen.has(full)) continue;
      seen.add(full);
      count++;
      let stat;
      try { stat = fs.statSync(full); } catch { continue; }
      out.push({
        name: ent.name,
        path: full,
        ext: ext.replace('.', ''),
        dir: label,
        size: stat.size,
        mtime: stat.mtimeMs,
      });
    }
  };
  // Project root artifacts (reports, generated files)
  scanDir(PROJECT_ROOT, 'Project');
  // UserData artifacts (saved outputs)
  scanDir(USER_DATA, 'App data');
  // Sort by mtime desc
  out.sort((a, b) => b.mtime - a.mtime);
  return out.slice(0, 100);
}

ipcMain.handle('artifacts:list', () => {
  try { return { success: true, items: scanArtifacts() }; }
  catch (e) { log('artifacts:list error: ' + e.message); return { success: false, error: e.message, items: [] }; }
});

ipcMain.handle('artifacts:open', (_e, filePath) => {
  if (!filePath || typeof filePath !== 'string') return { success: false, error: 'Invalid path' };
  // Only allow opening files within PROJECT_ROOT or USER_DATA
  const isAllowed = filePath.startsWith(PROJECT_ROOT) || filePath.startsWith(USER_DATA);
  if (!isAllowed) return { success: false, error: 'Path outside allowed directories' };
  return shell.openPath(filePath).then(err => {
    if (err) { shell.showItemInFolder(filePath); return { success: false, error: err }; }
    return { success: true };
  });
});

ipcMain.handle('artifacts:reveal', (_e, filePath) => {
  if (!filePath || typeof filePath !== 'string') return { success: false, error: 'Invalid path' };
  shell.showItemInFolder(filePath);
  return { success: true };
});

// ── Messaging: Telegram bot management ──
let telegramProcess = null;

function getTelegramConfig() {
  try {
    const yaml = require('yaml');
    const cfgPath = getConfigPath();
    if (!fs.existsSync(cfgPath)) return { enabled: false, botToken: '', hasToken: false, configPath: cfgPath };
    const doc = yaml.parse(fs.readFileSync(cfgPath, 'utf8')) || {};
    // Resolve telegram section across config shapes
    let tg = (doc.agent && doc.agent.tools && doc.agent.tools.telegram) || doc.telegram || {};
    // env override
    const envToken = process.env.TELEGRAM_BOT_TOKEN || '';
    const token = envToken || tg.bot_token || '';
    return {
      enabled: !!tg.enabled,
      botToken: token,
      hasToken: !!(token && token.trim()),
      polling: tg.polling !== false,
      allowedUserIds: tg.allowed_user_ids || [],
      configPath: cfgPath,
    };
  } catch (e) {
    log('getTelegramConfig error: ' + e.message);
    return { enabled: false, botToken: '', hasToken: false, error: e.message };
  }
}

function findTelegramStarter() {
  const candidates = [
    resolveResource('start_telegram.py'),
    path.join(process.resourcesPath, 'start_telegram.py'),
    path.join(PROJECT_ROOT, 'start_telegram.py'),
    path.resolve(__dirname, '..', '..', '..', 'start_telegram.py'),
  ];
  for (const p of candidates) { if (fs.existsSync(p)) return p; }
  return candidates[candidates.length - 1];
}

function startTelegram() {
  if (telegramProcess) return { alreadyRunning: true, pid: telegramProcess.pid };
  const cfg = getTelegramConfig();
  if (!cfg.hasToken) return { success: false, error: 'No Telegram bot token configured. Set TELEGRAM_BOT_TOKEN env or telegram.bot_token in config.' };
  const python = findPython();
  const starter = findTelegramStarter();
  if (!fs.existsSync(starter)) return { success: false, error: 'start_telegram.py not found: ' + starter };
  const cwd = isPackaged ? os.homedir() : PROJECT_ROOT;
  const env = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    LV_AGENT_USER_DATA: USER_DATA,
    LV_AGENT_CONFIG: path.join(USER_DATA, 'config.yaml'),
    TELEGRAM_BOT_TOKEN: cfg.botToken,
  };
  delete env.PYTHONHOME; delete env.PYTHONPATH;
  log('Starting telegram bot: ' + python + ' ' + starter);
  try {
    telegramProcess = spawn(python, ['-u', starter], { cwd, stdio: ['pipe', 'pipe', 'pipe'], env });
  } catch (e) { return { success: false, error: e.message }; }
  telegramProcess.stdout.on('data', (c) => {
    const msg = c.toString();
    log('telegram stdout: ' + msg.slice(0, 1000));
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('telegram:log', { stream: 'stdout', text: msg });
  });
  telegramProcess.stderr.on('data', (c) => {
    const msg = c.toString();
    log('telegram stderr: ' + msg.slice(0, 1000));
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('telegram:log', { stream: 'stderr', text: msg });
  });
  telegramProcess.on('close', (code) => {
    log('telegram exited code=' + code);
    telegramProcess = null;
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('telegram:exit', code);
  });
  telegramProcess.on('error', (err) => {
    log('telegram spawn error: ' + err.message);
    telegramProcess = null;
    if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('telegram:exit', -1);
  });
  return { success: true, pid: telegramProcess.pid };
}

function stopTelegram() {
  if (!telegramProcess) return { success: true, wasRunning: false };
  try {
    telegramProcess.kill('SIGTERM');
    const proc = telegramProcess;
    setTimeout(() => { try { if (proc && !proc.killed) proc.kill('SIGKILL'); } catch {} }, 2000);
  } catch (e) { log('telegram kill error: ' + e.message); }
  telegramProcess = null;
  return { success: true, wasRunning: true };
}

ipcMain.handle('telegram:config', () => getTelegramConfig());
ipcMain.handle('telegram:start', () => startTelegram());
ipcMain.handle('telegram:stop', () => stopTelegram());
ipcMain.handle('telegram:status', () => ({ running: !!telegramProcess, pid: telegramProcess ? telegramProcess.pid : null }));

// ── Scheduled jobs ──
// Reads a lightweight jobs file in userData (jobs.json). The agent/cron system
// can write here; the desktop UI displays them. No external deps.
function getJobsFile() { return path.join(USER_DATA, 'scheduled_jobs.json'); }

function readJobs() {
  const f = getJobsFile();
  if (!fs.existsSync(f)) return [];
  try { const arr = JSON.parse(fs.readFileSync(f, 'utf8')); return Array.isArray(arr) ? arr : []; }
  catch (e) { log('readJobs error: ' + e.message); return []; }
}

function writeJobs(arr) {
  fs.mkdirSync(USER_DATA, { recursive: true });
  fs.writeFileSync(getJobsFile(), JSON.stringify(arr, null, 2));
}

ipcMain.handle('jobs:list', () => readJobs());
ipcMain.handle('jobs:add', (_e, job) => {
  const arr = readJobs();
  const id = 'job_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  const entry = {
    id,
    name: (job && job.name) || 'Untitled job',
    schedule: (job && job.schedule) || '',
    prompt: (job && job.prompt) || '',
    enabled: true,
    createdAt: Date.now(),
    lastRun: null,
  };
  arr.push(entry);
  writeJobs(arr);
  return { success: true, job: entry };
});
ipcMain.handle('jobs:toggle', (_e, id) => {
  const arr = readJobs();
  const j = arr.find(x => x.id === id);
  if (!j) return { success: false, error: 'Job not found' };
  j.enabled = !j.enabled;
  writeJobs(arr);
  return { success: true, job: j };
});
ipcMain.handle('jobs:remove', (_e, id) => {
  let arr = readJobs();
  arr = arr.filter(x => x.id !== id);
  writeJobs(arr);
  return { success: true };
});

// ── Update checker (lightweight manifest, no signing required) ──
const https = require('https');
const UPDATE_SETTINGS_FILE = path.join(USER_DATA, 'update-settings.json');
const DEFAULT_UPDATE_URL = ''; // user sets their GitHub raw URL in Settings
const UPDATE_TIMEOUT_MS = 8000;

function loadUpdateSettings() {
  const defaults = { url: DEFAULT_UPDATE_URL, enabled: true, skipVersion: null, lastCheck: 0 };
  try {
    if (fs.existsSync(UPDATE_SETTINGS_FILE)) {
      const obj = JSON.parse(fs.readFileSync(UPDATE_SETTINGS_FILE, 'utf8'));
      return { ...defaults, ...obj };
    }
  } catch (e) { log('loadUpdateSettings error: ' + e.message); }
  return defaults;
}

function saveUpdateSettings(patch) {
  try {
    const cur = loadUpdateSettings();
    const next = { ...cur, ...patch };
    fs.writeFileSync(UPDATE_SETTINGS_FILE, JSON.stringify(next, null, 2));
    return next;
  } catch (e) { log('saveUpdateSettings error: ' + e.message); return loadUpdateSettings(); }
}

// Semver compare: returns 1 if a>b, -1 if a<b, 0 if equal. Tolerates pre-release tags (ignores them).
function compareSemver(a, b) {
  const parse = (v) => {
    const m = String(v || '').trim().replace(/^v/i, '').match(/^(\d+)\.(\d+)\.(\d+)/);
    return m ? [parseInt(m[1], 10), parseInt(m[2], 10), parseInt(m[3], 10)] : [0, 0, 0];
  };
  const pa = parse(a), pb = parse(b);
  for (let i = 0; i < 3; i++) {
    if (pa[i] > pb[i]) return 1;
    if (pa[i] < pb[i]) return -1;
  }
  return 0;
}

function fetchJson(url, timeoutMs) {
  return new Promise((resolve, reject) => {
    let req;
    try {
      req = https.get(url, { headers: { 'User-Agent': 'LV-Agent-Desktop/' + app.getVersion() } }, (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          res.resume();
          fetchJson(res.headers.location, timeoutMs).then(resolve, reject);
          return;
        }
        if (res.statusCode !== 200) {
          res.resume();
          reject(new Error('HTTP ' + res.statusCode));
          return;
        }
        let data = '';
        res.on('data', (c) => { data += c; if (data.length > 1024 * 1024) { req.destroy(); reject(new Error('manifest too large')); } });
        res.on('end', () => {
          try { resolve(JSON.parse(data)); } catch (e) { reject(new Error('invalid JSON: ' + e.message)); }
        });
      });
    } catch (e) { reject(e); return; }
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => { req.destroy(new Error('timeout')); });
  });
}

async function checkForUpdates(explicitUrl) {
  const settings = loadUpdateSettings();
  const url = (explicitUrl && explicitUrl.trim()) || settings.url;
  if (!url) return { hasUpdate: false, reason: 'no-url', currentVersion: app.getVersion() };
  try {
    const manifest = await fetchJson(url, UPDATE_TIMEOUT_MS);
    const latest = manifest.latestVersion || manifest.version;
    if (!latest) return { hasUpdate: false, reason: 'manifest-missing-version', currentVersion: app.getVersion() };
    const current = app.getVersion();
    const hasUpdate = compareSemver(latest, current) > 0;
    const skipped = settings.skipVersion && compareSemver(latest, settings.skipVersion) === 0;
    const minVersion = manifest.minVersion || null;
    const forced = minVersion && compareSemver(current, minVersion) < 0;
    saveUpdateSettings({ lastCheck: Date.now() });
    return {
      hasUpdate,
      forced: !!forced,
      suppressed: !explicitUrl && skipped && !forced,
      currentVersion: current,
      latestVersion: latest,
      downloadUrl: manifest.downloadUrl || manifest.url || '',
      releaseNotes: manifest.releaseNotes || manifest.notes || '',
      minVersion,
      manifestUrl: url,
    };
  } catch (e) {
    log('checkForUpdates error: ' + e.message);
    return { hasUpdate: false, reason: 'error', error: e.message, currentVersion: app.getVersion() };
  }
}

ipcMain.handle('updates:check', (_e, url) => checkForUpdates(url));
ipcMain.handle('updates:get-settings', () => loadUpdateSettings());
ipcMain.handle('updates:set-url', (_e, url) => { saveUpdateSettings({ url: (url || '').trim() }); return { success: true }; });
ipcMain.handle('updates:toggle-enabled', (_e, enabled) => { saveUpdateSettings({ enabled: !!enabled }); return { success: true }; });
ipcMain.handle('updates:skip-version', (_e, version) => { saveUpdateSettings({ skipVersion: version || null }); return { success: true }; });

// ── App lifecycle ──

app.whenReady().then(() => {
  ensureDirs();
  createWindow();
  log(`App ready v${app.getVersion()} packaged=${isPackaged} userData=${USER_DATA}`);
  // Auto-start agent? No — let user click +. But ensure health check
});

app.on('window-all-closed', () => {
  stopAgent();
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on('before-quit', () => {
  stopAgent();
  stopTelegram();
});

// Graceful handling of GPU etc.
app.on('render-process-gone', (_e, _wc, details) => {
  log('render-process-gone: ' + JSON.stringify(details));
});
