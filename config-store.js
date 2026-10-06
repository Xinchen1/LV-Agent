const fs = require('fs');
const path = require('path');
const os = require('os');
const log = require('electron-log');
const secrets = require('./secret-resolver');

function getUserDataPath() {
  try {
    const { app } = require('electron');
    if (app && app.getPath) {
      return app.getPath('userData');
    }
  } catch (e) {
    // Electron not available (e.g. tests or incomplete install)
  }
  return path.join(os.homedir(), '.superide');
}

const configPath = path.join(getUserDataPath(), 'config.json');

// 统一模型清单：cancri(AMD DeepSeek)、deepseek(官方 API)、openrouter(免费模型)、ngrok(自托管)、opencode-go / opencode-zen（OpenCode 官方网关）
// API Key / 隧道 URL 一律经 secret-resolver 解析（env → secrets.json → 内置兜底），源码不再散落明文密钥；
// opencode-go / opencode-zen 的 key 由 getAllModels() 运行时从 ~/.local/share/opencode/auth.json 回填。
const AVAILABLE_MODELS = [
  // ===== cancri（AMD DeepSeek，默认模型）=====
  { provider: 'cancri', name: 'cancri-fast-v1', apiName: 'DeepSeek-V4.1-Flash', displayName: 'cancri-fast', apiKey: secrets.getCancriKey(), endpoint: (secrets.getCancriBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 16384, top_p: 0.95 },

  // ===== superide-key-pool（自建算力池网关：多密钥轮换/额度记账/熔断都在网关侧）=====
  { provider: 'superide-key-pool', name: 'key-pool-fast', apiName: 'DeepSeek-V4.1-Flash', displayName: 'Key Pool 算力池', apiKey: secrets.getKeyPoolToken(), endpoint: (secrets.getKeyPoolUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 16384, top_p: 0.95 },

  // ===== deepseek（DeepSeek 官方 API 端点，deepseek-flash = V4.1 Flash）=====
  { provider: 'deepseek', name: 'deepseek-flash', apiName: 'deepseek-flash', displayName: 'DeepSeek V4.1 Flash', apiKey: secrets.getDeepSeekKey(), endpoint: 'https://api.deepseek.com/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },

  // ===== openrouter（OpenRouter 免费模型）=====
  { provider: 'openrouter', name: 'deepseek/deepseek-v4-flash-0731:free', displayName: 'DeepSeek V4 Flash (Free)', apiKey: secrets.getOpenRouterKey(), endpoint: 'https://openrouter.ai/api/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'openrouter', name: 'qwen/qwen3.8-27b:free', displayName: 'Qwen3.8 27B (Free)', apiKey: secrets.getOpenRouterKey(), endpoint: 'https://openrouter.ai/api/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'openrouter', name: 'z-ai/glm-5.2:free', displayName: 'GLM-5.2 (Free)', apiKey: secrets.getOpenRouterKey(), endpoint: 'https://openrouter.ai/api/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'openrouter', name: 'google/gemma-4-26b-a4b-it:free', displayName: 'Gemma 4 26B (Free)', apiKey: secrets.getOpenRouterKey(), endpoint: 'https://openrouter.ai/api/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },

  { provider: 'local-ollama', name: 'fredrezones55/Qwopus3.5', apiName: 'fredrezones55/Qwopus3.5', displayName: 'Qwopus3.5 (本地 Ollama)', apiKey: 'ollama', endpoint: 'http://localhost:11434/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'local-ollama', name: 'Qwopus3.5-9B-Coder', apiName: 'Qwopus3.5-9B-Coder', displayName: 'Qwopus3.5 9B Coder (本地 Ollama)', apiKey: 'ollama', endpoint: 'http://localhost:11434/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  // ===== ngrok（自托管 Ollama，无需 key；URL 经 secret-resolver 解析）=====
  { provider: 'ngrok', name: 'qwen3:4b', displayName: 'Qwen3 4B (ngrok自托管)', apiKey: '', endpoint: (secrets.getNgrokBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 4096, top_p: 0.95 },
  { provider: 'ngrok', name: 'qwen3:1.7b', displayName: 'Qwen3 1.7B (ngrok自托管 · 极速)', apiKey: '', endpoint: (secrets.getNgrokBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 4096, top_p: 0.95 },
  { provider: 'ngrok', name: 'llama3.2:3b', displayName: 'Llama 3.2 3B (ngrok自托管 · 工具)', apiKey: '', endpoint: (secrets.getNgrokBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 4096, top_p: 0.95 },
  { provider: 'ngrok', name: 'qwen2.5:3b', displayName: 'Qwen2.5 3B (ngrok自托管 · 工具)', apiKey: '', endpoint: (secrets.getNgrokBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 4096, top_p: 0.95 },
  { provider: 'ngrok', name: 'deepseek-r1:7b', displayName: 'DeepSeek R1 7B (ngrok自托管 · 推理)', apiKey: '', endpoint: (secrets.getNgrokBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 4096, top_p: 0.95 },
  { provider: 'ngrok', name: 'deepseek-r1:1.5b', displayName: 'DeepSeek R1 1.5B (ngrok自托管 · 推理/极速)', apiKey: '', endpoint: (secrets.getNgrokBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 4096, top_p: 0.95 },
  { provider: 'ngrok', name: 'phi3.5:latest', displayName: 'Phi-3.5 3.8B (ngrok自托管)', apiKey: '', endpoint: (secrets.getNgrokBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 4096, top_p: 0.95 },
  { provider: 'ngrok', name: 'minicpm3:4b', displayName: 'MiniCPM3 4B (ngrok自托管 · 长文本)', apiKey: '', endpoint: (secrets.getNgrokBaseUrl().replace(/\/+$/, '') + '/chat/completions'), temperature: 0.7, maxTokens: 4096, top_p: 0.95 },

  // ===== opencode-go（OpenCode Go · 订阅内模型，Chat 直连 / Composer 双模式可用）=====
  // apiKey 留空 → getAllModels() 从 ~/.local/share/opencode/auth.json 的 opencode-go 读取
  { provider: 'opencode-go', name: 'deepseek-v4-flash', displayName: 'DeepSeek V4 Flash', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'deepseek-v4-flash-vision-exp', displayName: 'DeepSeek V4 Flash Vision Exp', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'deepseek-v4-pro', displayName: 'DeepSeek V4 Pro', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'deepseek-v4.1-flash', displayName: 'DeepSeek V4.1 Flash', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'glm-5.1', displayName: 'GLM-5.1', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'glm-5.2', displayName: 'GLM-5.2', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'glm-5.3', displayName: 'GLM-5.3', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'glm-5.3-flash', displayName: 'GLM-5.3-Flash', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'hy3', displayName: 'Hy3', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'hy4-preview', displayName: 'Hy4 preview', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'kimi-k2.6', displayName: 'Kimi K2.6', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'kimi-k2.7-code', displayName: 'Kimi K2.7 Code', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'kimi-k3', displayName: 'Kimi K3', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'longcat-2.0', displayName: 'LongCat-2.0', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'mimo-v2.5', displayName: 'MiMo V2.5', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'mimo-v2.5-pro', displayName: 'MiMo V2.5 Pro', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'mimo-v2.6-flash', displayName: 'MiMo V2.6 Flash', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'mimo-v2.6-pro', displayName: 'MiMo V2.6 Pro', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  // 注：minimax-m2.5 / omen-alpha 未收录——Chat 直连可用但 opencode 注册表已下架（Composer 走它会 500）
  { provider: 'opencode-go', name: 'minimax-m3', displayName: 'MiniMax-M3', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'qwen3.6-plus', displayName: 'Qwen3.6 Plus', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'qwen3.7-max', displayName: 'Qwen3.7 Max', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'qwen3.7-plus', displayName: 'Qwen3.7 Plus', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'qwen3.8-flash', displayName: 'Qwen3.8 Flash', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'qwen3.8-max', displayName: 'Qwen3.8 Max', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },
  { provider: 'opencode-go', name: 'space-bunny-free', displayName: 'Space Bunny (Free)', apiKey: '', endpoint: 'https://opencode.ai/zen/go/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95 },

  // ===== opencode-zen（OpenCode Zen 免费档 · 仅 Composer 可用：Chat 直连被 FreeTierError 拒绝）=====
  // apiKey 留空 → 从 auth.json 的 opencode-zen 读取；opencode 内部 providerID 为 'opencode'
  { provider: 'opencode-zen', name: 'big-pickle', displayName: 'Big Pickle (Zen 免费 · Composer)', apiKey: '', endpoint: 'https://opencode.ai/zen/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95, composerOnly: true },
  { provider: 'opencode-zen', name: 'ling-3.0-flash-fin-free', displayName: 'Ling 3.0 Flash Fin (Free · Composer)', apiKey: '', endpoint: 'https://opencode.ai/zen/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95, composerOnly: true },
  { provider: 'opencode-zen', name: 'mimo-v2.6-flash-free', displayName: 'MiMo V2.6 Flash (Free · Composer)', apiKey: '', endpoint: 'https://opencode.ai/zen/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95, composerOnly: true },
  { provider: 'opencode-zen', name: 'nemotron-3-ultra-free', displayName: 'Nemotron 3 Ultra (Free · Composer)', apiKey: '', endpoint: 'https://opencode.ai/zen/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95, composerOnly: true },
  { provider: 'opencode-zen', name: 'nemotron-3.5-lightning-free', displayName: 'Nemotron 3.5 Lightning (Free · Composer)', apiKey: '', endpoint: 'https://opencode.ai/zen/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95, composerOnly: true },
  { provider: 'opencode-zen', name: 'space-bunny-free', displayName: 'Space Bunny (Free · Composer)', apiKey: '', endpoint: 'https://opencode.ai/zen/v1/chat/completions', temperature: 0.7, maxTokens: 16384, top_p: 0.95, composerOnly: true },
];

const defaultConfig = {
  model: { ...AVAILABLE_MODELS[0] },
  customModels: [],
  // Moonwalker 语义检索的嵌入/向量存储配置
  embeddings: {
    enabled: true,
    // 'cloudflare' = 远程 Workers AI 嵌入 + Vectorize 向量存储（向量存 Cloudflare）；
    // 'local'      = 使用下方自带的 OpenAI 兼容嵌入端点（本地/第三方）
    mode: 'cloudflare',
    cloudflare: {
      accountId: secrets.getCfAccountId(),
      proxyUrl: secrets.getCfProxyUrl(),
      index: 'superide-vectors',
      model: '@cf/baai/bge-m3',
      token: secrets.getCfApiToken() // 留空则用环境变量 CLOUDFLARE_API_TOKEN 或 ~/.wrangler 的 OAuth token
    },
    // 仅 mode='local' 时使用
    provider: 'nvidia',
    model: 'nvidia/nemotron-3-embed-1b',
    endpoint: 'https://integrate.api.nvidia.com/v1/embeddings',
    apiKey: ''
  },
  webSearch: {
    provider: 'google',
    apiKey: ''
  },
  ui: { theme: 'dark', language: 'zh-CN' },
  performance: {
    streamingEnabled: true,
    requestTimeout: 60000,
    retryAttempts: 2
  },
  network: {
    useSystemProxy: false
  }
};

function ensureConfigDir() {
  try {
    const dir = path.dirname(configPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  } catch (e) {
    log.error('Failed to create config dir:', e);
  }
}

function loadOpencodeAuth(provider = 'cancri') {
  try {
    const authPath = path.join(os.homedir(), '.local', 'share', 'opencode', 'auth.json');
    if (!fs.existsSync(authPath)) return null;
    const data = JSON.parse(fs.readFileSync(authPath, 'utf8'));
    const entry = data[provider] || data.providers?.find(p => p.provider === provider);
    return entry?.apiKey || entry?.key || null;
  } catch (e) {
    return null;
  }
}

function loadConfig() {
  try {
    let config = defaultConfig;
    if (fs.existsSync(configPath)) {
      const data = fs.readFileSync(configPath, 'utf8');
      config = { ...defaultConfig, ...JSON.parse(data) };
    }
    // 迁移：确保 webSearch 配置完整且默认使用免费 Google 搜索
    const validSearchProviders = ['google', 'bing', 'duckduckgo', 'serper'];
    const savedProvider = config.webSearch?.provider;
    const needsGoogleMigration = !savedProvider ||
      !validSearchProviders.includes(savedProvider) ||
      (savedProvider === 'serper' && !config.webSearch?.apiKey);
    if (needsGoogleMigration) {
      config.webSearch = { ...defaultConfig.webSearch, ...(config.webSearch || {}) };
      if (!validSearchProviders.includes(config.webSearch.provider)) {
        config.webSearch.provider = 'google';
      }
      saveConfig(config);
      log.info('[ConfigStore] Migrated webSearch config to free Google provider');
    }

    // 迁移：旧版 cancri 配置升级到 AMD DeepSeek-V4.1-Flash 默认配置（保留用户已保存的 key）
    if (config.model && (config.model.provider === 'cancri' || config.model.name === 'cancri-fast' || config.model.name === 'cancri-fast-v1')) {
      const savedKey = config.model.apiKey || '';
      config.model = {
        ...AVAILABLE_MODELS[0],
        ...(savedKey ? { apiKey: savedKey } : {})
      };
      saveConfig(config);
      log.info('[ConfigStore] Migrated cancri config to AMD DeepSeek-V4.1-Flash default');
    }
    // 如果配置中缺少 API key，尝试从 OpenCode auth.json 读取
    if (config.model && !config.model.apiKey) {
      const fallbackKey = loadOpencodeAuth(config.model.provider || 'cancri');
      if (fallbackKey) {
        config.model = { ...config.model, apiKey: fallbackKey };
      }
    }
    return config;
  } catch (e) {
    log.error('Failed to load config:', e);
  }
  return defaultConfig;
}

function saveConfig(config) {
  try {
    ensureConfigDir();
    fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
  } catch (e) {
    log.error('Failed to save config:', e);
  }
}

/**
 * 获取当前模型配置
 * 默认 cancri-fast 即 AMD Radeon DeepSeek-V4.1-Flash
 */
function getModelConfig() {
  const config = loadConfig().model;
  if (config.provider === 'cancri' || config.name === 'cancri-fast' || config.name === 'cancri-fast-v1') {
    return { ...AVAILABLE_MODELS[0], ...(config.apiKey ? { apiKey: config.apiKey } : {}) };
  }
  if (config.provider === 'ngrok') {
    const ngrokConfig = AVAILABLE_MODELS.find(m => m.provider === 'ngrok' && m.name === config.name);
    if (ngrokConfig) {
      return { ...ngrokConfig, ...config };
    }
  }
  return config;
}

function getCustomModels() {
  return loadConfig().customModels || [];
}

function getAllModels() {
  // 空 apiKey 的模型（如 ngrok）运行时从 OpenCode auth.json 补全，避免在源码中硬编码密钥
  return [...AVAILABLE_MODELS, ...getCustomModels()].map((m) => {
    if (!m || m.apiKey) return m;
    const key = loadOpencodeAuth(m.provider);
    return key ? { ...m, apiKey: key } : m;
  });
}

/**
 * 基于 JSON 文件的简易键值存储
 */
const store = {
  get: (key) => {
    const config = loadConfig();
    return key.split('.').reduce((obj, k) => obj?.[k], config);
  },
  set: (key, value) => {
    const config = loadConfig();
    const keys = key.split('.');
    let obj = config;
    for (let i = 0; i < keys.length - 1; i++) {
      if (!obj[keys[i]]) obj[keys[i]] = {};
      obj = obj[keys[i]];
    }
    obj[keys[keys.length - 1]] = value;
    saveConfig(config);
  }
};

// ---------- Agent 模板 ----------

const DEFAULT_AGENT_TEMPLATES = [
  {
    id: 'code-expert',
    name: '代码专家',
    icon: '◉',
    description: '专注于代码实现、优化和重构',
    systemPrompt: 'You are an expert software engineer. Focus on writing clean, efficient, and maintainable code. Always provide reasoning before acting.',
    model: null, // 使用当前选中模型
    temperature: 0.7,
    tools: ['file_read', 'file_write', 'file_list', 'terminal_execute', 'search_symbols'],
  },
  {
    id: 'architect',
    name: '架构师',
    icon: '◈',
    description: '负责系统架构设计和技术选型',
    systemPrompt: 'You are a senior software architect. Focus on system design, modularity, scalability, and separation of concerns. Provide architectural recommendations with clear rationale.',
    model: null,
    temperature: 0.7,
    tools: ['file_read', 'file_list', 'search_symbols'],
  },
  {
    id: 'debugger',
    name: '调试专家',
    icon: '◎',
    description: '擅长错误诊断和修复',
    systemPrompt: 'You are an expert debugger. Focus on identifying root causes of bugs, providing clear explanations, and implementing minimal fixes. Always verify fixes with tests.',
    model: null,
    temperature: 0.5,
    tools: ['file_read', 'file_write', 'terminal_execute', 'search_symbols', 'run_tests', 'run_lint'],
  },
  {
    id: 'reviewer',
    name: '代码审查',
    icon: '◆',
    description: '代码质量和规范检查',
    systemPrompt: 'You are a senior code reviewer. Focus on code quality, security vulnerabilities, performance issues, and adherence to best practices. Provide constructive feedback.',
    model: null,
    temperature: 0.3,
    tools: ['file_read', 'file_list', 'search_symbols', 'run_lint'],
  },
  {
    id: 'react-dev',
    name: 'React 开发',
    icon: '⚛',
    description: 'React/Next.js 前端开发专家',
    systemPrompt: 'You are a React/Next.js expert. Focus on component architecture, hooks, state management, performance optimization, and modern React patterns. Use TypeScript when possible.',
    model: null,
    temperature: 0.7,
    tools: ['file_read', 'file_write', 'file_list', 'terminal_execute', 'search_symbols'],
  },
  {
    id: 'devops',
    name: 'DevOps',
    icon: '⚙',
    description: 'CI/CD、部署和基础设施',
    systemPrompt: 'You are a DevOps engineer. Focus on CI/CD pipelines, containerization, infrastructure as code, monitoring, and deployment automation. Prefer Docker, Kubernetes, and cloud-native solutions.',
    model: null,
    temperature: 0.5,
    tools: ['file_read', 'file_write', 'file_list', 'terminal_execute'],
  },
];

function getAgentTemplates() {
  const config = loadConfig();
  return config.agentTemplates || DEFAULT_AGENT_TEMPLATES;
}

function saveAgentTemplate(template) {
  const config = loadConfig();
  const templates = config.agentTemplates || [...DEFAULT_AGENT_TEMPLATES];
  const idx = templates.findIndex(t => t.id === template.id);
  if (idx >= 0) {
    templates[idx] = { ...templates[idx], ...template };
  } else {
    templates.push(template);
  }
  config.agentTemplates = templates;
  saveConfig(config);
  return template;
}

function deleteAgentTemplate(templateId) {
  const config = loadConfig();
  const templates = (config.agentTemplates || [...DEFAULT_AGENT_TEMPLATES]).filter(t => t.id !== templateId);
  config.agentTemplates = templates;
  saveConfig(config);
}

module.exports = {
  configPath,
  AVAILABLE_MODELS,
  defaultConfig,
  loadConfig,
  saveConfig,
  getModelConfig,
  getCustomModels,
  getAllModels,
  store,
  getAgentTemplates,
  saveAgentTemplate,
  deleteAgentTemplate,
};
