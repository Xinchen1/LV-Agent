# 1. 退出当前代理或在网络允许的环境中操作
unset ALL_PROXY ALL_HTTP_PROXY HTTPS_PROXY

# 2. 登录 Cloudflare
wrangler login

# 3. 部署
cd /Users/mac/Desktop/agent_project/cloudflare-worker
wrangler deploy

# 4. 设置环境变量 (在 Cloudflare dashboard 或 wrangler.toml)
#    或使用环境变量传递: wrangler secret put OPENAI_API_KEY