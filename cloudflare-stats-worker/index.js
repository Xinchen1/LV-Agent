// LV Agent 匿名心跳统计 Worker
// POST /api/version  (心跳上报: {iid, v, platform, ts, uptime_s})
// GET  /stats         (查看统计, 需带 ?key=ADMIN_KEY)

const DAY = 86400 * 1000;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/version" && request.method === "POST") {
      let body;
      try {
        body = await request.json();
      } catch {
        return new Response("bad json", { status: 400 });
      }
      const iid = String(body.iid || "").slice(0, 64);
      if (!iid) return new Response("missing iid", { status: 400 });
      const key = `iid:${iid}`;
      const now = Date.now();
      const prev = await env.STATS.get(key, "json");
      const record = {
        first_seen: prev?.first_seen || now,
        last_seen: now,
        v: String(body.v || "unknown").slice(0, 32),
        platform: String(body.platform || "unknown").slice(0, 32),
        beats: (prev?.beats || 0) + 1,
      };
      await env.STATS.put(key, JSON.stringify(record));
      return Response.json({ ok: true, ts: now });
    }

    if (url.pathname === "/stats") {
      const adminKey = env.ADMIN_KEY || "";
      if (adminKey && url.searchParams.get("key") !== adminKey) {
        return new Response("forbidden", { status: 403 });
      }
      const now = Date.now();
      let total = 0, active24h = 0, active7d = 0;
      const versions = {}, platforms = {};
      let cursor;
      do {
        const list = await env.STATS.list({ prefix: "iid:", cursor });
        for (const k of list.keys) {
          total++;
          const rec = await env.STATS.get(k.name, "json");
          if (!rec) continue;
          if (now - rec.last_seen < DAY) active24h++;
          if (now - rec.last_seen < 7 * DAY) active7d++;
          versions[rec.v] = (versions[rec.v] || 0) + 1;
          platforms[rec.platform] = (platforms[rec.platform] || 0) + 1;
        }
        cursor = list.list_complete ? undefined : list.cursor;
      } while (cursor);
      return Response.json({ total_installs: total, active_24h: active24h, active_7d: active7d, versions, platforms });
    }

    return new Response("lv-agent-stats ok");
  },
};
