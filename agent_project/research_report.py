"""
Research Report Generator - Deep research + Markdown output workflow.

Handles requests like:
  "搜索 XXX 并生成深度调研报告"
  "调研 XXX, 输出 markdown 文件"
  "查一下 XXX, 写一份报告保存"

Flow:
  1. Extract research topic from the user's task.
  2. Run 2-3 complementary web searches in parallel.
  3. Enrich top results by fetching article bodies.
  4. Generate a structured Markdown report with LLM.
  5. Save the report to ~/OpenMythos/reports/<slug>_<date>.md.
  6. Return the file path and a short summary.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evidence import EvidenceLedger, claim_tokens, _data_numbers
from .tools import TOOLS_REGISTRY, ToolResult
from .tools.search_cache import SearchCache


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def is_research_report_task(task: str) -> bool:
    """Return True if the user wants a research report / markdown file output."""
    if not task:
        return False
    t = task.lower()

    research_verbs = [
        "搜索", "搜一下", "搜", "查找", "查一下", "查", "调研", "研究",
        "深度研究", "深度调研", "研究下", "调研下", "研究一下", "调研一下",
        "research", "search", "look up", "investigate",
    ]
    report_indicators = [
        "报告", "调研报告", "研究报告", "深度报告", "分析",
        "report", "analysis", "study", "overview",
    ]
    output_indicators = [
        "输出", "生成", "写", "保存", "写到", "导出", "md", "markdown",
        "文件", "output", "generate", "save", "write", "export",
    ]

    has_verb = any(v in t for v in research_verbs)
    has_report = any(r in t for r in report_indicators)
    has_output = any(o in t for o in output_indicators)

    # Explicit "output/save/write md" triggers research workflow even without "report"
    explicit_file = "md" in t or "markdown" in t or "输出文件" in t or "保存" in t or "写到" in t

    # 深度研究/调研类请求: "深度研究 X" 本身即隐含生成报告意图, 无需出现"报告/分析"
    # (但纯"搜索/查一下"问句如"帮我查一下天气"不应触发)
    deep_prefix = re.match(r"^(深度研究|深度调研|调研一下|研究一下|研究下|调研下).*", t, re.IGNORECASE)
    if deep_prefix and has_verb and len(t.strip()) > 6:
        return True

    return has_verb and (has_report or explicit_file)


def extract_research_topic(task: str) -> str:
    """Extract the core research topic from a request like '搜索 实在智能 并生成报告'.

    Preserves the full entity phrase (e.g. '实在智能的产品技术') instead of stopping
    at the first '的' or comma.
    """
    if not task:
        return ""

    # Strip common wrappers
    t = task.strip()
    for wrapper in ['"', "'", "“", "”", "‘", "’", "()", "（）"]:
        t = t.strip(wrapper)

    # Primary: capture from research verb until a report/output boundary.
    # The lookahead stops at report/action verbs or sentence end, NOT at '的'.
    boundary = r"(?:并|且|然后|接着|再|顺便|给我|帮我|为|给|输出|生成|写|保存|导出|export|save|write|generate|create)?\s*(?:报告|调研报告|研究报告|深度报告|分析|文件|md|markdown|output|report|study|analysis|overview)?\s*(?:$|[。；;，,、！!?？]|\n)"
    patterns = [
        r"(?:搜索|搜一下|搜|查找|查一下|查|调研|研究|search|look up|research)\s*(?:关于|有关|一下|下)?[\s:：，,、。]*(.+?)(?=" + boundary + r")",
        r"(?:调研|研究|分析|总结|investigate|analyze|summarise|summarize)\s*(?:关于|有关|一下|下)?[\s:：，,、。]*(.+?)(?=" + boundary + r")",
    ]
    for pat in patterns:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            topic = m.group(1).strip(" ，,。.、:：;")
            topic = _strip_trailing_fluff(topic)
            topic = _strip_report_suffixes(topic)
            topic = re.sub(r"\s+", " ", topic).strip()
            if len(topic) >= 2:
                return topic

    # Fallback: remove leading verbs and trailing output/report words
    head_verbs = r"^(搜索|搜一下|搜|查找|查一下|查|调研|研究|分析|总结|search|look up|research|investigate|analyze|summarise|summarize)[\s:：，,、。的]*"
    cleaned = re.sub(head_verbs, "", t, flags=re.IGNORECASE).strip()
    cleaned = _strip_trailing_fluff(cleaned)
    cleaned = _strip_report_suffixes(cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if len(cleaned) >= 2:
        return cleaned
    return t.strip()[:80]


def _strip_report_suffixes(text: str) -> str:
    tails = [
        "的信息", "的情况", "的内容", "的资料", "的数据",
        "生成报告", "生成调研报告", "生成研究报告", "生成深度报告",
        "写报告", "写调研报告", "写研究报告", "写深度报告",
        "输出报告", "输出文件", "输出md", "输出markdown",
        "保存报告", "保存文件", "保存md", "保存markdown",
        "的报告", "的调研报告", "的研究报告", "的深度报告",
        "并生成", "并输出", "并保存", "并写",
        "给我", "帮我", "一下", "详细", "深入",
    ]
    changed = True
    t = text.strip()
    while changed:
        changed = False
        for tail in tails:
            if t.endswith(tail):
                t = t[:-len(tail)].rstrip(" ，,。.、:：")
                changed = True
    return t.strip('"""\'\'\'"\'')


# ---------------------------------------------------------------------------
# Query grounding helpers (shared with agent.py)
# ---------------------------------------------------------------------------

def _strip_generic_search_suffixes(text: str) -> str:
    """Remove empty phrase tails without touching the core entity."""
    tails = [
        '的情况', '的内容', '的信息', '的最新信息', '的详细信息',
        '的报告', '的调研报告', '的研究报告', '的深度报告', '的分析报告',
        '的资料', '的最新消息', '的新闻', '的介绍', '的详细介绍',
        '怎么样', '如何', '怎么用', '是什么', '什么是',
        '一下', '下', '详细点', '详细', '深入', '尽可能多',
        '的资料', '的产品技术', '的技术', '的产品', '的公司', '的企业',
        '相关信息', '相关', '的相关', '大全', '汇总', '总结',
    ]
    changed = True
    t = text.strip()
    while changed:
        changed = False
        for tail in tails:
            if t.endswith(tail):
                t = t[:-len(tail)].rstrip(' ，,。.、:：')
                changed = True
    return t.strip(' ，,。.、:：；;！？!?')


def _strip_trailing_fluff(text: str) -> str:
    """Remove trailing conjunctions and dangling output verbs left over after capture."""
    t = text.strip()
    patterns = [
        r'(?:^|[\s,，.。;；:：])(?:并|并且|然后|接着|再|顺便|给我|帮我|为|给)\s*$',
        r'\s+(?:输出|生成|写|保存|导出|export|save|write|generate|create)\s*$',
        r'\s+(?:深度|详细|深入)\s*$',
    ]
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            new_t = re.sub(pat, '', t).strip(" ，,。.、:：;")
            if new_t != t:
                t = new_t
                changed = True
    # Final safety: a lone trailing '并' after stripping output verbs is a conjunction.
    while len(t) > 1 and t.endswith('并'):
        t = t[:-1].rstrip(" ，,。.、:：;")
    return t


def extract_search_keywords(task: str) -> str:
    """从用户原始任务中抽取出真正想搜的关键词字符串。"""
    if not task:
        return ""
    t = task.strip()
    # 去除系统提示拼接的元指令前缀
    import re
    t = re.sub(r'^继续执行上一条任务[:：]\s*', '', t)
    t = re.sub(r'^重执.*?:\s*', '', t)
    # 去除括号里的元指令，如 (用户已同意...若未完成则...)
    t = re.sub(r'\s*\([^)]*(?:用户已同意|若已完成|若未完成|若已|若未|确认结果|现在真正执行并给出结果)[^)]*\)', '', t)
    t = re.sub(r'\s*[（(][^）)]*(?:同意|确认|完成|执行|给出结果)[^）)]*[）)]', '', t)
    # 去除多余的英文提示括号
    t = re.sub(r'\s*\([^)]*\)', '', t)
    t = t.strip()

    verb_patterns = [
        r'(?:帮我搜一下|帮我搜索|帮我查找|帮我查|帮我找一下|帮我找|帮我研究|帮我调研)',
        r'(?:搜索一下|搜一下|查一下|找一下|研究一下|调研一下|了解一下)',
        r'(?:搜索|搜下|搜|查找|查下|查查|查询|找找|找|调研|研究)',
    ]
    for vb in verb_patterns:
        m = re.search(vb + r'\s*(?:关于|有关|下|下)[\s:：，,、。.]*', t)
        if not m:
            m = re.search(vb + r'[\s:：，,、。.]*', t)
        if m:
            rest = t[m.end():].strip()
            # Stop at conjunctions/verbs that clearly begin the output/report clause
            rest = re.split(r'[。；;\n!?！？]|\s*(?:并|然后|接着|再|顺便|给我|帮我|输出|生成|写|保存|export|save|write)', rest, maxsplit=1)[0].strip(' ，,、:：')
            if rest:
                rest = rest.strip('"\'""''()（）')
                kw = _strip_generic_search_suffixes(rest)
                kw = kw[:80].strip()
                if 1 < len(kw):
                    return kw

    for m in re.finditer(r'"([^"]{2,60})"|“([^”]{2,60})”|‘([^’]{2,60})’', t):
        kw = next(g for g in m.groups() if g)
        kw = _strip_generic_search_suffixes(kw)
        if 1 < len(kw) <= 60:
            return kw

    head_strip = r'^(帮我搜一下|帮我搜索|帮我查找|帮我查|帮我找一下|帮我找|帮我研究|帮我调研|搜索一下|搜一下|查一下|找一下|研究一下|调研一下|了解一下|搜索|搜|查找|查下|查查|查询|找找|找|调研|研究)[\s:：，,、。.的]*'
    cleaned = re.sub(r'\s{2,}', ' ', _strip_generic_search_suffixes(t)).strip()
    cleaned = re.sub(head_strip, '', cleaned, count=1).strip()
    # Drop trailing output/report clause remnants
    cleaned = re.split(r'\s*(?:并|然后|接着|再|顺便|给我|帮我|输出|生成|写|保存|export|save|write)', cleaned, maxsplit=1)[0].strip(' ，,、:：')
    if 2 <= len(cleaned) <= 80:
        return cleaned
    return ""


def extract_search_keywords_hybrid(task: str, *, use_llm: bool = False) -> str:
    """
    混合抽取：先用规则清洗，若结果可疑则回退到 LLM 抽取。
    可疑信号：长度过短/过长、包含元指令关键词、仍含括号。
    """
    import re
    rule_kw = extract_search_keywords(task)
    meta_signals = ['用户已同意', '若已完成', '若未完成', '确认结果', '继续执行', '现在真正执行']
    # 可疑判断
    suspicious = (
        not rule_kw or
        len(rule_kw) < 2 or
        any(sig in rule_kw for sig in meta_signals) or
        '(' in rule_kw or ')' in rule_kw
    )
    if use_llm and suspicious:
        try:
            # 轻量 LLM 抽取，单次调用
            from .model_backends import get_backend
            backend = get_backend()
            prompt = f"""任务：从用户任务中只抽取真正想搜索的关键词，不要输出任何系统提示、执行指令、确认条件。
输入：{task}
输出：只输出1-15个中文/英文词组成的关键词，不要解释。"""
            # 这里使用简单的 generate，实际项目中可用专门的抽取工具
            # 为避免额外依赖，先用规则兜底，返回 rule_kw
            # 如需开启 LLM，可取消注释下面的代码并确保后端可用
            # resp = backend.generate(prompt, max_tokens=32)
            # return resp.strip()
        except Exception:
            pass
        # 回退到规则结果
    return rule_kw or ""


def ground_search_query(orig_keywords: str, generated_query: str) -> str:
    """如果模型生成的 query 丢失了用户原始核心实体，强制拼回。"""
    if not orig_keywords:
        return generated_query
    orig = orig_keywords.strip()
    gen = (generated_query or "").strip()
    if not gen:
        return orig

    if " " in orig or all(ord(c) < 128 for c in orig if c.strip()):
        orig_tokens = [w for w in re.split(r'[\s,，。.、:：;；]+', orig) if 2 <= len(w) <= 20]
        if orig_tokens:
            miss = [w for w in orig_tokens if w.lower() not in gen.lower()]
            if len(miss) * 10 >= len(orig_tokens) * 6:
                suffix = ""
                useful_mods = [w for w in re.findall(r'(20\d{2}|最新|进展|趋势|报告|行业|技术|产品)', gen) if orig and w not in orig][:2]
                if useful_mods:
                    suffix = " " + " ".join(useful_mods)
                return orig + suffix
        return gen

    n = len(orig)
    if n <= 1:
        return gen
    windows = []
    for w in (2, 3, 4):
        for i in range(max(0, n - w + 1)):
            windows.append(orig[i:i+w])
    windows = [w for w in windows if not re.fullmatch(r'[的了是和与或在及对于就都不也一二三四五六七八九十]+', w)]
    if not windows:
        return gen

    miss_windows = [w for w in windows if w not in gen]
    miss_ratio = len(miss_windows) / max(len(windows), 1)

    if miss_ratio >= 0.5:
        useful_mods = []
        for kw in ['最新', '进展', '趋势', '报告', '行业', '技术', '产品', '2025', '2026']:
            if kw in gen and kw not in orig:
                useful_mods.append(kw)
                if len(useful_mods) >= 2:
                    break
        joined = orig
        if useful_mods:
            joined = orig + " " + " ".join(useful_mods)
        return joined[:80]

    return gen


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------

class ResearchReportGenerator:
    """Generates a markdown research report for a given topic.

    Deep-research mode scales the search breadth (hundreds of candidate sources),
    fetches many article bodies, and synthesizes the report through iterative
    deepening + verification.
    """

    def __init__(
        self,
        backend,
        config: Optional[Any] = None,
        output_dir: Optional[Path] = None,
    ):
        self.backend = backend
        # Accept either a full AgentConfig or a ResearchConfig-like object.
        self.cfg = getattr(config, "research", config) if config else None
        if self.cfg is None:
            from .config import ResearchConfig
            self.cfg = ResearchConfig()
        self.output_dir = output_dir or (Path.home() / "OpenMythos" / "reports")
        # 对话上下文: 深度研究需结合用户之前的讨论背景, 而不是孤立搜索孤立主题。
        self._context: str = ""
        self.web_search_tool = TOOLS_REGISTRY.get("web_search")

    def run(
        self,
        task: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        context: str = "",
    ) -> Dict[str, Any]:
        self._context = (context or "").strip()
        topic = extract_research_topic(task)
        if not topic:
            return {
                "success": False,
                "final_answer": "无法从任务中识别出研究主题，请明确说明要调研的对象。",
                "metadata": {"error": "no_topic"},
            }

        self._emit(stream_callback, "status", f"researching: {topic}")
        started_at = time.time()
        all_queries: List[str] = []
        seen_queries = set()
        # 证据台账: 跨轮次累积来源(多源去重/质量分/正文), 支撑置信度评估与持续深挖
        ledger = EvidenceLedger(max_sources=self.cfg.max_total_search_results)

        # 1) Build an expanded set of queries (complementary angles)
        #    种子查询立即生成; LLM 变体在后台异步补充, 避免启动时长时间无反馈阻塞。
        self._emit(stream_callback, "status", f"researching: {topic} · 构建查询")
        queries = self._build_seed_queries(topic, original_task=task)
        for q in queries:
            seen_queries.add(self._qkey(q))
        all_queries.extend(queries)

        # 后台预热 LLM 查询变体(不阻塞首批搜索)
        llm_query_future: Optional[Any] = None
        if self.cfg.max_search_queries > len(queries):
            try:
                executor = ThreadPoolExecutor(max_workers=1)
                llm_query_future = executor.submit(
                    self._generate_query_variants, topic, extract_search_keywords(original_task or topic)
                )
            except Exception:
                llm_query_future = None

        # 2) Iterative deepening: search -> strengthen evidence -> confidence-gap analysis -> search again
        draft = ""
        for round_idx in range(1, self.cfg.iterative_rounds + 1):
            self._emit(stream_callback, "status", f"research round {round_idx}/{self.cfg.iterative_rounds} ({ledger.count()} sources)")
            if not queries:
                break
            # 首轮前合并后台 LLM 查询变体(不阻塞, 结果就绪即补充)
            if llm_query_future is not None:
                try:
                    llm_queries = llm_query_future.result(timeout=0)
                    if llm_queries:
                        merged = []
                        for q in list(queries) + list(llm_queries):
                            if self._qkey(q) not in seen_queries:
                                seen_queries.add(self._qkey(q))
                                merged.append(q)
                        if merged:
                            queries = merged[: self.cfg.max_search_queries]
                            all_queries.extend(q for q in queries if self._qkey(q) not in seen_queries)
                except Exception:
                    pass
                finally:
                    llm_query_future = None
            round_results = self._run_searches(queries, stream_callback)
            new_sources = ledger.add_sources(round_results)
            self._emit(stream_callback, "tool_result", f"round {round_idx}: +{new_sources} new sources (total {ledger.count()})")

            # 预抓本轮新增的高质量源正文, 让缺口分析看到更细的证据
            if new_sources > 0 and ledger.count() > 0:
                top_new = ledger.recent(min(new_sources, 10))
                enriched = self._enrich_sources(top_new, stream_callback)
                ledger.update_sources(enriched)
                self._emit(stream_callback, "tool_result", f"enriched {min(new_sources, 10)} new pages")

            last_round = round_idx >= self.cfg.iterative_rounds
            if last_round or not self.cfg.enable_follow_up_search:
                break

            # 生成进度草案, 让 gap 分析基于"已知什么/缺什么"来挖证据
            if ledger.count() >= 3:
                draft = self._synthesize_progress_draft(topic, ledger.get_all(), token_callback)

            followups = self._identify_gaps(topic, ledger.get_all(), draft, seen_queries, ledger)
            fresh = [q for q in followups if self._qkey(q) not in seen_queries]
            for q in fresh:
                seen_queries.add(self._qkey(q))
            if not fresh:
                self._emit(stream_callback, "status", "no new angles; stopping early")
                break
            queries = fresh[: self.cfg.max_followup_queries]
            all_queries.extend(queries)

        sources = ledger.get_all()
        if not sources:
            return {
                "success": False,
                "final_answer": f"未找到关于'{topic}'的有效搜索结果，无法生成报告。",
                "metadata": {"error": "no_search_results", "queries": all_queries},
            }

        # 3) 全量抓正文, 补足最终报告的证据细节
        self._emit(stream_callback, "status", f"enriching {min(len(sources), self.cfg.max_urls_to_fetch)} pages")
        sources = self._enrich_sources(sources, stream_callback)
        ledger.update_sources(sources)

        # 3.5) 关键论断抽取 + 证据核验 + 低支撑论断定向补搜(严谨/实在)
        claims: List[str] = []
        assessments: List[Dict[str, Any]] = []
        if self.cfg.enable_follow_up_search and ledger.count() >= 3:
            self._emit(stream_callback, "status", "extracting & verifying key claims")
            claims, assessments = self._verify_claims(
                topic, ledger, seen_queries, stream_callback, token_callback
            )
            sources = ledger.get_all()

        # 4) 证据置信度驱动的合成(带 [n] 引用、逐条论断置信度与说明)
        self._emit(stream_callback, "status", "synthesizing report")
        report_md = self._synthesize_report(topic, sources, ledger, token_callback, claims, assessments)

        # 5) 对照证据台账逐条核验
        if self.cfg.verification_rounds > 0:
            self._emit(stream_callback, "status", "verifying report against evidence")
            for v_round in range(1, self.cfg.verification_rounds + 1):
                report_md = self._verify_and_refine(
                    topic, report_md, sources, v_round, ledger, assessments, token_callback
                )

        # 6) Save to file (.md + .html)
        report_path, html_path, report_md = self._save_report(topic, report_md, deep=True)

        duration_ms = int((time.time() - started_at) * 1000)
        summary = self._generate_summary(topic, report_path, sources, all_queries, duration_ms, ledger, html_path)

        return {
            "success": True,
            "final_answer": summary,
            "report_path": str(report_path),
            "report_html_path": str(html_path) if html_path else None,
            "report_markdown": report_md,
            "metadata": {
                "topic": topic,
                "queries": all_queries,
                "sources_count": len(sources),
                "confidence": ledger.confidence_summary(),
                "fetched_sources": [s.get("url") or s.get("title") for s in sources],
                "duration_ms": duration_ms,
                "report_path": str(report_path),
                "report_html_path": str(html_path) if html_path else None,
            },
        }

    def _build_seed_queries(self, topic: str, original_task: str = "") -> List[str]:
        """仅基于用户关键词 + 固定研究角度构建种子查询(无 LLM 调用, 立即返回)."""
        orig_keywords = extract_search_keywords(original_task or topic)
        if orig_keywords:
            topic = ground_search_query(orig_keywords, topic)

        seed_queries = [topic]
        angles = ["最新", "报告", "技术", "产品", "市场", "竞品", "趋势", "案例", "数据", "风险"]
        for angle in angles:
            seed_queries.append(ground_search_query(orig_keywords or topic, f"{topic} {angle}"))

        seen = set()
        unique = []
        for q in seed_queries:
            anchored = ground_search_query(orig_keywords or topic, q)
            key = anchored.replace(" ", "").lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(anchored)
        return unique[: self.cfg.max_search_queries]

    def _build_queries(self, topic: str, original_task: str = "") -> List[str]:
        """Build a broad set of complementary search queries anchored to user keywords."""
        orig_keywords = extract_search_keywords(original_task or topic)
        if orig_keywords:
            topic = ground_search_query(orig_keywords, topic)

        # Seed queries covering the core topic + common research angles
        seed_queries = [topic]
        angles = ["最新", "报告", "技术", "产品", "市场", "竞品", "趋势", "案例", "数据", "风险"]
        for angle in angles:
            seed_queries.append(ground_search_query(orig_keywords or topic, f"{topic} {angle}"))

        # Use LLM to generate additional diverse queries when budget allows
        if self.cfg.max_search_queries > len(seed_queries):
            try:
                llm_queries = self._generate_query_variants(topic, orig_keywords)
                seed_queries.extend(llm_queries)
            except Exception:
                pass

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for q in seed_queries:
            anchored = ground_search_query(orig_keywords or topic, q)
            key = anchored.replace(" ", "").lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(anchored)
        return unique[: self.cfg.max_search_queries]

    def _generate_query_variants(self, topic: str, orig_keywords: str) -> List[str]:
        """Ask the LLM for complementary search queries."""
        today = datetime.now().strftime("%Y-%m-%d")
        _ctx_note = ""
        if self._context:
            _ctx_note = f"""
User conversation background (research must align with this context):
{self._context[:800]}
"""
        prompt = f"""You are a research assistant. Given the topic below, generate 6 to 10 diverse Chinese web-search queries that together cover definitions, market status, key products/players, technology, trends, risks, and recent news.

Today is {today}. Use the CURRENT year ({datetime.now().year}) for any time-sensitive queries (e.g. "2026 最新进展"). Do NOT use 2025/2024 unless the user explicitly asks for that year.
{_ctx_note}
Topic: {topic}
Core keywords to preserve: {orig_keywords or topic}

Rules:
- Each query on its own line.
- Preserve the core entity ({orig_keywords or topic}).
- Do NOT output numbering, bullets, or explanations.
- Output ONLY the queries.
"""
        raw = self.backend.generate(
            prompt,
            n_loops=1,
            temperature=0.5,
            max_tokens=1024,
        )
        queries = []
        for line in raw.splitlines():
            line = line.strip().strip("-•0123456789.)")
            if line:
                grounded = ground_search_query(orig_keywords or topic, line)
                if grounded:
                    queries.append(grounded)
        return queries[: self.cfg.max_search_queries]

    @staticmethod
    def _qkey(q: str) -> str:
        """查询去重键: 去除空白与小写."""
        return q.replace(" ", "").lower() if q else ""

    def _synthesize_progress_draft(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        token_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """生成紧凑进度草案: 已知要点 + 证据缺口, 供 gap 分析判断往哪挖."""
        if not sources:
            return ""
        block = []
        for i, s in enumerate(sources[: self.cfg.max_sources_for_gap_analysis], 1):
            title = (s.get("title") or "").strip()
            snippet = (s.get("snippet") or "").strip()
            score = EvidenceLedger.score_of(s)
            block.append(f"[{i}] {title} (质量{score:.2f}): {snippet[:160]}")
        _ctx_note = ""
        if self._context:
            _ctx_note = f"\n\n对话背景(缺口判断需贴合这些关注点):\n{self._context[:600]}"
        prompt = (
            f"你是研究分析师。基于以下已收集证据, 写一段 150 字以内的当前已知要点总结, "
            f"并用一句话指出最缺证据或最矛盾的方向。\n\n主题: {topic}{_ctx_note}\n\n证据:\n"
            + "\n".join(block)
            + "\n\n输出格式:\n已知要点: <两三点>\n证据缺口: <一句>\n"
        )
        try:
            raw = self.backend.generate(
                prompt, n_loops=1, temperature=0.3, max_tokens=600, token_callback=token_callback
            )
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
        except Exception:
            return ""

    def _run_searches(
        self,
        queries: List[str],
        stream_callback: Optional[Callable[[str, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        if self.web_search_tool is None:
            return []

        all_results: List[Dict[str, Any]] = []
        deep_cap = self.cfg.max_search_results_per_query

        # ---- Query-level semantic dedup: 同一核心实体 + 相似角度的查询只搜一次 ----
        # 例如 "实在智能 报告 / 实在智能 产品 / 实在智能 市场" 共享核心实体,
        # 若语义指纹高度重叠则只保留一个, 减少重复搜索与 token 消耗。
        if len(queries) > 1:
            try:
                _kept: List[str] = []
                for _q in queries:
                    _dup = False
                    for _k in _kept:
                        _a = SearchCache._semantic_ngrams(_q)
                        _b = SearchCache._semantic_ngrams(_k)
                        if SearchCache._similarity(_a, _b) >= 0.55:
                            _dup = True
                            break
                    if not _dup:
                        _kept.append(_q)
                if _kept:
                    queries = _kept
            except Exception:
                pass

        def _search_once(q: str) -> List[Dict[str, Any]]:
            """单次搜索调用(带短超时保护)."""
            try:
                result = self.web_search_tool.execute(
                    query=q,
                    max_results=5,
                    deep_max_results=deep_cap,
                )
                if result.success and result.output:
                    data = json.loads(result.output)
                    if isinstance(data, list):
                        return data
                return []
            except Exception as e:
                self._emit(stream_callback, "tool_result", f"search error for '{q}': {e}")
                return []

        def _search_one(q: str) -> List[Dict[str, Any]]:
            """单查询搜索, 带重试与查询简化兜底.

            1) 原样查询, 失败/0 结果时指数退避重试(最多 2 次);
            2) 仍无结果时用简化版查询(去掉引号/括号/长数字串)再搜一次;
            3) 彻底失败返回空。
            """
            results = _search_once(q)
            if results:
                return results
            # 网络类失败: 短退避后重试
            for attempt in range(2):
                time.sleep(0.8 * (attempt + 1))
                results = _search_once(q)
                if results:
                    return results
            # 0 结果: 简化查询(放宽)再试一次, 应对过具体/带括号的查询
            import re as _re
            relaxed = _re.sub(r"[\(\)\[\]\{\}<>\"']", " ", q)
            relaxed = _re.sub(r"\d{6,}", "", relaxed)      # 去掉长数字串(如年份统计口径)
            relaxed = _re.sub(r"\s{2,}", " ", relaxed).strip()
            if relaxed and relaxed != q:
                self._emit(stream_callback, "tool_result", f"retrying simplified: '{relaxed}'")
                results = _search_once(relaxed)
                if results:
                    return results
            return []

        # Limit concurrency to avoid overwhelming providers
        max_workers = min(len(queries), 6)
        executed_queries: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_search_one, q): q for q in queries}
            for future in as_completed(futures):
                q = futures[future]
                try:
                    results = future.result()
                    self._emit(stream_callback, "tool_result", f"search '{q}' -> {len(results)} results")
                    all_results.extend(results)
                    executed_queries.append(q)
                except Exception as e:
                    self._emit(stream_callback, "tool_result", f"search '{q}' failed: {e}")

        return self._dedupe_results(all_results)

    def _dedupe_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for r in results:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            key = (title[:60].lower(), url[:100].lower())
            if not title and not url:
                continue
            if "no results for query" in title.lower():
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
        # Prefer results with richer content
        deduped.sort(key=lambda r: len((r.get("snippet") or "").strip()), reverse=True)
        return deduped[: self.cfg.max_total_search_results]

    def _enrich_sources(
        self,
        results: List[Dict[str, Any]],
        stream_callback: Optional[Callable[[str, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch article bodies for the top URLs using the tool's internal enricher."""
        try:
            tool = self.web_search_tool
            if tool is not None and hasattr(tool, "fetcher"):
                enriched = [dict(r) for r in results]
                tool.fetcher.enrich_results(enriched, max_urls=self.cfg.max_urls_to_fetch)
                return enriched
        except Exception as e:
            self._emit(stream_callback, "tool_result", f"page fetch warning: {e}")
        return results

    def _identify_gaps(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        draft_report: str,
        seen_queries: Optional[set] = None,
        ledger: Optional[EvidenceLedger] = None,
    ) -> List[str]:
        """基于当前证据与置信度, 生成有针对性的补搜查询(排除已搜过).

        这是"持续深挖"的核心: 根据已获取信息, 有目的有条件地去找更细的证据
        (证据薄弱的方面 / 相互矛盾的结论 / 需要权威数据佐证的具体论断)。
        """
        if not sources or not self.cfg.enable_follow_up_search:
            return []

        snippet_block = []
        conf = ledger.confidence_summary() if ledger else {}
        conf_line = (
            f"整体置信度: {conf.get('confidence', 'N/A')}, "
            f"高质量源 {conf.get('high_quality', 0)}/{conf.get('total', 0)}, "
            f"来源分布 {conf.get('providers', {})}"
        )
        for s in sources[: self.cfg.max_sources_for_gap_analysis]:
            title = (s.get("title") or "").strip()
            snippet = (s.get("snippet") or "").strip()
            score = EvidenceLedger.score_of(s)
            if title or snippet:
                snippet_block.append(f"- {title} (质量{score:.2f}): {snippet[:160]}")
        source_summary = "\n".join(snippet_block)

        seen = seen_queries or set()
        already = "\n".join(f"- {q}" for q in list(seen)[-20:]) if seen else "(无)"

        prompt = f"""你是研究策略师。基于主题、当前证据与置信度, 找出 2-4 个知识缺口, 输出针对性的中文搜索查询去补强证据。

主题: {topic}

{conf_line}

当前证据:
{source_summary}

已有进度草案:
{draft_report or "(尚未生成)"}

已搜索过的查询(不要重复):
{already}

要求:
- 优先针对: 证据薄弱的方面、相互矛盾的结论、需要权威/数据来源佐证的具体论断。
- 每条查询要具体(实体+维度, 如 "XXX 2024 融资规模 亿元")。
- 每行一条, 只输出查询, 不输出解释/编号。
"""
        try:
            raw = self.backend.generate(
                prompt,
                n_loops=1,
                temperature=0.5,
                max_tokens=1024,
            )
            queries = []
            for line in raw.splitlines():
                line = line.strip().strip("-•0123456789.)")
                if line:
                    grounded = ground_search_query(topic, line)
                    if grounded and self._qkey(grounded) not in seen:
                        queries.append(grounded)
            return queries[: self.cfg.max_followup_queries]
        except Exception:
            return []

    # ===================== 关键论断抽取与置信度核验 =====================

    def _verify_claims(
        self,
        topic: str,
        ledger: EvidenceLedger,
        seen_queries: set,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        token_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """抽取关键论断 -> 逐条证据核验 -> 对低支撑论断做定向补搜(严谨)."""
        claims = self._extract_claims(topic, ledger.get_all(), token_callback)
        if not claims:
            return [], []

        assessments = [self._assess_claim_confidence(c, ledger) for c in claims]
        low = [a for a in assessments if a["support_count"] < 2]

        # 有目的有条件的深挖: 为低支撑/无支撑论断定向找更细的证据
        if low and self.cfg.enable_follow_up_search:
            self._emit(stream_callback, "status", f"corroborating {len(low)} low-evidence claims")
            extra = []
            for a in low[: self.cfg.max_claims_to_verify]:
                extra.extend(self._claim_search_queries(topic, a["claim"]))
            fresh = [q for q in extra if self._qkey(q) not in seen_queries]
            for q in fresh:
                seen_queries.add(self._qkey(q))
            if fresh:
                results = self._run_searches(fresh[: self.cfg.max_followup_queries * 2], stream_callback)
                added = ledger.add_sources(results)
                if added:
                    new_top = ledger.recent(min(added, 8))
                    enriched = self._enrich_sources(new_top, stream_callback)
                    ledger.update_sources(enriched)
                    self._emit(stream_callback, "tool_result", f"corroboration round: +{added} sources")
            # 补搜后重新评估所有论断
            assessments = [self._assess_claim_confidence(c, ledger) for c in claims]

        return claims, assessments

    def _extract_claims(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        token_callback: Optional[Callable[[int], None]] = None,
    ) -> List[str]:
        """从已抓取证据中提炼关键的事实性论断(数字/规模/时间/事件/关系)."""
        if not sources:
            return []
        block = []
        for s in sources[: self.cfg.max_sources_for_gap_analysis]:
            snippet = (s.get("snippet") or "").strip()[:200]
            body = (s.get("_body") or "").strip()[:200]
            content = body if len(body) > len(snippet) else snippet
            if content:
                block.append(f"- {s.get('title') or '':}: {content}")
        prompt = (
            f"从以下关于'{topic}'的证据中, 提炼 {self.cfg.max_claims_to_verify} 条以内最关键、"
            f"可被证实或证伪的事实性论断(包含具体数字/规模/时间/事件/关系)。\n"
            f"每条一行, 只输出论断, 不要编号、不要解释、不要推测性表述。\n\n证据:\n"
            + "\n".join(block)
        )
        try:
            raw = self.backend.generate(
                prompt, n_loops=1, temperature=0.3, max_tokens=1024, token_callback=token_callback
            )
            claims = []
            for line in raw.splitlines():
                line = line.strip().strip("-•0123456789.)")
                if len(line) >= 8:
                    claims.append(line)
            return claims[: self.cfg.max_claims_to_verify]
        except Exception:
            return []

    def _assess_claim_confidence(
        self,
        claim: str,
        ledger: EvidenceLedger,
    ) -> Dict[str, Any]:
        """逐条论断的置信度研判: 支撑来源数/质量分/来源多样性/矛盾检测/多源数字求证."""
        supporting = ledger.find_supporting(claim, min_overlap=2)
        contradictions = ledger.find_contradicting(claim, min_overlap=2)
        count = len(supporting)
        avg_score = sum(ledger.score_of(s) for s, _ in supporting) / count if count else 0.0
        providers = {ledger._provider_of(s) for s, _ in supporting}

        # 多源数字求证: 数据要绝对准确
        numeric = ledger.numeric_evidence(claim)
        num_min_agree = numeric["min_agree"] if numeric else None
        num_conflict = numeric["has_conflict"] if numeric else False
        num_conflict_detail = numeric["conflict_per_number"] if numeric else {}

        level = ledger.claim_confidence(
            support_count=count,
            avg_score=avg_score,
            provider_count=len(providers),
            contradiction_count=len(contradictions),
            min_support_high=self.cfg.min_support_for_high_confidence,
            num_min_agree=num_min_agree,
            num_conflict=num_conflict,
        )
        return {
            "claim": claim,
            "confidence": level,
            "support_count": count,
            "avg_score": round(avg_score, 2),
            "providers": sorted(providers),
            "contradictions": len(contradictions),
            "has_numbers": numeric is not None,
            "num_agree": num_min_agree,
            "num_conflict": num_conflict,
            "num_conflict_detail": num_conflict_detail,
        }

    def _claim_search_queries(self, topic: str, claim_text: str) -> List[str]:
        """为单条论断生成 1-2 条定向补搜查询; 含数字的论断附带数字+单位查询."""
        q = ground_search_query(topic, claim_text.strip()[:40])
        queries = [q] if q else []
        toks = [t for t in claim_tokens(claim_text) if len(t) >= 3]
        if toks:
            queries.append(ground_search_query(topic, f"{topic} {toks[0]}"))
        # 数字求证: 补一条带数据型数字+单位的查询, 便于多源交叉印证
        nums = _data_numbers(claim_text)
        if nums:
            v, unit, _ = nums[0]
            queries.append(ground_search_query(topic, f"{topic} {v:g}{unit}"))
        return queries

    def _synthesize_report(
        self,
        topic: str,
        sources: List[Dict[str, Any]],
        ledger: Optional[EvidenceLedger] = None,
        token_callback: Optional[Callable[[int], None]] = None,
        claims: Optional[List[str]] = None,
        assessments: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Use LLM to synthesize a markdown report with explicit <think> reasoning,
        per-claim citation markers and a confidence section."""
        source_texts = []
        for i, s in enumerate(sources[: self.cfg.max_sources_for_report], 1):
            title = (s.get("title") or "").strip()
            url = (s.get("url") or "").strip()
            snippet = (s.get("snippet") or "").strip()
            body = (s.get("_body") or "").strip()
            content = body if len(body) > len(snippet) else snippet
            score = EvidenceLedger.score_of(s)
            source_texts.append(
                f"[Source {i}] (quality {score:.2f})\nTitle: {title}\nURL: {url}\nContent:\n{content[:1200]}\n"
            )

        source_block = "\n".join(source_texts)
        today = datetime.now().strftime("%Y-%m-%d")

        cite_rule = ""
        if self.cfg.require_citations:
            cite_rule = (
                "- 关键事实/数字末尾标注引用源编号, 如 [1][2]; 编号必须对应 ## 参考来源 的条目。\n"
            )

        _ctx_note = ""
        if self._context:
            _ctx_note = f"""
## 对话背景 (报告需呼应这些上下文, 聚焦用户真正关心的点)
{self._context[:1200]}
"""
        prompt = f"""你是资深研究分析师。请基于以下证据撰写一份完整、专业、信息丰富的中文 Markdown 深度调研报告。

主题: {topic}
研究日期: {today}
{_ctx_note}

先在 <think>...</think> 内思考: 各来源可信度、观点冲突、证据缺口、报告结构。然后输出报告。

## 证据材料
{source_block}

## 写作要求
- 标题: # {topic} 深度调研报告
- 结构必须包含:
  - ## 摘要
  - ## 背景与定义
  - ## 现状分析
  - ## 主要参与者或关键产品
  - ## 发展趋势
  - ## 风险与挑战
  - ## 结论与建议
  - ## 参考来源 (numbered list, 每行: [n] URL - 标题)
- 报告要完整、专业、信息量大: 每个维度都应有实质内容(数据、事实、分析), 不要空洞或敷衍。
- 用表格/要点/层级标题组织, 让报告美观易读; 关键数据加粗或用表格呈现。
- 数据严谨: 引用事实尽量给出数字/时间/主体; 无法多源证实的数字标注"据公开信息"。
- 参考来源编号 [n] 与 ## 参考来源 一一对应。
- 不要输出"置信度/证据评估/数据分歧"等内部推理过程章节——直接呈现干净完整的分析结论。
- 只输出 Markdown 内容。
"""
        report = self.backend.generate(
            prompt,
            n_loops=1,
            temperature=0.4,
            max_tokens=self.cfg.report_max_tokens,
            token_callback=token_callback,
        ).strip()

        # Strip the thinking block from final report (kept in stream as REASONING)
        report = re.sub(r"<think>.*?</think>", "", report, flags=re.DOTALL | re.IGNORECASE).strip()
        # 清理可能的"证据与置信度/关键论断"残留章节(若模型仍输出则剥离)
        report = re.sub(r"##+\s*(?:证据与置信度|关键论断与置信度|数据分歧|逐条置信度).*?(?=\n##|\Z)", "", report, flags=re.DOTALL)

        # Ensure markdown title
        if not report.startswith("# "):
            report = f"# {topic} 深度调研报告\n\n" + report

        # Clean up accidental tool syntax
        report = re.sub(r"\[TOOL:\w+\].*?\[/TOOL\]", "", report, flags=re.DOTALL)
        report = re.sub(r"\[/?TOOL:?\w*\]", "", report)
        return report.strip()

    def _verify_and_refine(
        self,
        topic: str,
        report_md: str,
        sources: List[Dict[str, Any]],
        round_idx: int,
        ledger: Optional[EvidenceLedger] = None,
        assessments: Optional[List[Dict[str, Any]]] = None,
        token_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """对照证据清单与逐条置信度核验报告: 修正引用、剔除无来源论断、标注低置信."""
        numbered = []
        for i, s in enumerate(sources[: self.cfg.max_sources_for_report], 1):
            url = (s.get("url") or "")
            title = (s.get("title") or "").strip()
            numbered.append(f"[{i}] {url} - {title}")
        source_list = "\n".join(numbered)

        conf = ledger.confidence_summary() if ledger else {}
        conf_line = f"整体证据置信度: {conf.get('confidence', 'N/A')}, 高质量源 {conf.get('high_quality', 0)}/{conf.get('total', 0)}"

        assess_block = ""
        if assessments:
            rows = []
            for a in assessments:
                agree = a.get("num_agree")
                agree_txt = f"{agree}" if a.get("has_numbers") else "-"
                conflict_txt = "有" if a.get("num_conflict") else "无"
                rows.append(
                    f"- {a['confidence']} | {a['support_count']}源 | 数字印证{agree_txt} | 分歧{conflict_txt} | {a['claim'][:70]}"
                )
            assess_block = "逐条置信度研判(核验时不得放宽):\n" + "\n".join(rows) + "\n\n"

        prompt = f"""你是批判性审稿人。请对照证据清单核验这份调研报告的正确性、引用与数据严谨性。

主题: {topic}
{conf_line}

{assess_block}证据清单(编号与参考来源对应):
{source_list}

报告:
{report_md}

要求:
- 逐条核验关键论断是否有对应编号来源支持; 无来源支持的论断要么删除, 要么标注"据公开信息"。
- 修正引用编号, 确保与 ## 参考来源 一致。
- 数据严谨: 单一来源/未经多源证实的数字不得表述为确定事实; 存在多源分歧的数据尽量并列各来源数值。
- 检查事实一致性、逻辑与平衡。
- 保持报告的干净完整, 不要加入"置信度/证据评估"等内部过程章节。
- 输出完整 Markdown 报告; 若已优秀则原样返回。
"""
        refined = self.backend.generate(
            prompt,
            n_loops=1,
            temperature=0.35,
            max_tokens=self.cfg.report_max_tokens,
            token_callback=token_callback,
        ).strip()

        refined = re.sub(r"<think>.*?</think>", "", refined, flags=re.DOTALL | re.IGNORECASE).strip()
        # 清理核验可能引入的置信度/证据章节
        refined = re.sub(r"##+\s*(?:证据与置信度|关键论断与置信度|数据分歧|逐条置信度).*?(?=\n##|\Z)", "", refined, flags=re.DOTALL)
        if not refined.startswith("# "):
            refined = f"# {topic} 深度调研报告\n\n" + refined
        refined = re.sub(r"\[TOOL:\w+\].*?\[/TOOL\]", "", refined, flags=re.DOTALL)
        refined = re.sub(r"\[/?TOOL:?\w*\]", "", refined)
        return refined.strip()

    @staticmethod
    def _dedupe_tables(report_md: str) -> str:
        """去重报告中完全重复的 Markdown 表格块.

        LLM 在合成/核验时可能把"关键论断与置信度"表输出两遍, 这里按表格
        的规范化内容(去掉表头分隔行)做块级去重, 保留第一次出现的位置。
        """
        if not report_md or "|" not in report_md:
            return report_md

        def norm_table(block: List[str]) -> str:
            # 去掉 `|---|---|` 分隔行, 仅比对数据行
            rows = [ln for ln in block if not re.fullmatch(r"\s*\|[\s:\-|]+\|\s*", ln)]
            return "\n".join(r.strip() for r in rows)

        lines = report_md.splitlines()
        out: List[str] = []
        seen: set = set()
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                # 向前收集紧邻的标题(如 `## 关键论断与置信度`), 连同表格一起去重
                heading_idx = i - 1
                heading = ""
                if heading_idx >= 0 and lines[heading_idx].strip().startswith("#"):
                    heading = lines[heading_idx].strip()
                block: List[str] = []
                j = i
                while j < n:
                    s = lines[j].strip()
                    if s.startswith("|") and s.endswith("|"):
                        block.append(lines[j])
                        j += 1
                    else:
                        break
                key = norm_table(block)
                if key and key in seen:
                    # 跳过重复表格及其标题
                    if heading and out and out[-1].strip() == heading:
                        out.pop()
                    if out and out[-1].strip():
                        out.append("")
                    i = j
                    continue
                if key:
                    seen.add(key)
                out.extend(block)
                i = j
                continue
            out.append(line)
            i += 1
        return "\n".join(out)

    def _save_report(self, topic: str, report_md: str, deep: bool = False):
        """保存报告为 .md, 并按配置同时输出 .html. 返回 (md_path, html_path)."""
        report_md = self._dedupe_tables(report_md)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", topic).strip("_").lower()[:40] or "report"
        date_str = datetime.now().strftime("%Y-%m-%d")
        suffix = "_deep" if deep else ""
        base_name = f"{date_str}_{slug}{suffix}"
        report_path = self.output_dir / f"{base_name}.md"

        counter = 1
        while report_path.exists():
            report_path = self.output_dir / f"{base_name}_{counter}.md"
            counter += 1

        report_path.write_text(report_md, encoding="utf-8")

        html_path = None
        formats = getattr(self.cfg, "report_formats", None) or ["md"]
        if "html" in formats:
            html = self._render_html(topic, report_md)
            html_path = report_path.with_suffix(".html")
            html_path.write_text(html, encoding="utf-8")

        return report_path, html_path, report_md

    @staticmethod
    def _md_to_html(md: str) -> str:
        """轻量 Markdown -> HTML(面向调研报告: 标题/表格/列表/代码块/加粗/行内代码/链接)."""
        import html as _html
        lines = md.split("\n")
        out: List[str] = []
        i, n = 0, len(lines)

        def inline(s: str) -> str:
            s = _html.escape(s)
            s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
            s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
            s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', s)
            return s

        while i < n:
            stripped = lines[i].strip()
            if stripped.startswith("```"):
                lang = stripped[3:].strip()
                code = []
                i += 1
                while i < n and not lines[i].strip().startswith("```"):
                    code.append(lines[i])
                    i += 1
                i += 1
                out.append(
                    f'<pre><code class="language-{_html.escape(lang or "text")}">'
                    f"{_html.escape(chr(10).join(code))}</code></pre>"
                )
                continue
            if stripped.startswith("|"):
                rows = []
                while i < n and lines[i].strip().startswith("|"):
                    rows.append(lines[i])
                    i += 1
                if rows:
                    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
                    body_rows = []
                    for r in rows[1:]:
                        cells = [c.strip() for c in r.strip().strip("|").split("|")]
                        if set(cells) <= {"-", ":", "---", ":--", "--:", ":---", "---:"}:
                            continue
                        body_rows.append(cells)
                    t = ["<table>", "<thead><tr>"]
                    for c in header:
                        t.append(f"<th>{inline(c)}</th>")
                    t.append("</tr></thead><tbody>")
                    for r in body_rows:
                        t.append("<tr>")
                        for c in r:
                            t.append(f"<td>{inline(c)}</td>")
                        t.append("</tr>")
                    t.append("</tbody></table>")
                    out.append("".join(t))
                continue
            if stripped.startswith("#"):
                level = min(len(stripped) - len(stripped.lstrip("#")), 6)
                text = stripped.lstrip("#").strip()
                out.append(f"<h{level}>{inline(text)}</h{level}>")
                i += 1
                continue
            is_ul = re.match(r"^\s*(?:[-*+])\s+", lines[i])
            is_ol = re.match(r"^\s*\d+[.、]\s+", lines[i])
            if is_ul or is_ol:
                items = []
                ordered = bool(is_ol)
                while i < n:
                    s2 = lines[i].strip()
                    m = re.match(r"^\s*(?:[-*+])\s+", lines[i]) or re.match(r"^\s*\d+[.、]\s+", lines[i])
                    if m:
                        items.append(re.sub(r"^\s*(?:[-*+]|\d+[.、])\s+", "", s2))
                        i += 1
                    elif not s2:
                        i += 1
                        break
                    else:
                        break
                tag = "ol" if ordered else "ul"
                out.append(f"<{tag}>")
                for it in items:
                    out.append(f"<li>{inline(it)}</li>")
                out.append(f"</{tag}>")
                continue
            if stripped.startswith(">"):
                quote = []
                while i < n and lines[i].strip().startswith(">"):
                    quote.append(lines[i].strip().lstrip(">").strip())
                    i += 1
                out.append(f"<blockquote>{inline(' '.join(quote))}</blockquote>")
                continue
            if not stripped:
                i += 1
                continue
            para = [stripped]
            i += 1
            while (
                i < n
                and lines[i].strip()
                and not lines[i].strip().startswith(("#", "|", "```", ">"))
                and not re.match(r"^\s*(?:[-*+]|\d+[.、])\s+", lines[i])
            ):
                para.append(lines[i].strip())
                i += 1
            out.append(f"<p>{inline(' '.join(para))}</p>")
        return "\n".join(out)

    def _render_html(self, topic: str, report_md: str) -> str:
        """把 Markdown 报告包进带样式的 HTML 页面."""
        import html as _html
        body = self._md_to_html(report_md)
        today = datetime.now().strftime("%Y-%m-%d")
        css = """
body{font-family:-apple-system,"PingFang SC","Microsoft YaHei",Helvetica,Arial,sans-serif;max-width:880px;margin:0 auto;padding:48px 28px;line-height:1.8;color:#1f2430;background:#fff}
h1{font-size:1.8em;border-bottom:2px solid #e5e7eb;padding-bottom:10px;margin-top:0}
h2{color:#1a56db;margin-top:2em;border-bottom:1px solid #eee;padding-bottom:6px}
h3{color:#374151;margin-top:1.5em}
table{border-collapse:collapse;width:100%;margin:1em 0;font-size:0.95em}
th,td{border:1px solid #e2e8f0;padding:9px 12px;text-align:left}
th{background:#f5f7fa;font-weight:600}
tr:nth-child(even){background:#fafbfc}
pre{background:#f6f8fa;padding:14px 18px;border-radius:8px;overflow-x:auto;font-size:0.9em}
code{background:#f0f2f5;padding:2px 6px;border-radius:4px;font-size:0.92em}
pre code{background:none;padding:0}
blockquote{border-left:4px solid #d1d5db;margin:1em 0;padding:6px 18px;color:#6b7280}
a{color:#1a56db;text-decoration:none}a:hover{text-decoration:underline}
li{margin:5px 0}
.footer{margin-top:3.5em;color:#9ca3af;font-size:12px;border-top:1px solid #eee;padding-top:14px}
"""
        return (
            "<!DOCTYPE html>\n<html lang=\"zh\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{_html.escape(topic)} 深度调研报告</title>\n"
            f"<style>{css}</style>\n</head>\n<body>\n{body}\n"
            f'<div class="footer">由 Lv Super Agent (OpenMythos) 生成 · {today} · 数据经多源头综合调研</div>\n</body>\n</html>\n'
        )

    def _generate_summary(
        self,
        topic: str,
        report_path: Path,
        sources: List[Dict[str, Any]],
        queries: List[str],
        duration_ms: int,
        ledger: Optional[EvidenceLedger] = None,
        html_path: Optional[Path] = None,
    ) -> str:
        """Generate a short Chinese summary pointing the user to the saved file."""
        conf = ledger.confidence_summary() if ledger else {}
        conf_label = ledger.confidence_label() if ledger else "N/A"
        urls = [s.get("url") or "" for s in sources[:8] if s.get("url")]
        source_list = "\n".join(f"- {u}" for u in urls if u)
        duration_s = duration_ms / 1000
        files = f"**Markdown**: {report_path}"
        if html_path:
            files += f"\n**HTML**: {html_path}"
        return (
            f"已完成《{topic}》深度调研报告（{len(sources)} 个来源，{len(queries)} 次查询，"
            f"证据置信度 {conf_label}（{conf.get('confidence', 'N/A')}），耗时 {duration_s:.1f}s），"
            f"并保存为：\n\n{files}\n\n"
            f"参考来源：\n{source_list}"
        )

    @staticmethod
    def _emit(
        stream_callback: Optional[Callable[[str, str], None]],
        kind: str,
        text: str,
    ) -> None:
        if stream_callback:
            stream_callback(kind, text)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def generate_research_report(
    task: str,
    backend,
    config: Optional[Any] = None,
    stream_callback: Optional[Callable[[str, str], None]] = None,
    token_callback: Optional[Callable[[int], None]] = None,
    output_dir: Optional[Path] = None,
    context: str = "",
) -> Dict[str, Any]:
    """Generate a research report for the given task and return result dict.

    context: 可选对话背景, 让研究结合用户之前的讨论, 而非孤立搜索孤主题.
    """
    generator = ResearchReportGenerator(backend, config=config, output_dir=output_dir)
    return generator.run(task, stream_callback=stream_callback, token_callback=token_callback, context=context)
