"""
EvidenceLedger - 证据台账与置信度引擎.

持续深挖研究的证据基础设施:
- 跨轮次累积研究来源, 按规范 URL 去重(同一 URL 只保留摘要最丰富的一份)
- 保留来源溯源(provider / score / score_factors / 抓取正文), 供证据引用
- 提供来源质量分聚合与全局置信度评估(平均质量 / 高质量源占比 / 来源多样性)
- 供深挖循环判断"还缺多少证据"与"提前停止"
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional


def canonical_url(url: str) -> str:
    """规范化 URL 用于去重: 去协议/www/锚点/常见追踪参数/尾部斜杠."""
    if not url:
        return ""
    u = url.strip().split("#", 1)[0]
    u = re.sub(r"^https?://", "", u, flags=re.IGNORECASE)
    u = re.sub(r"^www\.", "", u, flags=re.IGNORECASE)
    u = re.sub(
        r"[?&](?:utm_[a-z]+|fbclid|gclid|from|spm|ref|source|share_token)=[^&]*",
        "",
        u,
        flags=re.IGNORECASE,
    )
    u = u.rstrip("?&/")
    return u.lower()


def claim_tokens(text: str) -> set:
    """从论断中提取可匹配的检索特征: 数字/英文词 + 中文短语(2-6字滑窗)."""
    if not text:
        return set()
    tokens = set()
    for m in re.finditer(r"[A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*", text):
        tokens.add(m.group(0).lower())
    for w in re.findall(r"[\u4e00-\u9fff]{2,6}", text):
        if len(w) >= 2:
            tokens.add(w)
    return tokens


_NEGATION_WORDS = (
    "不是", "没有", "并未", "并不", "相反", "否认", "错误", "并非",
    "并非如此", "虚假", "夸大", "辟谣", "回应称", "未", "无证据",
)

# 数字+单位: 用于多源数字求证(数据绝对准确的关键)
_NUM_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(亿元|万元|万亿|亿美元|欧元|日元|元|％|%|倍|个百分点|万亿|万|亿|"
    r"个|家|人|次|年|天|月|美元|MB|GB|TB|KB|万人|万辆|万台|度|项|笔)?"
)


def _claim_numbers(text: str) -> List[Tuple[float, str, str]]:
    """从文本中提取 (数值, 单位, 原文) 列表."""
    if not text:
        return []
    out = []
    for m in _NUM_UNIT_RE.finditer(text):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        unit = (m.group(2) or "").strip()
        out.append((val, unit, m.group(0)))
    return out


_TIME_UNITS = {"年", "月", "日"}


def _norm_unit(unit: str) -> str:
    """单位归一化, 保证 '1000亿' 与 '1000亿元' 视为同一量纲."""
    if unit in ("亿", "亿元"):
        return "亿元"
    if unit in ("万", "万元"):
        return "万元"
    if unit in ("%", "％", "个百分点"):
        return "%"
    return unit


def _is_context_number(val: float, unit: str) -> bool:
    """把年份/日期这类"上下文数字"排除出数据点(它们不是待求证指标)."""
    if unit in _TIME_UNITS:
        return True
    if not unit and 1900 <= val <= 2100:
        return True
    return False


def _data_numbers(text: str) -> List[Tuple[float, str, str]]:
    """提取"数据型"数字: 归一化单位、归一化万亿、剔除年份/日期等上下文数字."""
    out = []
    for val, unit, raw in _claim_numbers(text):
        if _is_context_number(val, unit):
            continue
        if unit == "万亿":
            val = val * 10000.0
            unit = "亿元"
        out.append((val, _norm_unit(unit), raw))
    return out


def _source_text(s: Dict[str, Any]) -> str:
    return " ".join([
        s.get("title") or "",
        s.get("snippet") or "",
        s.get("_body") or "",
    ]).lower()


class EvidenceLedger:
    """按规范 URL 去重保存来源, 保留溯源字段, 支持置信度聚合."""

    def __init__(self, max_sources: int = 200):
        self.max_sources = max_sources
        self._by_url: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []

    def _qkey(self, url: str) -> str:
        return canonical_url(url)

    def add_sources(self, sources: List[Dict[str, Any]]) -> int:
        """把一批搜索结果并入台账, 返回本轮新增数量."""
        added = 0
        for s in sources or []:
            url = s.get("url") or ""
            key = self._qkey(url)
            if not key:
                continue
            existing = self._by_url.get(key)
            if existing is not None:
                # 摘要更丰富则升级, 保留原来源分
                if len((s.get("snippet") or "")) > len((existing.get("snippet") or "")):
                    merged = dict(existing)
                    merged["snippet"] = s.get("snippet")
                    merged["_merged_at"] = datetime.now().isoformat()
                    self._by_url[key] = merged
                continue
            entry = dict(s)
            entry["_canonical"] = key
            entry["_discovered_at"] = datetime.now().isoformat()
            self._by_url[key] = entry
            self._order.append(key)
            added += 1
        while len(self._order) > self.max_sources:
            old = self._order.pop(0)
            self._by_url.pop(old, None)
        return added

    def count(self) -> int:
        return len(self._order)

    def get_all(self) -> List[Dict[str, Any]]:
        return [self._by_url[k] for k in self._order]

    def recent(self, n: int) -> List[Dict[str, Any]]:
        return [self._by_url[k] for k in self._order[-n:]]

    def update_sources(self, enriched: List[Dict[str, Any]]) -> None:
        """把抓取正文/更全摘要写回台账(按规范 URL 匹配)."""
        for s in enriched or []:
            key = s.get("_canonical") or self._qkey(s.get("url") or "")
            if key and key in self._by_url:
                if (s.get("snippet") or ""):
                    self._by_url[key]["snippet"] = s["snippet"]
                if (s.get("_body") or ""):
                    self._by_url[key]["_body"] = s["_body"]
                self._by_url[key]["_enriched_from"] = True

    @staticmethod
    def score_of(s: Dict[str, Any]) -> float:
        """来源质量分(0-1): 优先 SearchScorer 的 score, 缺失时用因子估计."""
        score = s.get("score")
        if isinstance(score, (int, float)):
            return max(0.0, min(1.0, float(score)))
        factors = s.get("score_factors")
        if isinstance(factors, dict):
            vals = [v for v in factors.values() if isinstance(v, (int, float))]
            if vals:
                return max(0.0, min(1.0, sum(vals) / len(vals)))
        return 0.5

    def provider_stats(self) -> Dict[str, int]:
        """按来源统计 provider 覆盖度."""
        stats: Dict[str, int] = {}
        for k in self._order:
            prov = self._by_url[k].get("_provider") or self._by_url[k].get("provider") or "unknown"
            stats[prov] = stats.get(prov, 0) + 1
        return stats

    def confidence_summary(self) -> Dict[str, Any]:
        """全局证据置信度聚合: 平均质量 / 高质量源占比 / 来源多样性."""
        sources = self.get_all()
        if not sources:
            return {"confidence": 0.0, "high_quality": 0, "total": 0, "providers": {}}
        scores = [self.score_of(s) for s in sources]
        high = sum(1 for s in scores if s >= 0.7)
        avg = sum(scores) / len(scores)
        diversity = min(1.0, len(self.provider_stats()) / 4.0)
        confidence = 0.6 * avg + 0.25 * (high / len(scores)) + 0.15 * diversity
        return {
            "confidence": round(min(1.0, confidence), 2),
            "high_quality": high,
            "total": len(sources),
            "avg_score": round(avg, 2),
            "providers": self.provider_stats(),
        }

    def confidence_label(self) -> str:
        c = self.confidence_summary()["confidence"]
        if c >= 0.75:
            return "高"
        if c >= 0.5:
            return "中"
        return "低"

    # ===================== 论断级证据核验 =====================

    def find_supporting(self, claim_text: str, min_overlap: int = 2) -> List[Dict[str, Any]]:
        """返回支持某论断的来源(按特征重叠度降序)."""
        tokens = claim_tokens(claim_text)
        if not tokens:
            return []
        scored = []
        for s in self.get_all():
            hay = _source_text(s)
            overlap = sum(1 for t in tokens if t in hay)
            if overlap >= min_overlap:
                scored.append((s, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def find_contradicting(self, claim_text: str, min_overlap: int = 2) -> List[Dict[str, Any]]:
        """检测与论断冲突的来源(包含论断特征词 + 否定/反驳语义)."""
        tokens = claim_tokens(claim_text)
        if not tokens:
            return []
        out = []
        for s in self.get_all():
            hay = _source_text(s)
            overlap = sum(1 for t in tokens if t in hay)
            if overlap >= min_overlap and any(n in hay for n in _NEGATION_WORDS):
                out.append(s)
        return out

    @staticmethod
    def _provider_of(s: Dict[str, Any]) -> str:
        return s.get("_provider") or s.get("provider") or "unknown"

    def numeric_evidence(
        self,
        claim_text: str,
        min_overlap: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """多源数字求证: 论断中的"数据型数字"(排除年份/日期)被多少来源一致印证.

        数学严谨性:
        - 只对待求证的指标数字做印证(年份/日期不参与);
        - 单位归一化(1000亿 == 1000亿元), 万亿换算为亿元;
        - 同一指标出现不同数值即视为数据分歧, 必须显式暴露, 不得任取其一。
        """
        nums = _data_numbers(claim_text)
        if not nums:
            return None
        tokens = claim_tokens(claim_text)
        agree: Dict[str, set] = {}
        conflict: Dict[str, set] = {}
        checked = 0
        for s in self.get_all():
            hay = _source_text(s)
            overlap = sum(1 for t in tokens if t in hay)
            if overlap < min_overlap:
                continue
            checked += 1
            hay_nums = _data_numbers(hay)
            for val, unit, _raw in nums:
                same = any(
                    abs(val - hv) < max(1.0, val * 0.01) and hu == unit
                    for hv, hu, _ in hay_nums
                )
                if same:
                    agree.setdefault(unit, set()).add(s.get("url") or "")
                diffs = [
                    (hv, hu) for hv, hu, _ in hay_nums
                    if hu == unit and abs(val - hv) >= max(1.0, val * 0.01)
                ]
                if diffs:
                    for dv, du in diffs:
                        conflict.setdefault(unit, set()).add((s.get("url") or "", dv, du))
        return {
            "numbers": [f"{v:g}{u}" for v, u, _ in nums],
            "agreeing_per_number": {
                f"{v:g}{u}": len(agree.get(u, set())) for v, u, _ in nums
            },
            "min_agree": min((len(v) for v in agree.values()), default=0),
            "conflict_per_number": {
                f"{v:g}{u}": sorted({(u2, f"{dv:g}{du}") for u2, dv, du in conflict.get(u, set())})
                for v, u, _ in nums
            },
            "has_conflict": bool(conflict),
            "checked_sources": checked,
        }

    @classmethod
    def claim_confidence(
        cls,
        support_count: int,
        avg_score: float,
        provider_count: int,
        contradiction_count: int,
        min_support_high: int = 3,
        num_min_agree: Optional[int] = None,
        num_conflict: bool = False,
    ) -> str:
        """严谨的论断置信度研判(含多源数字求证).

        - 高: >=3 个独立来源交叉印证, 平均质量分高, >=2 个来源, 且无矛盾
        - 中: 2 个来源印证或单个高质量来源, 无矛盾
        - 低: 单一来源支撑(证据薄弱)
        - 极低: 无直接来源支撑 或 存在矛盾来源 或 数字存在多源分歧
        含数字的论断, 若数字未获 min_support_high 个来源印证, 最高只能判"中"。
        """
        if contradiction_count > 0 or num_conflict:
            return "极低"
        if support_count >= min_support_high and avg_score >= 0.6 and provider_count >= 2:
            base = "高"
        elif support_count >= 2 and avg_score >= 0.5:
            base = "中"
        elif support_count == 1 and avg_score >= 0.7:
            base = "中"
        elif support_count >= 1:
            base = "低"
        else:
            return "极低"
        # 数字多源求证: 数字未获充分印证则不可判"高"
        if num_min_agree is not None and num_min_agree < min_support_high and base == "高":
            return "中"
        return base
