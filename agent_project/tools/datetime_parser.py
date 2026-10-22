"""
Smart Date/Time Parsing Utility - Natural language date/time parsing for all personal assistant tools.
Supports Chinese and English relative dates, recurring patterns, and fuzzy matching.
"""

import re
from datetime import datetime, date, time, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class TimeGranularity(Enum):
    EXACT = "exact"           # Specific date/time: "2026-10-15 14:30"
    DAY = "day"               # Day only: "tomorrow", "next Friday"
    WEEK = "week"             # Week range: "this week", "next week"
    MONTH = "month"           # Month range: "this month", "October"
    RANGE = "range"           # Date range: "Oct 15-20", "this weekend"


@dataclass
class ParsedDateTime:
    """Result of parsing a natural language date/time expression."""
    start: datetime
    end: Optional[datetime] = None
    granularity: TimeGranularity = TimeGranularity.EXACT
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None  # daily, weekly, monthly, yearly
    recurrence_end: Optional[datetime] = None
    confidence: float = 1.0
    original_text: str = ""
    
    def to_iso_range(self) -> Tuple[str, str]:
        """Return (start_iso, end_iso) for database queries."""
        start_iso = self.start.isoformat()
        end_iso = (self.end or self.start).isoformat()
        return start_iso, end_iso


class DateTimeParser:
    """Parse natural language date/time expressions in Chinese and English."""
    
    # Chinese patterns
    CN_RELATIVE_DAYS = {
        "今天": 0, "今日": 0,
        "明天": 1, "明日": 1,
        "后天": 2,
        "大后天": 3,
        "昨天": -1, "昨日": -1,
        "前天": -2,
        "今晚": 0, "今夜": 0,
        "明晚": 1, "明日晚上": 1,
        "昨晚": -1,
        "早上": 0, "上午": 0, "中午": 0,
    }
    
    CN_WEEKDAYS = {
        "周一": 0, "星期一": 0, "礼拜一": 0,
        "周二": 1, "星期二": 1, "礼拜二": 1,
        "周三": 2, "星期三": 2, "礼拜三": 2,
        "周四": 3, "星期四": 3, "礼拜四": 3,
        "周五": 4, "星期五": 4, "礼拜五": 4,
        "周六": 5, "星期六": 5, "礼拜六": 5, "周末": 5,
        "周日": 6, "星期日": 6, "礼拜日": 6, "星期天": 6,
    }
    
    CN_MONTHS = {
        "一月": 1, "二月": 2, "三月": 3, "四月": 4, "五月": 5, "六月": 6,
        "七月": 7, "八月": 8, "九月": 9, "十月": 10, "十一月": 11, "十二月": 12,
        "1月": 1, "2月": 2, "3月": 3, "4月": 4, "5月": 5, "6月": 6,
        "7月": 7, "8月": 8, "9月": 9, "10月": 10, "11月": 11, "12月": 12,
    }
    
    CN_RELATIVE_PERIODS = {
        "这周": ("week", 0), "本周": ("week", 0), "这个周": ("week", 0),
        "下周": ("week", 1), "下个周": ("week", 1),
        "上周": ("week", -1), "上个周": ("week", -1),
        "这个月": ("month", 0), "本月": ("month", 0),
        "下个月": ("month", 1), "下月": ("month", 1),
        "上个月": ("month", -1), "上月": ("month", -1),
        "今年": ("year", 0), "明年": ("year", 1), "去年": ("year", -1),
    }
    
    # English patterns
    EN_RELATIVE_DAYS = {
        "today": 0, "tomorrow": 1, "day after tomorrow": 2,
        "yesterday": -1, "day before yesterday": -2,
    }
    
    EN_WEEKDAYS = {
        "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
        "weekend": 5,
    }
    
    EN_MONTHS = {
        "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
        "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6,
        "july": 7, "jul": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
        "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
    }
    
    EN_RELATIVE_PERIODS = {
        "this week": ("week", 0), "next week": ("week", 1), "last week": ("week", -1),
        "this month": ("month", 0), "next month": ("month", 1), "last month": ("month", -1),
        "this year": ("year", 0), "next year": ("year", 1), "last year": ("year", -1),
    }
    
    # Recurrence patterns
    RECURRENCE_PATTERNS = {
        "每天": "daily", "天天": "daily", "每日": "daily", "daily": "daily", "every day": "daily",
        "每周": "weekly", "每星期": "weekly", "周周": "weekly", "weekly": "weekly", "every week": "weekly",
        "每月": "monthly", "月月": "monthly", "monthly": "monthly", "every month": "monthly",
        "每年": "yearly", "年年": "yearly", "yearly": "yearly", "every year": "yearly",
        "工作日": "weekdays", "weekdays": "weekdays", "working days": "weekdays",
        "周末": "weekends", "weekends": "weekends",
    }
    
    def __init__(self, reference_date: Optional[date] = None):
        self.ref_date = reference_date or date.today()
        self.ref_datetime = datetime.combine(self.ref_date, time(9, 0))  # Default 9 AM
    
    def parse(self, text: str) -> ParsedDateTime:
        """Parse natural language text into structured datetime."""
        if not text or not text.strip():
            return ParsedDateTime(start=self.ref_datetime, original_text=text, confidence=0.0)
        
        text = text.strip()
        original = text
        text_lower = text.lower()
        
        # Try ISO format first
        iso_result = self._try_parse_iso(text)
        if iso_result:
            iso_result.original_text = original
            return iso_result
        
        # Try Chinese patterns
        cn_result = self._parse_chinese(text)
        if cn_result and cn_result.confidence > 0.7:
            cn_result.original_text = original
            return cn_result
        
        # Try English patterns
        en_result = self._parse_english(text_lower)
        if en_result and en_result.confidence > 0.7:
            en_result.original_text = original
            return en_result
        
        # Fallback: try to extract any date-like pattern
        fallback = self._parse_fallback(text)
        fallback.original_text = original
        return fallback
    
    def _try_parse_iso(self, text: str) -> Optional[ParsedDateTime]:
        """Try parsing ISO format dates."""
        patterns = [
            (r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', "%Y-%m-%dT%H:%M:%S"),
            (r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})', "%Y-%m-%dT%H:%M"),
            (r'^(\d{4}-\d{2}-\d{2})', "%Y-%m-%d"),
            (r'^(\d{4}/\d{2}/\d{2})', "%Y/%m/%d"),
            (r'^(\d{2}:\d{2}:\d{2})', "%H:%M:%S"),
            (r'^(\d{2}:\d{2})', "%H:%M"),
        ]
        
        for pattern, fmt in patterns:
            match = re.match(pattern, text.strip())
            if match:
                try:
                    dt = datetime.strptime(match.group(1), fmt)
                    # If only time, use today's date
                    if fmt in ("%H:%M:%S", "%H:%M"):
                        dt = datetime.combine(self.ref_date, dt.time())
                    return ParsedDateTime(start=dt, granularity=TimeGranularity.EXACT, confidence=1.0)
                except ValueError:
                    continue
        return None
    
    def _parse_chinese(self, text: str) -> Optional[ParsedDateTime]:
        """Parse Chinese natural language date/time."""
        # Check for recurrence first
        recurrence = None
        for pattern, recur_type in self.RECURRENCE_PATTERNS.items():
            if pattern in text:
                recurrence = recur_type
                text = text.replace(pattern, "").strip()
                break
        
        # Relative days
        for word, offset in self.CN_RELATIVE_DAYS.items():
            if word in text:
                target_date = self.ref_date + timedelta(days=offset)
                # Check for time specification
                time_match = re.search(r'(\d{1,2})[点时:：](\d{2})?', text)
                if not time_match and any(c in text for c in "点时:："):
                    # 中文数字时间: "三点" "十二点半" "两点半" → 归一成阿拉伯数字再匹配
                    _cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "零": 0}
                    m = re.search(
                        r"([零一二两三四五六七八九十]{1,3})点(?:([零一二两三四五六七八九十]+|半)?)分?",
                        text,
                    )
                    if m and m.group(1):
                        h_part, m_part = m.group(1), m.group(2)
                        def _to_int(s):
                            if s is None:
                                return ""
                            if s == "半":
                                return "30"
                            s = s.replace("零", "").replace("两", "二")
                            if s and all(ch in "零一二三四五六七八九十" for ch in s):
                                # 十X / X十 / 十 / 普通位值
                                if s in _cn:
                                    return str(_cn[s])
                                if s.startswith("十"):
                                    nxt = _cn.get(s[1:])
                                    return ("1" + str(nxt)) if nxt is not None else "10"
                                if "十" in s:
                                    a, b = s.split("十", 1)
                                    return str(_cn.get(a, 0) * 10 + (_cn.get(b, 0) if b else 0))
                                return ""
                            return s
                        num_text = f"{_to_int(h_part)}点{_to_int(m_part)}"
                        time_match = re.search(r'(\d{1,2})[点时:：](\d{2})?', num_text)
                if time_match:
                    hour = int(time_match.group(1))
                    # 上午/下午/晚上 半日修正: "下午3点"→15:00; "中午12点"→12:00
                    if any(m in text for m in ("下午", "晚", "夜")) and hour < 12:
                        hour += 12
                    minute = int(time_match.group(2)) if time_match.group(2) else 0
                    dt = datetime.combine(target_date, time(hour, minute))
                else:
                    dt = datetime.combine(target_date, time(9, 0))
                
                end_dt = dt + timedelta(hours=1)  # Default 1 hour duration
                return ParsedDateTime(
                    start=dt, end=end_dt,
                    granularity=TimeGranularity.DAY,
                    is_recurring=bool(recurrence),
                    recurrence_pattern=recurrence,
                    confidence=0.95
                )
        
        # Weekday references
        for word, weekday in self.CN_WEEKDAYS.items():
            if word in text:
                days_ahead = weekday - self.ref_date.weekday()
                if "下" in text or "next" in text.lower():
                    days_ahead += 7
                elif "上" in text or "last" in text.lower():
                    days_ahead -= 7
                elif days_ahead <= 0:
                    days_ahead += 7
                
                target_date = self.ref_date + timedelta(days=days_ahead)
                dt = datetime.combine(target_date, time(9, 0))
                return ParsedDateTime(
                    start=dt, end=dt + timedelta(hours=1),
                    granularity=TimeGranularity.DAY,
                    is_recurring=bool(recurrence),
                    recurrence_pattern=recurrence,
                    confidence=0.9
                )
        
        # Relative periods (week, month, year)
        for word, (period, offset) in self.CN_RELATIVE_PERIODS.items():
            if word in text:
                return self._calculate_period_range(period, offset, recurrence)
        
        # Specific date: "10月15日", "10/15", "2026年10月15日"
        date_patterns = [
            (r'(\d{4})年?(\d{1,2})月(\d{1,2})[日号]', lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
            (r'(\d{1,2})月(\d{1,2})[日号]', lambda m: (self.ref_date.year, int(m.group(1)), int(m.group(2)))),
            (r'(\d{1,2})/(\d{1,2})', lambda m: (self.ref_date.year, int(m.group(1)), int(m.group(2)))),
        ]
        
        for pattern, extractor in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    year, month, day = extractor(match)
                    target_date = date(year, month, day)
                    # Check for time
                    time_match = re.search(r'(\d{1,2})[点时:：](\d{2})?', text)
                    if time_match:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2)) if time_match.group(2) else 0
                        dt = datetime.combine(target_date, time(hour, minute))
                    else:
                        dt = datetime.combine(target_date, time(9, 0))
                    return ParsedDateTime(
                        start=dt, end=dt + timedelta(hours=1),
                        granularity=TimeGranularity.DAY,
                        is_recurring=bool(recurrence),
                        recurrence_pattern=recurrence,
                        confidence=0.9
                    )
                except ValueError:
                    continue
        
        return None
    
    def _parse_english(self, text: str) -> Optional[ParsedDateTime]:
        """Parse English natural language date/time."""
        # Check for recurrence
        recurrence = None
        for pattern, recur_type in self.RECURRENCE_PATTERNS.items():
            if pattern in text:
                recurrence = recur_type
                text = text.replace(pattern, "").strip()
                break
        
        # Relative days
        for word, offset in self.EN_RELATIVE_DAYS.items():
            if word in text:
                target_date = self.ref_date + timedelta(days=offset)
                time_match = re.search(r'(\d{1,2}):(\d{2})', text)
                if time_match:
                    dt = datetime.combine(target_date, time(int(time_match.group(1)), int(time_match.group(2))))
                else:
                    dt = datetime.combine(target_date, time(9, 0))
                return ParsedDateTime(
                    start=dt, end=dt + timedelta(hours=1),
                    granularity=TimeGranularity.DAY,
                    is_recurring=bool(recurrence),
                    recurrence_pattern=recurrence,
                    confidence=0.95
                )
        
        # Weekday references
        for word, weekday in self.EN_WEEKDAYS.items():
            if word in text:
                days_ahead = weekday - self.ref_date.weekday()
                if "next" in text:
                    days_ahead += 7
                elif "last" in text:
                    days_ahead -= 7
                elif days_ahead <= 0:
                    days_ahead += 7
                
                target_date = self.ref_date + timedelta(days=days_ahead)
                dt = datetime.combine(target_date, time(9, 0))
                return ParsedDateTime(
                    start=dt, end=dt + timedelta(hours=1),
                    granularity=TimeGranularity.DAY,
                    is_recurring=bool(recurrence),
                    recurrence_pattern=recurrence,
                    confidence=0.9
                )
        
        # Relative periods
        for word, (period, offset) in self.EN_RELATIVE_PERIODS.items():
            if word in text:
                return self._calculate_period_range(period, offset, recurrence)
        
        # Specific date: "Oct 15", "October 15, 2026"
        date_patterns = [
            (r'(\w+)\s+(\d{1,2}),?\s*(\d{4})?', lambda m: (int(m.group(3)) if m.group(3) else self.ref_date.year, self.EN_MONTHS.get(m.group(1).lower(), 1), int(m.group(2)))),
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2)))),
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', lambda m: (int(m.group(3)), int(m.group(1)), int(m.group(2)))),
        ]
        
        for pattern, extractor in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    year, month, day = extractor(match)
                    target_date = date(year, month, day)
                    dt = datetime.combine(target_date, time(9, 0))
                    return ParsedDateTime(
                        start=dt, end=dt + timedelta(hours=1),
                        granularity=TimeGranularity.DAY,
                        is_recurring=bool(recurrence),
                        recurrence_pattern=recurrence,
                        confidence=0.9
                    )
                except ValueError:
                    continue
        
        return None
    
    def _calculate_period_range(self, period: str, offset: int, recurrence: Optional[str]) -> ParsedDateTime:
        """Calculate date range for week/month/year periods."""
        if period == "week":
            # Start of week (Monday)
            week_start = self.ref_date - timedelta(days=self.ref_date.weekday())
            start_date = week_start + timedelta(weeks=offset)
            end_date = start_date + timedelta(days=6)
            start_dt = datetime.combine(start_date, time(0, 0))
            end_dt = datetime.combine(end_date, time(23, 59))
            gran = TimeGranularity.WEEK
        elif period == "month":
            # First day of month
            if offset >= 0:
                start_date = date(self.ref_date.year + (self.ref_date.month + offset - 1) // 12, 
                                 (self.ref_date.month + offset - 1) % 12 + 1, 1)
            else:
                # Handle negative offset
                year = self.ref_date.year
                month = self.ref_date.month + offset
                while month <= 0:
                    month += 12
                    year -= 1
                start_date = date(year, month, 1)
            # Last day of month
            if start_date.month == 12:
                end_date = date(start_date.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(start_date.year, start_date.month + 1, 1) - timedelta(days=1)
            start_dt = datetime.combine(start_date, time(0, 0))
            end_dt = datetime.combine(end_date, time(23, 59))
            gran = TimeGranularity.MONTH
        else:  # year
            start_date = date(self.ref_date.year + offset, 1, 1)
            end_date = date(self.ref_date.year + offset, 12, 31)
            start_dt = datetime.combine(start_date, time(0, 0))
            end_dt = datetime.combine(end_date, time(23, 59))
            gran = TimeGranularity.MONTH
        
        return ParsedDateTime(
            start=start_dt, end=end_dt,
            granularity=gran,
            is_recurring=bool(recurrence),
            recurrence_pattern=recurrence,
            confidence=0.85
        )
    
    def _parse_fallback(self, text: str) -> ParsedDateTime:
        """Fallback parsing for unrecognized patterns."""
        # Try to find any date-like pattern
        patterns = [
            (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', lambda m: datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
            (r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', lambda m: datetime(int(m.group(3)), int(m.group(1)), int(m.group(2)))),
        ]
        
        for pattern, extractor in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    dt = extractor(match)
                    return ParsedDateTime(
                        start=dt, end=dt + timedelta(hours=1),
                        granularity=TimeGranularity.DAY,
                        confidence=0.5
                    )
                except ValueError:
                    continue

        # 最后兜底: 纯时间词("两点半"/"十二点"/"3点20") → 今天该时刻
        bare_hour = re.search(r'(\d{1,2})[点时:：](\d{2})?', text)
        if not bare_hour and "点" in text:
            # 中文数字兜底
            m = re.search(
                r"([零一二两三四五六七八九十]{1,3})点(?:([零一二两三四五六七八九十]+|半)?)分?",
                text or "",
            )
            if m and m.group(1):
                cur = _to_int_cn(m.group(1))
                if cur is not None:
                    mpart = _to_int_cn(m.group(2))
                    bare_hour = re.search(r'(\d{1,2})[点时:：](\d{2})?', f"{cur}点{mpart or ''}")
        if bare_hour:
            hour = int(bare_hour.group(1))
            if any(w in text for w in ("下午", "晚", "夜")) and hour < 12:
                hour += 12
            minute = int(bare_hour.group(2)) if bare_hour.group(2) else 0
            if 0 <= hour < 24 and 0 <= minute < 60:
                dt = datetime.combine(self.ref_date, time(hour, minute))
                return ParsedDateTime(
                    start=dt, end=dt + timedelta(hours=1),
                    granularity=TimeGranularity.EXACT,
                    confidence=0.6
                )

        # Default to today
        return ParsedDateTime(
            start=self.ref_datetime,
            confidence=0.1
        )
    
    def parse_range(self, text: str) -> Tuple[datetime, datetime]:
        """Parse text and return (start, end) datetime tuple for range queries."""
        result = self.parse(text)
        return result.to_iso_range()
    
    def parse_recurring(self, text: str) -> Dict[str, Any]:
        """Parse recurring pattern from text."""
        result = self.parse(text)
        return {
            "is_recurring": result.is_recurring,
            "pattern": result.recurrence_pattern,
            "start": result.start.isoformat(),
            "end": result.recurrence_end.isoformat() if result.recurrence_end else None
        }


# Convenience functions
_default_parser = DateTimeParser()


def parse_datetime(text: str, reference_date: Optional[date] = None) -> ParsedDateTime:
    """Parse natural language date/time text."""
    if reference_date:
        parser = DateTimeParser(reference_date)
        return parser.parse(text)
    return _default_parser.parse(text)


def parse_date_range(text: str, reference_date: Optional[date] = None) -> Tuple[str, str]:
    """Parse text and return (start_iso, end_iso) for database queries."""
    result = parse_datetime(text, reference_date)
    return result.to_iso_range()


def extract_recurrence(text: str) -> Dict[str, Any]:
    """Extract recurrence pattern from text."""
    result = parse_datetime(text)
    return {
        "is_recurring": result.is_recurring,
        "pattern": result.recurrence_pattern,
        "start": result.start.isoformat(),
    }


# Test function
if __name__ == "__main__":
    parser = DateTimeParser()
    
    test_cases = [
        "今天",
        "明天下午3点",
        "后天上午10:30",
        "下周一",
        "下周三下午2点",
        "这个周末",
        "下个月",
        "10月15日",
        "2026年10月15日14:30",
        "每天早上8点",
        "每周一开会",
        "每月1号发工资",
        "today",
        "tomorrow at 3pm",
        "next friday",
        "this weekend",
        "next month",
        "october 15",
        "daily at 9am",
        "weekly on monday",
    ]
    
    for case in test_cases:
        result = parser.parse(case)
        print(f"'{case}' -> {result.start.isoformat()} to {result.end.isoformat() if result.end else 'N/A'} "
              f"({result.granularity.value}, recur={result.recurrence_pattern}, conf={result.confidence:.2f})")

def _to_int_cn(s):
    """中文数字 → int; 非中文数字透传, 失败返回 None。"""
    if s is None:
        return None
    if s == "半":
        return 30
    s = s.replace("零", "").replace("两", "二")
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    if all(ch in "零一二三四五六七八九十" for ch in s):
        cn = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        if s in cn:
            return cn[s]
        if s.startswith("十"):
            nxt = cn.get(s[1:])
            return (10 + nxt) if nxt is not None else 10
        if "十" in s:
            a, b = s.split("十", 1)
            return cn.get(a, 0) * 10 + (cn.get(b, 0) if b else 0)
    return None
