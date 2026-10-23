"""Smoke/contract tests for the expanded personal-agent tool set."""

import json
from datetime import date, timedelta

import pytest

from agent_project.tools import TOOLS_REGISTRY
from agent_project.tools.archive_tool import ArchiveTool
from agent_project.tools.bookmarks_tool import BookmarksTool
from agent_project.tools.clock_tool import ClockTool
from agent_project.tools.desktop_tool import DesktopTool
from agent_project.tools.flashcards_tool import FlashcardsTool
from agent_project.tools.goals_tool import GoalsTool
from agent_project.tools.habits_tool import HabitsTool
from agent_project.tools.journal_tool import JournalTool
from agent_project.tools.meals_tool import MealsTool
from agent_project.tools.password_gen_tool import PasswordGenTool
from agent_project.tools.shopping_tool import ShoppingTool
from agent_project.tools.subscriptions_tool import SubscriptionsTool
from agent_project.tools.system_info_tool import SystemInfoTool
from agent_project.tools.text_transform_tool import TextTransformTool
from agent_project.tools.unit_convert_tool import UnitConvertTool

NEW_TOOLS = [
    "journal", "habits", "bookmarks", "shopping", "subscriptions", "goals",
    "flashcards", "meals", "text_transform", "archive", "desktop", "clock",
    "system_info", "unit_convert", "password_gen",
]


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LV_AGENT_ASSISTANT_DATA_DIR", str(tmp_path / "assistant"))
    return tmp_path


def test_all_new_tools_registered():
    names = set(TOOLS_REGISTRY.list_tools())
    missing = [name for name in NEW_TOOLS if name not in names]
    assert not missing, f"未注册工具: {missing}"


def test_journal_create_list_search_and_mood():
    tool = JournalTool()
    created = tool.execute(action="create", content="今天完成个人助理改造",
                           mood="great", gratitude="团队", tags=["work"])
    assert created.success
    entry_id = created.metadata["entry_id"]

    listed = tool.execute(action="list")
    assert listed.success and entry_id in [e["id"] for e in listed.metadata["entries"]]

    found = tool.execute(action="search", query="个人助理")
    assert found.success and found.metadata["entries"]

    stats = tool.execute(action="mood_stats", days=7)
    assert stats.success and stats.metadata["entries"] == 1
    assert stats.metadata["average"] == 5.0

    updated = tool.execute(action="update", entry_id=entry_id, content="改好了")
    assert updated.success
    deleted = tool.execute(action="delete", entry_id=entry_id)
    assert deleted.success


def test_habits_streak_and_check():
    tool = HabitsTool()
    created = tool.execute(action="create", name="喝水")
    assert created.success
    habit_id = created.metadata["habit_id"]

    today = date.today()
    for offset in (0, 1, 2):
        day = (today - timedelta(days=offset)).isoformat()
        assert tool.execute(action="check", habit_id=habit_id, date=day).success

    listed = tool.execute(action="list")
    habit = listed.metadata["habits"][0]
    assert habit["done_today"] is True
    assert habit["current"] == 3

    stats = tool.execute(action="stats", habit_id=habit_id, days=7)
    assert stats.success and stats.metadata["current_streak"] == 3

    assert tool.execute(action="uncheck", habit_id=habit_id,
                        date=today.isoformat()).success
    assert tool.execute(action="delete", habit_id=habit_id).success


def test_bookmarks_add_search_archive():
    tool = BookmarksTool()
    added = tool.execute(action="add", url="example.com", title="Example", tags=["dev"])
    assert added.success
    bookmark_id = added.metadata["bookmark_id"]
    url = added.metadata["url"]
    assert url.startswith("https://")

    duplicate = tool.execute(action="add", url=url)
    assert not duplicate.success

    listed = tool.execute(action="list")
    assert listed.success and listed.metadata["bookmarks"][0]["tags"] == ["dev"]

    assert tool.execute(action="search", query="dev").success
    assert tool.execute(action="read", bookmark_id=bookmark_id).success
    assert tool.execute(action="archive", bookmark_id=bookmark_id).success
    assert tool.execute(action="list").metadata["bookmarks"] == []
    assert len(tool.execute(action="list", include_archived=True).metadata["bookmarks"]) == 1
    assert tool.execute(action="stats").metadata["archived"] == 1


def test_shopping_add_check_and_estimate():
    tool = ShoppingTool()
    first = tool.execute(action="add", name="牛奶", quantity=2, unit="盒", price=6.5)
    second = tool.execute(action="add", name="面包", quantity=1, price=10)
    assert first.success and second.success

    estimate = tool.execute(action="estimate")
    assert estimate.success and estimate.metadata["total"] == 23.0

    assert tool.execute(action="check", item_id=first.metadata["item_id"]).success
    listed = tool.execute(action="list")
    assert len(listed.metadata["items"]) == 1
    assert tool.execute(action="clear_checked").metadata["removed"] == 1


def test_subscriptions_upcoming_and_monthly_total():
    tool = SubscriptionsTool()
    next_due = (date.today() + timedelta(days=5)).isoformat()
    created = tool.execute(action="add", name="视频会员", amount=120, cycle="yearly",
                           next_due=next_due)
    assert created.success

    upcoming = tool.execute(action="upcoming", days=30)
    assert upcoming.success and upcoming.metadata["upcoming"][0]["name"] == "视频会员"

    total = tool.execute(action="total")
    assert total.success and total.metadata["totals"]["CNY"] == 10.0

    sid = created.metadata["subscription_id"]
    assert tool.execute(action="pause", subscription_id=sid).success
    assert tool.execute(action="upcoming", days=30).metadata["upcoming"] == []
    assert tool.execute(action="delete", subscription_id=sid).success


def test_goals_progress_and_milestones():
    tool = GoalsTool()
    created = tool.execute(action="create", title="读完 12 本书",
                           target_value=12, unit="本", due_date="2026-12-31")
    assert created.success
    goal_id = created.metadata["goal_id"]

    milestone = tool.execute(action="add_milestone", goal_id=goal_id,
                             milestone_title="读完第一本")
    assert milestone.success
    assert tool.execute(action="complete_milestone",
                        milestone_id=milestone.metadata["milestone_id"]).success

    progress = tool.execute(action="progress", goal_id=goal_id, value=6)
    assert progress.success and progress.metadata["progress_percent"] == 50.0

    achieved = tool.execute(action="progress", goal_id=goal_id, delta=6)
    assert achieved.metadata["status"] == "achieved"

    review = tool.execute(action="review")
    assert review.metadata["achieved"] == 1


def test_flashcards_review_schedule():
    tool = FlashcardsTool()
    added = tool.execute(action="add", front="2+2", back="4", deck="math")
    assert added.success
    card_id = added.metadata["card_id"]

    due = tool.execute(action="due")
    assert due.success and due.metadata["due"]

    good = tool.execute(action="review", card_id=card_id, rating="good")
    assert good.success and good.metadata["interval_days"] == 1
    assert good.metadata["due_date"] > date.today().isoformat()

    again = tool.execute(action="review", card_id=card_id, rating="again")
    assert again.metadata["interval_days"] == 0
    assert again.metadata["due_date"] == date.today().isoformat()

    assert tool.execute(action="decks").metadata["decks"]["math"] == 1


def test_meals_recipe_plan_and_shopping_list():
    tool = MealsTool()
    created = tool.execute(
        action="add_recipe",
        name="番茄炒蛋",
        ingredients=[
            {"name": "番茄", "quantity": 2, "unit": "个"},
            {"name": "鸡蛋", "quantity": 3, "unit": "个"},
        ],
        steps="炒",
        servings=2,
    )
    assert created.success
    recipe_id = created.metadata["recipe_id"]

    assert tool.execute(action="plan", plan_date="2026-01-05",
                        meal_type="dinner", recipe_id=recipe_id).success
    week = tool.execute(action="plan_week", week_start="2026-01-05")
    assert week.success and week.metadata["plans"][0]["recipe_name"] == "番茄炒蛋"

    shopping = tool.execute(action="shopping_list", week_start="2026-01-05")
    assert shopping.success
    names = {item["name"]: item["quantity"] for item in shopping.metadata["items"]}
    assert names == {"番茄": 2.0, "鸡蛋": 3.0}


def test_text_transform_actions():
    tool = TextTransformTool()
    assert tool.execute(action="hash", text="abc").output == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    encoded = tool.execute(action="base64_encode", text="你好")
    assert tool.execute(action="base64_decode", text=encoded.output).output == "你好"
    assert tool.execute(action="slug", text="Hello World 你好").output == "hello-world"
    assert tool.execute(action="case", text="hello world", mode="camel").output == "HelloWorld"
    formatted = tool.execute(action="json_format", text='{"b":1,"a":2}')
    assert json.loads(formatted.output) == {"b": 1, "a": 2}
    counts = tool.execute(action="count", text="one two\n三")
    assert counts.metadata["words"] == 3
    extracted = tool.execute(action="regex_extract", text="a1b22", pattern=r"\d+")
    assert extracted.output.splitlines() == ["1", "22"]
    assert tool.execute(action="regex_replace", text="a1", pattern=r"\d", replacement="#").output == "a#"


def test_unit_convert_temperature_and_data():
    tool = UnitConvertTool()
    temp = tool.execute(action="convert", value=100, from_unit="c", to_unit="f")
    assert temp.success and temp.metadata["result"] == 212.0
    data = tool.execute(action="convert", value=1, from_unit="gib", to_unit="mib")
    assert data.metadata["result"] == 1024.0
    incompatible = tool.execute(action="convert", value=1, from_unit="kg", to_unit="m")
    assert not incompatible.success
    assert tool.execute(action="list", category="length").success


def test_clock_now_convert_and_workdays():
    tool = ClockTool()
    now = tool.execute(action="now", timezone="Asia/Shanghai")
    assert now.success and now.metadata["timezone"] == "Asia/Shanghai"
    converted = tool.execute(action="convert", value="2026-01-01T00:00:00",
                             from_timezone="UTC", to_timezone="Asia/Shanghai")
    assert converted.success and converted.output.startswith("2026-01-01T08:00:00")
    diff = tool.execute(action="diff", start="2026-01-01", end="2026-01-11")
    assert diff.metadata["days"] == 10
    workdays = tool.execute(action="workdays", start="2026-01-05", end="2026-01-09")
    assert workdays.metadata["business_days"] == 5
    countdown = tool.execute(action="countdown", target="2030-01-01")
    assert countdown.success and not countdown.metadata["overdue"]


def test_password_gen_classes_and_length():
    tool = PasswordGenTool()
    result = tool.execute(action="password", length=24, count=3, exclude_ambiguous=True)
    assert result.success
    values = result.output.splitlines()
    assert len(values) == 3
    for value in values:
        assert len(value) == 24
        assert not any(ch in "Il1O0o|`'\"<" for ch in value)
        assert any(ch.islower() for ch in value)
        assert any(ch.isupper() for ch in value)
        assert any(ch.isdigit() for ch in value)
        assert any(not ch.isalnum() for ch in value)

    phrase = tool.execute(action="passphrase", word_count=5, separator="_")
    assert len(phrase.output.split("_")) == 5
    pin = tool.execute(action="pin", length=6)
    assert pin.output.isdigit() and len(pin.output) == 6
    uuid_value = tool.execute(action="uuid").output
    assert len(uuid_value.split("-")) == 5


def test_archive_create_list_extract(tmp_path):
    tool = ArchiveTool()
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("hello", encoding="utf-8")
    archive = tmp_path / "bundle.zip"

    created = tool.execute(action="create", path=str(archive), sources=[str(source)])
    assert created.success and archive.exists()

    listed = tool.execute(action="list", path=str(archive))
    assert listed.success and "src/a.txt" in listed.metadata["members"]

    dest = tmp_path / "out"
    extracted = tool.execute(action="extract", path=str(archive), dest=str(dest))
    assert extracted.success
    assert (dest / "src" / "a.txt").read_text(encoding="utf-8") == "hello"

    info = tool.execute(action="info", path=str(archive))
    assert info.success and info.metadata["format"] == "zip"


def test_archive_rejects_traversal(tmp_path):
    import zipfile

    tool = ArchiveTool()
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")
    result = tool.execute(action="extract", path=str(archive), dest=str(tmp_path / "out"))
    assert not result.success
    assert "Unsafe" in result.error


def test_system_info_and_desktop_capabilities():
    system = SystemInfoTool()
    summary = system.execute(action="summary")
    assert summary.success and summary.metadata["os"]
    disk = system.execute(action="disk")
    assert disk.success and 0 <= disk.metadata["percent_used"] <= 100

    desktop = DesktopTool()
    capabilities = desktop.execute(action="capabilities")
    assert capabilities.success and "platform" in capabilities.metadata


def test_config_enables_personal_tools():
    from pathlib import Path

    from agent_project.config import load_config

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    config = load_config(str(config_path))
    enabled = set(config.tools.enabled_tools)
    missing = [name for name in NEW_TOOLS if name not in enabled]
    assert not missing, f"config.yaml tools.enabled 缺少: {missing}"
