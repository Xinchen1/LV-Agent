"""Password generation tool - secure random passwords, passphrases, PINs and tokens."""

from __future__ import annotations

import logging
import secrets
import string
import uuid
from typing import List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY

logger = logging.getLogger(__name__)

_AMBIGUOUS = set("Il1O0o|`'\"")
_WORDS = (
    "amber anchor apple arrow autumn badge bamboo basket beacon berry blossom bolt "
    "breeze bridge bronze bubble cabin cactus candle canyon carbon castle cedar cherry "
    "cipher citrus cloud cobalt comet copper coral cosmos cotton crimson crystal dawn "
    "delta desert dolphin dragon dynamo eagle ember falcon feather fern flame forest "
    "fossil frost galaxy garden glacier granite harbor hazel helium horizon island ivory "
    "jade jasmine jewel jungle kettle lagoon lantern lava lemon lily lunar maple marble "
    "meadow meteor mint mirror monsoon moon moss nectar nickel north nova oasis ocean "
    "olive onyx opal orbit orchid otter pearl pebble pepper pine planet plum polar prism "
    "quartz quill raven reef ripple river robin ruby saffron sage salmon satin scarlet "
    "shadow shell silver sky solar spark spruce star stone storm summit sunset surf "
    "thunder tide timber topaz tulip tundra turbine twilight umber valley velvet violet "
    "walnut wave willow winter wolf yarrow zephyr zinc"
).split()


class PasswordGenTool(BaseTool):
    name = "password_gen"
    description = (
        "Generate strong secrets locally: random passwords with character-class control, "
        "memorable passphrases, PINs, URL-safe tokens and UUIDs. Nothing is stored or sent."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["password", "passphrase", "pin", "token", "uuid"],
                "description": "Kind of secret to generate.",
            },
            "length": {"type": "integer", "description": "Length for password/pin/token."},
            "count": {"type": "integer", "description": "How many to generate (default 1)."},
            "lowercase": {"type": "boolean", "description": "Include lowercase (default true)."},
            "uppercase": {"type": "boolean", "description": "Include uppercase (default true)."},
            "digits": {"type": "boolean", "description": "Include digits (default true)."},
            "symbols": {"type": "boolean", "description": "Include symbols (default true)."},
            "exclude_ambiguous": {"type": "boolean", "description": "Drop look-alike characters."},
            "word_count": {"type": "integer", "description": "Words in passphrase (default 4)."},
            "separator": {"type": "string", "description": "Passphrase separator (default -)."},
            "capitalize": {"type": "boolean", "description": "Capitalize passphrase words (default true)."},
        },
        "required": ["action"],
    }

    def execute(self, action: str, **kwargs) -> ToolResult:
        try:
            if action == "password":
                return self._passwords(**kwargs)
            if action == "passphrase":
                return self._passphrases(**kwargs)
            if action == "pin":
                return self._pins(**kwargs)
            if action == "token":
                return self._tokens(**kwargs)
            if action == "uuid":
                count = max(1, min(50, int(kwargs.get("count") or 1)))
                values = [str(uuid.uuid4()) for _ in range(count)]
                return ToolResult(success=True, output="\n".join(values),
                                  metadata={"kind": "uuid", "count": count})
            return ToolResult(success=False, output="", error=f"Unknown action: {action}")
        except Exception as e:
            logger.debug("password_gen %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _passwords(self, length: int = 20, count: int = 1, lowercase: bool = True,
                   uppercase: bool = True, digits: bool = True, symbols: bool = True,
                   exclude_ambiguous: bool = False, **kwargs) -> ToolResult:
        pools: List[str] = []
        if lowercase:
            pools.append(string.ascii_lowercase)
        if uppercase:
            pools.append(string.ascii_uppercase)
        if digits:
            pools.append(string.digits)
        if symbols:
            pools.append("!@#$%^&*()-_=+[]{};:,.?/")
        if not pools:
            return ToolResult(success=False, output="", error="Select at least one character class")
        if exclude_ambiguous:
            pools = ["".join(ch for ch in pool if ch not in _AMBIGUOUS) for pool in pools]
            pools = [pool for pool in pools if pool]
        length = max(len(pools), min(256, int(length or 20)))
        count = max(1, min(50, int(count or 1)))
        alphabet = "".join(pools)
        generated = []
        for _ in range(count):
            while True:
                candidate = [secrets.choice(pool) for pool in pools]
                candidate += [secrets.choice(alphabet) for _ in range(length - len(candidate))]
                secrets.SystemRandom().shuffle(candidate)
                value = "".join(candidate)
                if all(any(ch in pool for ch in value) for pool in pools):
                    generated.append(value)
                    break
        return ToolResult(success=True, output="\n".join(generated),
                          metadata={"kind": "password", "length": length, "count": count,
                                    "classes": len(pools)})

    def _passphrases(self, word_count: int = 4, separator: str = "-", capitalize: bool = True,
                     count: int = 1, **kwargs) -> ToolResult:
        word_count = max(2, min(12, int(word_count or 4)))
        count = max(1, min(50, int(count or 1)))
        sep = separator if separator is not None else "-"
        values = []
        for _ in range(count):
            words = [secrets.choice(_WORDS) for _ in range(word_count)]
            if capitalize:
                words = [w.capitalize() for w in words]
            values.append(sep.join(words))
        return ToolResult(success=True, output="\n".join(values),
                          metadata={"kind": "passphrase", "words": word_count, "count": count,
                                    "entropy_bits_approx": round(word_count * 9.2, 1)})

    def _pins(self, length: int = 6, count: int = 1, **kwargs) -> ToolResult:
        length = max(4, min(12, int(length or 6)))
        count = max(1, min(50, int(count or 1)))
        values = ["".join(secrets.choice(string.digits) for _ in range(length)) for _ in range(count)]
        return ToolResult(success=True, output="\n".join(values),
                          metadata={"kind": "pin", "length": length, "count": count})

    def _tokens(self, length: int = 32, count: int = 1, **kwargs) -> ToolResult:
        length = max(8, min(128, int(length or 32)))
        count = max(1, min(50, int(count or 1)))
        values = [secrets.token_urlsafe(length)[:length] for _ in range(count)]
        return ToolResult(success=True, output="\n".join(values),
                          metadata={"kind": "token", "length": length, "count": count})


TOOLS_REGISTRY.register(PasswordGenTool())
