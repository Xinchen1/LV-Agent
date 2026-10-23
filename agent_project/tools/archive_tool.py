"""Archive tool - create/list/extract zip and tar archives safely."""

from __future__ import annotations

import logging
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import BaseTool, ToolResult, TOOLS_REGISTRY

logger = logging.getLogger(__name__)

_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")


def _is_tar(path: Path) -> bool:
    return any(str(path).lower().endswith(s) for s in _TAR_SUFFIXES)


def _is_within(base: Path, target: Path) -> bool:
    try:
        base_r = base.resolve()
        target_r = target.resolve()
        return os.path.commonpath([str(base_r), str(target_r)]) == str(base_r)
    except (ValueError, OSError):
        return False


def _safe_members(names: List[str]) -> List[str]:
    bad = []
    for name in names:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            bad.append(name)
    return bad


class ArchiveTool(BaseTool):
    name = "archive"
    description = (
        "Create, list and extract zip/tar archives. Refuses path-traversal entries on extract "
        "and supports .zip, .tar, .tar.gz/.tgz, .tar.bz2 and .tar.xz."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "extract", "info"],
                "description": "Action to perform.",
            },
            "path": {"type": "string", "description": "Archive file path."},
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files/directories to add (create).",
            },
            "dest": {"type": "string", "description": "Extraction directory (default ./extracted)."},
            "members": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of members to extract.",
            },
        },
        "required": ["action", "path"],
    }

    def execute(self, action: str, path: str = "", **kwargs) -> ToolResult:
        try:
            if not path:
                return ToolResult(success=False, output="", error="path is required")
            handler = {
                "create": self._create,
                "list": self._list,
                "extract": self._extract,
                "info": self._info,
            }.get(action)
            if handler is None:
                return ToolResult(success=False, output="", error=f"Unknown action: {action}")
            return handler(Path(path).expanduser(), **kwargs)
        except Exception as e:
            logger.debug("archive %s failed: %s", action, e)
            return ToolResult(success=False, output="", error=str(e))

    def _create(self, archive: Path, sources: Optional[List[str]] = None, **kwargs) -> ToolResult:
        if not sources:
            return ToolResult(success=False, output="", error="sources is required")
        resolved = [Path(s).expanduser() for s in sources]
        missing = [str(s) for s in resolved if not s.exists()]
        if missing:
            return ToolResult(success=False, output="", error=f"Missing sources: {missing}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        added = 0
        if _is_tar(archive):
            mode = "w:gz" if str(archive).lower().endswith((".gz", ".tgz")) else (
                "w:bz2" if str(archive).lower().endswith((".bz2", ".tbz2")) else (
                    "w:xz" if str(archive).lower().endswith((".xz", ".txz")) else "w"))
            with tarfile.open(archive, mode) as tar:
                for src in resolved:
                    tar.add(src, arcname=src.name, recursive=True)
                    added += 1
        else:
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for src in resolved:
                    if src.is_dir():
                        for file in src.rglob("*"):
                            if file.is_file():
                                zf.write(file, arcname=str(file.relative_to(src.parent)))
                                added += 1
                    else:
                        zf.write(src, arcname=src.name)
                        added += 1
        size = archive.stat().st_size
        return ToolResult(success=True, output=f"Created {archive} with {added} entr{'y' if added == 1 else 'ies'} "
                                            f"({size} bytes)",
                          metadata={"path": str(archive), "entries": added, "bytes": size})

    def _list(self, archive: Path, **kwargs) -> ToolResult:
        if not archive.exists():
            return ToolResult(success=False, output="", error=f"Archive not found: {archive}")
        names: List[str] = []
        if _is_tar(archive):
            with tarfile.open(archive, "r:*") as tar:
                names = tar.getnames()
        else:
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
        if not names:
            return ToolResult(success=True, output="Archive is empty", metadata={"members": []})
        lines = [f"{len(names)} member(s) in {archive}:"] + [f"  {n}" for n in names[:200]]
        if len(names) > 200:
            lines.append(f"  ... and {len(names) - 200} more")
        return ToolResult(success=True, output="\n".join(lines), metadata={"members": names})

    def _info(self, archive: Path, **kwargs) -> ToolResult:
        if not archive.exists():
            return ToolResult(success=False, output="", error=f"Archive not found: {archive}")
        stat = archive.stat()
        data: Dict[str, Any] = {
            "path": str(archive),
            "format": "tar" if _is_tar(archive) else "zip",
            "bytes": stat.st_size,
        }
        if _is_tar(archive):
            with tarfile.open(archive, "r:*") as tar:
                data["members"] = len(tar.getnames())
        else:
            with zipfile.ZipFile(archive) as zf:
                data["members"] = len(zf.namelist())
                bad = zf.testzip()
                data["integrity_ok"] = bad is None
                if bad:
                    data["first_bad_member"] = bad
        lines = [f"{k}: {v}" for k, v in data.items()]
        return ToolResult(success=True, output="\n".join(lines), metadata=data)

    def _extract(self, archive: Path, dest: str = "./extracted",
                 members: Optional[List[str]] = None, **kwargs) -> ToolResult:
        if not archive.exists():
            return ToolResult(success=False, output="", error=f"Archive not found: {archive}")
        target = Path(dest).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        if _is_tar(archive):
            with tarfile.open(archive, "r:*") as tar:
                names = members or tar.getnames()
                bad = _safe_members(names)
                if bad:
                    return ToolResult(success=False, output="",
                                      error=f"Unsafe archive members: {bad[:5]}")
                try:
                    tar.extractall(target, members=names, filter="data")
                except TypeError:
                    for member in tar.getmembers():
                        if member.name in names:
                            tar.extract(member, target)
        else:
            with zipfile.ZipFile(archive) as zf:
                names = members or zf.namelist()
                bad = _safe_members(names)
                if bad:
                    return ToolResult(success=False, output="",
                                      error=f"Unsafe archive members: {bad[:5]}")
                for name in names:
                    zf.extract(name, target)
        return ToolResult(success=True, output=f"Extracted {len(names)} member(s) to {target}",
                          metadata={"dest": str(target), "members": len(names)})


TOOLS_REGISTRY.register(ArchiveTool())
