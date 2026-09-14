"""
Training Data Module - 把经验库转成 SFT 训练集并(可选)提交云端微调.

闭环:
  experience_buffer (ChromaDB/JSON, 真实执行轨迹)
      → 过滤 (success=1, final_answer 非空)
      → 格式化 (OpenAI 兼容 messages JSONL)
      → 训练元数据 manifest.json (数量/任务类型分布/时间戳)
      → (可选) OpenAI-compatible fine-tuning API 提交

默认只导出数据集(无额外依赖, 可审计、可复算);
train_mode="api" 时才要求 openai SDK 与有效 api_key/base_url。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("training")


class SFTDatasetExporter:
    """将 ExperienceBuffer 的经验转为 OpenAI 兼容的 chat-format SFT JSONL.

    Format (每行一个 JSON):
      {"messages": [{"role":"system",...}, {"role":"user","content":task}, ...,
                    {"role":"assistant","content":final_answer}]}

    只导出有真实 final_answer 的案例(无空壳数据);
    支持 task_type 过滤与 train/valid 划分。
    """

    def __init__(
        self,
        system_prompt: str = "You are LV Agent, a helpful AI assistant that thinks step by step and uses tools when needed.",
        include_failures: bool = False,
        min_episodes: int = 5,
        train_ratio: float = 0.8,
    ):
        self.system_prompt = system_prompt
        self.include_failures = include_failures
        self.min_episodes = min_episodes
        self.train_ratio = train_ratio

    def _is_usable(self, exp: Any) -> bool:
        """一条经验是否可用于 SFT(有任务、有答案、成功或显式允许失败)."""
        if exp is None:
            return False
        task = getattr(exp, "task", "") or ""
        traj = getattr(exp, "trajectory", None) or {}
        if not isinstance(traj, dict) or not task.strip():
            return False
        if not self.include_failures and not traj.get("success", False):
            return False
        answer = self._extract_answer(traj)
        return bool(answer and str(answer).strip())

    @staticmethod
    def _extract_answer(traj: Dict[str, Any]) -> str:
        """从轨迹提取最终答案(优先 final_answer, 兜底最后一条 observation/output)."""
        ans = traj.get("final_answer") or ""
        if isinstance(ans, str) and ans.strip():
            return ans
        if isinstance(ans, (tuple, list)) and ans:
            return str(ans[0])
        obs = traj.get("observations") or []
        if obs:
            last = obs[-1]
            if isinstance(last, dict):
                out = last.get("output") or last.get("content") or ""
                if isinstance(out, str) and out.strip():
                    return out
            elif isinstance(last, str) and last.strip():
                return last
        return ""

    @staticmethod
    def _trajectory_tool_note(traj: Dict[str, Any], max_actions: int = 6) -> str:
        """把工具调用序列压成一段提示, 引导模型复用相似动作链."""
        actions = traj.get("actions") or []
        seq = []
        for a in actions[:max_actions]:
            if isinstance(a, dict):
                name = a.get("tool_name") or a.get("name") or ""
                if name:
                    seq.append(name)
            elif isinstance(a, str):
                seq.append(a)
        return "，".join(seq) if seq else ""

    def to_messages(self, exp: Any) -> Optional[Dict[str, Any]]:
        """单条经验 → OpenAI chat messages dict."""
        if not self._is_usable(exp):
            return None
        traj = exp.trajectory or {}
        user = (getattr(exp, "task", "") or "").strip()
        answer = self._extract_answer(traj)
        sys = self.system_prompt
        tool_note = self._trajectory_tool_note(traj)
        if tool_note:
            sys += (
                "\n\n参考工具链(来自相似成功案例, 仅作提示): "
                f"{tool_note}"
            )
        return {
            "messages": [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ]
        }

    def export(
        self,
        experiences: List[Any],
        output_dir: str = "./data/train",
        task_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """批量导出为 JSONL, 返回统计报告.

        Returns:
            {count, train_count, valid_count, output_dir, train_file,
             valid_file, manifest_file, task_types, skipped, exported_at}
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 过滤 + 可选 task_type 过滤
        usable = []
        skipped = 0
        task_type_counts: Dict[str, int] = {}
        for exp in experiences:
            if not self._is_usable(exp):
                skipped += 1
                continue
            tt = getattr(exp, "task_type", "unknown") or "unknown"
            if task_types and tt not in task_types:
                skipped += 1
                continue
            usable.append(exp)
            task_type_counts[tt] = task_type_counts.get(tt, 0) + 1

        if len(usable) < self.min_episodes:
            return {
                "count": 0,
                "train_count": 0,
                "valid_count": 0,
                "output_dir": str(out),
                "train_file": "",
                "valid_file": "",
                "manifest_file": str(out / "manifest.json"),
                "task_types": task_type_counts,
                "skipped": skipped,
                "status": "insufficient_episodes",
                "exported_at": datetime.now().isoformat(),
            }

        # 按时间戳排序后切分 (旧样本作训练, 新样本作验证 → 验证更接近当前行为)
        usable_sorted = sorted(
            usable,
            key=lambda e: getattr(e, "timestamp", "") or "",
        )
        split = max(1, int(len(usable_sorted) * self.train_ratio))
        train_rows = [self.to_messages(e) for e in usable_sorted[:split]]
        valid_rows = [self.to_messages(e) for e in usable_sorted[split:]]
        train_rows = [r for r in train_rows if r]
        valid_rows = [r for r in valid_rows if r]

        train_file = out / "sft_train.jsonl"
        valid_file = out / "sft_valid.jsonl"
        self._write_jsonl(train_file, train_rows)
        self._write_jsonl(valid_file, valid_rows)

        manifest = {
            "format": "openai-chat",
            "created_at": datetime.now().isoformat(),
            "count": len(train_rows) + len(valid_rows),
            "train_count": len(train_rows),
            "valid_count": len(valid_rows),
            "task_types": task_type_counts,
            "skipped": skipped,
            "include_failures": self.include_failures,
            "system_prompt": self.system_prompt,
        }
        manifest_file = out / "manifest.json"
        manifest_file.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {
            "count": manifest["count"],
            "train_count": manifest["train_count"],
            "valid_count": manifest["valid_count"],
            "output_dir": str(out),
            "train_file": str(train_file),
            "valid_file": str(valid_file),
            "manifest_file": str(manifest_file),
            "task_types": task_type_counts,
            "skipped": skipped,
            "status": "ok",
            "exported_at": manifest["created_at"],
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class CloudFineTuner:
    """(可选) 通过 OpenAI-compatible fine-tuning API 提交训练.

    仅当 train_mode="api" 且提供 api_key/base_url 时使用;
    否则返回 reason="not_configured", 不产生任何外部调用。
    """

    def submit(self, train_file: str, *, model: str, api_key: str, base_url: str) -> Dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError:
            return {"status": "skipped", "reason": "openai_sdk_missing"}

        try:
            client = OpenAI(api_key=api_key, base_url=base_url or None, timeout=120)
            with open(train_file, "rb") as f:
                upload = client.files.create(file=f, purpose="fine-tune")
            job = client.fine_tuning.jobs.create(
                training_file=upload.id, model=model,
            )
            return {"status": "submitted", "file_id": upload.id, "job_id": job.id}
        except Exception as e:  # noqa: BLE001
            logger.warning("fine-tune submit failed: %s", e)
            return {"status": "failed", "reason": str(e)}


def run_training_pipeline(
    experiences: List[Any],
    *,
    output_dir: str = "./data/train",
    task_types: Optional[List[str]] = None,
    train_mode: str = "export",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    min_episodes: int = 5,
    include_failures: bool = False,
) -> Dict[str, Any]:
    """自进化训练管线的统一入口: 导出数据集 → (可选) 提交云端微调.

    返回结构化报告, 供 agent 记录到日志 / UI 展示。
    """
    exporter = SFTDatasetExporter(
        include_failures=include_failures,
        min_episodes=min_episodes,
    )
    report = exporter.export(experiences, output_dir=output_dir, task_types=task_types)

    if report.get("status") != "ok":
        return report

    report["train_mode"] = train_mode
    if train_mode == "api" and report.get("train_file"):
        ft = CloudFineTuner()
        report["finetune"] = ft.submit(
            report["train_file"],
            model=model or "",
            api_key=api_key or "",
            base_url=base_url or "",
        )
    else:
        report["finetune"] = {
            "status": "skipped",
            "reason": "train_mode=export (set train_mode=api + api_key to submit)",
        }
    report["ran_at"] = time.time()
    return report