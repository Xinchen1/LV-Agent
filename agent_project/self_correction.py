"""
Self-Correction Module - Real-time quality control and adaptation
Monitors agent performance and triggers corrections
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from statistics import mean, stdev
import logging

from pydantic import BaseModel, Field


# ============ Types ============

@dataclass
class QualityMetrics:
    """Quality metrics for an agent execution"""
    success: bool
    confidence_score: float  # 0-1
    coherence_score: float  # 0-1
    efficiency_score: float  # 0-1
    tool_usage_count: int
    reasoning_depth: int
    duration_ms: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def overall_score(self) -> float:
        """Compute overall quality score"""
        weights = {
            'success': 0.4,
            'confidence': 0.2,
            'coherence': 0.2,
            'efficiency': 0.2
        }
        return (
            weights['success'] * float(self.success) +
            weights['confidence'] * self.confidence_score +
            weights['coherence'] * self.coherence_score +
            weights['efficiency'] * self.efficiency_score
        )


@dataclass
class CorrectionAction:
    """An action to correct agent behavior"""
    action_type: str
    priority: int  # 1=critical, 2=high, 3=medium, 4=low
    description: str
    reasoning: str
    recommended_loops: Optional[int] = None
    strategy_override: Optional[str] = None
    tool_adjustments: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8


# ============ Quality Evaluator ============

class QualityEvaluator:
    """
    Evaluates execution quality using multiple signals
    """

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger("QualityEvaluator")

        # Thresholds
        self.low_confidence_threshold = config.get('low_confidence_threshold', 0.6)
        self.high_error_threshold = config.get('high_error_threshold', 0.3)
        self.inefficiency_threshold = config.get('inefficiency_threshold', 0.4)

    def evaluate(
        self,
        trace: Any,  # ReasoningTrace
        tools_used: List[str],
        expected_tools: Optional[List[str]] = None
    ) -> QualityMetrics:
        """
        Evaluate execution quality

        Args:
            trace: Reasoning trace from execution
            tools_used: List of tools that were actually used
            expected_tools: Tools that should have been used (if known)

        Returns:
            QualityMetrics object
        """
        errors = []
        warnings = []

        # 1. Success indicator
        success = getattr(trace, 'success', False) if trace is not None else False

        # 2. Confidence score (from trace or heuristic)
        confidence = getattr(trace, 'quality_score', None) if trace is not None else None
        if confidence is None:
            confidence = self._estimate_confidence(trace)

        # 3. Coherence (logical flow of reasoning)
        coherence = self._assess_coherence(trace)
        if coherence < 0.7:
            warnings.append("Reasoning coherence is low - possible logical gaps")

        # 4. Efficiency (steps/duration vs outcome)
        efficiency = self._assess_efficiency(trace)
        if efficiency < 0.5:
            warnings.append("Inefficient reasoning - too many steps or slow")

        # 5. Tool usage analysis
        if expected_tools:
            missing = set(expected_tools) - set(tools_used)
            if missing:
                warnings.append(f"Expected tools not used: {missing}")
                # Adjust efficiency down
                efficiency *= 0.8

        # 6. Error detection from trace
        if trace is not None and getattr(trace, 'metadata', {}).get('error'):
            errors.append(trace.metadata['error'])

        reasoning_depth = 0
        if trace is not None:
            reasoning_depth = getattr(trace, 'total_loops', 0) or len(getattr(trace, 'steps', []))

        return QualityMetrics(
            success=success,
            confidence_score=confidence,
            coherence_score=coherence,
            efficiency_score=efficiency,
            tool_usage_count=len(tools_used),
            reasoning_depth=reasoning_depth,
            duration_ms=getattr(trace, 'duration_ms', 0) if trace is not None else 0,
            errors=errors,
            warnings=warnings
        )

    def _estimate_confidence(self, trace: Any) -> float:
        """Estimate confidence from trace characteristics"""
        if trace is None or not hasattr(trace, 'steps') or not trace.steps:
            return 0.5

        # Average step confidence
        avg_step_conf = mean(getattr(s, 'confidence', 0.8) for s in trace.steps)

        # Length factor (shorter = more confident, up to point)
        length_factor = 1.0
        if len(trace.steps) > 15:
            length_factor = 0.8
        elif len(trace.steps) < 3:
            length_factor = 1.0

        return min(avg_step_conf * length_factor, 1.0)

    def _assess_coherence(self, trace: Any) -> float:
        """Assess logical coherence of reasoning"""
        if trace is None or not hasattr(trace, 'steps') or len(trace.steps) < 2:
            return 1.0

        # Heuristics:
        # - Transition words usage (First, Next, Then, Therefore, etc.)
        # - Non-repetition (steps not repeating same content)
        # - Logical connectors

        coherence = 0.8  # baseline

        # Check for repetition
        contents = [s.content.lower() for s in trace.steps]
        unique_ratio = len(set(contents)) / len(contents)
        if unique_ratio < 0.6:
            coherence -= 0.2

        # Check for conclusion indicators
        last_step = trace.steps[-1].content.lower() if trace.steps else ""
        conclusion_words = ['therefore', 'thus', 'hence', 'conclude', 'final answer', 'in summary']
        if not any(cw in last_step for cw in conclusion_words):
            coherence -= 0.1

        return max(0.0, coherence)

    def _assess_efficiency(self, trace: Any) -> float:
        """Assess efficiency of reasoning"""
        if trace is None or not hasattr(trace, 'steps'):
            return 0.7

        num_steps = len(trace.steps)
        duration = getattr(trace, 'duration_ms', num_steps * 1000)

        # Efficiency baseline: 5-10 steps in < 10s is efficient
        if 3 <= num_steps <= 12 and duration < 10000:
            return 0.9
        elif num_steps > 20 or duration > 30000:
            return 0.4
        else:
            return 0.7


# ============ Corrective Action Generator ============

class CorrectionGenerator:
    """
    Generate corrective actions based on quality issues
    """

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger("CorrectionGenerator")

    def generate_corrections(
        self,
        metrics: QualityMetrics,
        task: str,
        current_loops: int,
        current_strategy: str
    ) -> List[CorrectionAction]:
        """
        Generate list of corrective actions
        """
        actions = []

        # 1. Low success rate
        if not metrics.success:
            actions.append(CorrectionAction(
                action_type="increase_reasoning_depth",
                priority=1,
                description="Increase thinking loops significantly",
                reasoning="Task failed with current depth",
                recommended_loops=min(current_loops * 2, 32),
                confidence=0.8
            ))

            actions.append(CorrectionAction(
                action_type="switch_strategy",
                priority=1,
                description="Switch to more robust reasoning strategy",
                reasoning="Default strategy insufficient for this task",
                strategy_override="self_consistency",  # use multiple samples
                confidence=0.7
            ))

        # 2. Low confidence
        if metrics.confidence_score < 0.6:
            actions.append(CorrectionAction(
                action_type="verify_answer",
                priority=2,
                description="Run verification step after reasoning",
                reasoning="Low confidence indicates possible errors",
                confidence=0.9
            ))

        # 3. Inefficiency
        if metrics.efficiency_score < 0.5:
            actions.append(CorrectionAction(
                action_type="increase_convergence_horizon",
                priority=3,
                description="Enable early stopping based on answer convergence",
                reasoning="Too many unnecessary loops detected",
                tool_adjustments={"early_stop": True, "convergence_threshold": 0.9},
                confidence=0.7
            ))

        # 4. Poor coherence
        if metrics.coherence_score < 0.6:
            actions.append(CorrectionAction(
                action_type="use_structured_prompt",
                priority=2,
                description="Switch to structured reasoning template",
                reasoning="Unstructured reasoning causing incoherence",
                tool_adjustments={"template": "react"},  # use ReAct
                confidence=0.8
            ))

        # 5. Tool misuse
        if "Expected tools not used" in str(metrics.errors) or "Expected tools not used" in str(metrics.warnings):
            actions.append(CorrectionAction(
                action_type="tool_reminder",
                priority=3,
                description="Ensure required tools are available and prompted",
                reasoning="Missing critical tool usage",
                confidence=0.9
            ))

        return actions

    def prioritize_actions(self, actions: List[CorrectionAction]) -> List[CorrectionAction]:
        """Sort actions by priority"""
        return sorted(actions, key=lambda a: a.priority)


# ============ Adaptive Controller ============

class AdaptiveController:
    """
    Real-time adaptation of agent parameters based on performance
    """

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger("AdaptiveController")

        # Performance history
        self.recent_metrics: List[QualityMetrics] = []
        self.correction_history: List[CorrectionAction] = []
        self.max_history = 100

        # Current adjustment factors
        self.loop_multiplier = 1.0
        self.temperature_adjustment = 0.0
        self.strategy_override = None

        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict() if hasattr(config, 'dict') else config
        self.evaluator = QualityEvaluator(config_dict)
        self.generator = CorrectionGenerator(config)

    def observe(
        self,
        metrics: QualityMetrics,
        task: str,
        strategy: str
    ) -> Optional[CorrectionAction]:
        """
        Observe execution and decide on corrections
        Returns most important correction or None if no action needed
        """
        self.recent_metrics.append(metrics)

        # Keep history bounded
        if len(self.recent_metrics) > self.max_history:
            self.recent_metrics = self.recent_metrics[-self.max_history:]

        # Check if intervention is needed
        if self._should_intervene():
            # Generate corrections
            actions = self.generator.generate_corrections(
                metrics,
                task,
                current_loops=self._estimate_current_loops(metrics),
                current_strategy=strategy
            )

            if actions:
                prioritized = self.generator.prioritize_actions(actions)
                top_action = prioritized[0]

                # Record history
                self.correction_history.append(top_action)

                # Apply immediate adjustments
                self._apply_correction(top_action)

                self.logger.info(
                    f"Intervention: {top_action.action_type} (priority={top_action.priority})"
                )
                return top_action

        return None

    def _should_intervene(self) -> bool:
        """Decide whether to intervene based on recent performance"""
        # Immediate intervention for the latest failed execution
        if self.recent_metrics and not self.recent_metrics[-1].success:
            return True

        if len(self.recent_metrics) < 5:
            return False  # not enough data for trend analysis

        # Calculate recent failure rate
        recent = self.recent_metrics[-10:]  # last 10
        failure_rate = sum(1 for m in recent if not m.success) / len(recent)

        if failure_rate > 0.4:
            return True

        # Check for degrading quality
        if len(recent) >= 5:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]
            avg_first = mean(m.overall_score() for m in first_half)
            avg_second = mean(m.overall_score() for m in second_half)
            if avg_second < avg_first * 0.7:
                return True

        return False

    def _estimate_current_loops(self, metrics: QualityMetrics) -> int:
        """Estimate current loop count from metrics"""
        return metrics.reasoning_depth

    def _apply_correction(self, action: CorrectionAction):
        """Apply correction to controller state"""
        if action.recommended_loops:
            self.loop_multiplier = action.recommended_loops / 8.0  # assuming default 8

        if action.strategy_override:
            self.strategy_override = action.strategy_override

        if 'temperature' in action.tool_adjustments:
            self.temperature_adjustment = action.tool_adjustments['temperature']

    def get_current_adjustments(self) -> Dict[str, Any]:
        """Get current parameter adjustments"""
        return {
            'loop_multiplier': self.loop_multiplier,
            'temperature_adjustment': self.temperature_adjustment,
            'strategy_override': self.strategy_override
        }

    def reset_adjustments(self):
        """Reset adjustments to defaults"""
        self.loop_multiplier = 1.0
        self.temperature_adjustment = 0.0
        self.strategy_override = None


# ============ Confidence Tracker ============

class ConfidenceTracker:
    """
    Track confidence over time and detect drift
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: List[float] = []
        self.logger = logging.getLogger("ConfidenceTracker")

    def record(self, confidence: float):
        """Record a confidence score"""
        self.history.append(confidence)
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    def get_trend(self) -> float:
        """Get confidence trend (positive = improving)"""
        if len(self.history) < 5:
            return 0.0

        # Simple linear regression
        n = len(self.history)
        x_mean = (n - 1) / 2
        y_mean = sum(self.history) / n

        numerator = sum(i * y for i, y in enumerate(self.history))
        numerator -= n * x_mean * y_mean

        denominator = sum(i * i for i in range(n))
        denominator -= n * x_mean * x_mean

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope

    def is_declining(self, threshold: float = -0.01) -> bool:
        """Check if confidence is declining"""
        trend = self.get_trend()
        return trend < threshold

    def average_confidence(self) -> float:
        """Get average confidence over window"""
        if not self.history:
            return 0.5
        return sum(self.history) / len(self.history)


# ============ Main Self-Correction Module ============

class SelfCorrectionModule:
    """
    Main module for real-time self-correction and quality control
    """

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger("SelfCorrection")

        # Convert Pydantic model to dict (v2 compatible)
        config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict() if hasattr(config, 'dict') else config

        # Components
        self.evaluator = QualityEvaluator(config_dict)
        self.controller = AdaptiveController(config_dict)
        self.confidence_tracker = ConfidenceTracker(window_size=20)

        # Statistics
        self.corrections_applied = 0
        self.intervention_success_rate = 0.0

    def process_execution(
        self,
        trace: Any,
        task: str,
        strategy: str,
        tools_used: List[str],
        expected_tools: Optional[List[str]] = None
    ) -> Tuple[QualityMetrics, Optional[CorrectionAction]]:
        """
        Process execution result and generate corrections if needed

        Args:
            trace: Reasoning trace
            task: The task being executed
            strategy: Reasoning strategy used
            tools_used: Tools that were used
            expected_tools: Tools that should have been used

        Returns:
            (metrics, correction_action)
        """
        # Evaluate quality
        metrics = self.evaluator.evaluate(trace, tools_used, expected_tools)

        # Track confidence
        self.confidence_tracker.record(metrics.confidence_score)

        # Generate corrections
        correction = self.controller.observe(metrics, task, strategy)

        if correction:
            self.corrections_applied += 1
            self.logger.info(
                f"Correction generated: {correction.action_type} "
                f"(priority={correction.priority}, confidence={correction.confidence:.2f})"
            )

        return metrics, correction

    def get_status_report(self) -> Dict[str, Any]:
        """Get current self-correction system status"""
        return {
            'total_corrections': self.corrections_applied,
            'recent_confidence': self.confidence_tracker.average_confidence(),
            'confidence_trend': self.confidence_tracker.get_trend(),
            'current_adjustments': self.controller.get_current_adjustments(),
            'window_size': len(self.confidence_tracker.history)
        }

    def should_trigger_retraining(self) -> Tuple[bool, str]:
        """
        Check if performance degradation warrants retraining
        Returns (should_retrain, reason)
        """
        recent = self.controller.recent_metrics[-20:] if len(self.controller.recent_metrics) >= 20 else []

        if len(recent) < 10:
            return False, "Insufficient data"

        failure_rate = sum(1 for m in recent if not m.success) / len(recent)
        if failure_rate > 0.6:
            return True, f"High failure rate: {failure_rate:.1%}"

        avg_score = mean(m.overall_score() for m in recent)
        if avg_score < 0.3:
            return True, f"Low quality score: {avg_score:.2f}"

        if self.confidence_tracker.is_declining(threshold=-0.02):
            return True, "Confidence declining trend detected"

        return False, "Performance acceptable"


# Convenience functions

def create_self_correction_module(config: Any) -> SelfCorrectionModule:
    """Factory function"""
    return SelfCorrectionModule(config)
