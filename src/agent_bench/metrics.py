"""Aggregate metrics, Wilson confidence intervals and leaderboard ordering.

All reported rates are proportions with a Wilson score interval, computed on the
task-pass indicator (a task counts as passed only when every grader passes).
Wilson is used instead of the normal approximation because suites are small and
rates of 0% or 100% would otherwise get a zero-width, meaningless interval.

The scoring formula is intentionally published and simple:

    task_score     = sum(passed grader weight) / sum(grader weight)        in [0, 1]
    task_passed    = 1 if every grader passed else 0
    success_rate   = sum(task_passed * task.weight) / sum(task.weight)
    efficiency     = total tool calls / tasks
    wasted_rate    = failed calls / total tool calls
    recovery_rate  = recovered faults / injected faults
    schema_rate    = schema violating calls / total tool calls
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .models import CATEGORIES, EpisodeRecord

Z_95 = 1.959963984540054


@dataclass
class Interval:
    """A Wilson score interval for a binomial proportion."""

    successes: int
    total: int
    low: float
    high: float
    estimate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "successes": self.successes,
            "total": self.total,
            "estimate": round(self.estimate, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
        }

    def render(self) -> str:
        return f"{self.estimate * 100:.1f}% [{self.low * 100:.1f}, {self.high * 100:.1f}]"


def wilson(successes: int, total: int, z: float = Z_95) -> Interval:
    """Wilson score interval for successes/total; (0, [0, 0]) when total is 0."""

    if total <= 0:
        return Interval(0, 0, 0.0, 0.0, 0.0)
    if successes < 0 or successes > total:
        raise ValueError(f"successes ({successes}) must be between 0 and total ({total})")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return Interval(successes, total, max(0.0, centre - margin), min(1.0, centre + margin), p)


@dataclass
class CategoryStats:
    category: str
    episodes: int = 0
    passed: int = 0
    weighted_score: float = 0.0
    weighted_total: float = 0.0
    calls: int = 0
    failed_calls: int = 0
    interval: Interval | None = None

    @property
    def success_rate(self) -> float:
        return 0.0 if self.episodes == 0 else self.passed / self.episodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "episodes": self.episodes,
            "passed": self.passed,
            "success_rate": round(self.success_rate, 6),
            "mean_score": round(
                self.weighted_score / self.weighted_total if self.weighted_total else 0.0, 6
            ),
            "ci": self.interval.to_dict() if self.interval else None,
            "calls": self.calls,
            "wasted_calls": self.failed_calls,
        }


@dataclass
class PolicyReport:
    """Everything measured for one (suite, adapter) pair."""

    policy: str
    suite: str
    repeats: int
    episodes: list[EpisodeRecord] = field(default_factory=list)
    categories: dict[str, CategoryStats] = field(default_factory=dict)

    # -- aggregate ------------------------------------------------------------
    @property
    def task_count(self) -> int:
        return len({episode.task_id for episode in self.episodes})

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def passed_episodes(self) -> int:
        return sum(1 for episode in self.episodes if episode.passed)

    @property
    def success_rate(self) -> float:
        return 0.0 if not self.episodes else self.passed_episodes / len(self.episodes)

    @property
    def mean_score(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(episode.score for episode in self.episodes) / len(self.episodes)

    @property
    def total_calls(self) -> int:
        return sum(len(episode.calls) for episode in self.episodes)

    @property
    def failed_calls(self) -> int:
        return sum(episode.failed_calls for episode in self.episodes)

    @property
    def schema_violations(self) -> int:
        return sum(episode.schema_violations for episode in self.episodes)

    @property
    def injected_faults(self) -> int:
        return sum(episode.injected_faults for episode in self.episodes)

    @property
    def recovered_faults(self) -> int:
        return sum(episode.recovered_faults for episode in self.episodes)

    @property
    def efficiency(self) -> float:
        return 0.0 if not self.episodes else self.total_calls / len(self.episodes)

    @property
    def wasted_rate(self) -> float:
        return 0.0 if not self.total_calls else self.failed_calls / self.total_calls

    @property
    def schema_rate(self) -> float:
        return 0.0 if not self.total_calls else self.schema_violations / self.total_calls

    @property
    def recovery_rate(self) -> float:
        if not self.injected_faults:
            return 0.0
        return self.recovered_faults / self.injected_faults

    @property
    def interval(self) -> Interval:
        return wilson(self.passed_episodes, self.episode_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "suite": self.suite,
            "repeats": self.repeats,
            "tasks": self.task_count,
            "episodes": self.episode_count,
            "passed_episodes": self.passed_episodes,
            "success_rate": round(self.success_rate, 6),
            "success_ci": self.interval.to_dict(),
            "mean_score": round(self.mean_score, 6),
            "efficiency": round(self.efficiency, 6),
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "wasted_rate": round(self.wasted_rate, 6),
            "schema_violations": self.schema_violations,
            "schema_rate": round(self.schema_rate, 6),
            "injected_faults": self.injected_faults,
            "recovered_faults": self.recovered_faults,
            "recovery_rate": round(self.recovery_rate, 6),
            "categories": [
                stats.to_dict() for stats in self.categories.values() if stats.episodes
            ],
        }


def summarise(policy: str, suite: str, repeats: int,
              episodes: Iterable[EpisodeRecord]) -> PolicyReport:
    """Fold a flat list of episodes into a report."""

    report = PolicyReport(policy=policy, suite=suite, repeats=repeats)
    report.episodes = list(episodes)
    report.categories = {category: CategoryStats(category=category) for category in CATEGORIES}
    for episode in report.episodes:
        stats = report.categories.setdefault(
            episode.category, CategoryStats(category=episode.category)
        )
        stats.episodes += 1
        stats.passed += 1 if episode.passed else 0
        stats.weighted_score += episode.score
        stats.weighted_total += 1.0
        stats.calls += len(episode.calls)
        stats.failed_calls += episode.failed_calls
    for stats in report.categories.values():
        stats.interval = wilson(stats.passed, stats.episodes)
    return report


def leaderboard(reports: list[PolicyReport]) -> list[PolicyReport]:
    """Order policies by success rate, then mean score, then efficiency.

    Efficiency is the tie breaker in the *right* direction: fewer calls wins.
    """

    return sorted(
        reports,
        key=lambda report: (
            -report.success_rate,
            -report.mean_score,
            report.efficiency,
            report.policy,
        ),
    )


def compare_notes(careful: PolicyReport, naive: PolicyReport) -> list[str]:
    """Explain, per category, where the two scripted personas diverge."""

    notes: list[str] = []
    for category in CATEGORIES:
        good = careful.categories.get(category)
        bad = naive.categories.get(category)
        if not good or not bad or not good.episodes or not bad.episodes:
            continue
        delta = good.success_rate - bad.success_rate
        if abs(delta) < 1e-9:
            notes.append(
                f"{category}: no separation "
                f"({good.interval.render()} vs {bad.interval.render()})"
            )
        else:
            notes.append(
                f"{category}: {good.success_rate * 100:.0f}% vs {bad.success_rate * 100:.0f}% "
                f"(delta {delta * 100:+.0f} points)"
            )
    return notes